from abc import ABC, abstractmethod

from ..common_imports import *
from ..util.common_ut import get_uid, extract_uniques
from ..util.shell_ut import ask
from ..core.bases import (
    Call, Operation, ValueRef, summarize_calls, summarize_vref_locs, 
    group_calls_by_op
)
from ..core.config import CoreConsts, CoreConfig, EnvConfig, CALLS
if EnvConfig.has_rich:
    import rich
    from rich.syntax import Syntax
from ..core.idx import BuiltinOpClasses
from ..core.impl import SimpleFunc
from ..core.sig import BaseSignature

from ..storages.kv import KVStore
from ..storages.kv_impl.joblib_impl import JoblibStorage
from ..storages.calls import (
    BaseCallStorage, PartitionedCallStorage, CallLocation, 
    PartitionedCallLocation
)
from ..storages.objects import BaseObjStorage, ObjStorage, BaseObjLocation
from ..storages.relations import RelStorage, Connection
from ..storages.rel_impl.utils import transaction
from ..storages.rel_impl.sqlite_impl import SQLiteRelStorage
from ..storages.call_graph import BaseCallGraphStorage, CallGraphStorage
if EnvConfig.has_psql:
    from ..storages.rel_impl.psql_impl import PSQLRelStorage

from ..adapters.rels import BaseRelAdapter, RelAdapter, Prov
from ..adapters.vals import BaseValAdapter, ValAdapter
from ..adapters.calls import BaseCallAdapter, CallAdapter
from ..adapters.ops import OpAdapter, BaseOpAdapter
from ..queries.rel_weaver import ValQuery, OpQuery
from ..queries.compiler import BaseCompiler, Compiler
from ..queries.graph_weaver import print_history

UID_COL = CoreConsts.UID_COL

class Storage(object):

    def __init__(
        self, 
        ### fs storage
        in_memory:bool=False,
        fs_rel:str=None,
        fs_abs:Path=None,
        obj_kv:TType[KVStore]=None,
        call_kv:TType[KVStore]=None,
        ### rel storage
        db_backend:str=CoreConfig.db_backend,
        # for psql
        psql_name:str=None, 
        # for sqlite
        sqlite_rel:str=None, # a relative path to desired sqlite file
        sqlite_abs:Path=None,
    ):
        ### figure out db backend
        if psql_name is not None:
            assert sqlite_rel is None
            db_backend = 'psql'
        elif sqlite_rel is not None:
            assert psql_name is None 
            db_backend = 'sqlite'
        else:
            db_backend = db_backend
        ### produce absolute path for fs storage
        if fs_abs is None:
            if in_memory: # todo: disable use of any files for this
                assert fs_rel is None
                fs_abs = Path(tempfile.gettempdir()) / get_uid()
                _is_temp = True
            else:
                assert CoreConfig.fs_storage_root is not None
                if fs_rel is None:
                    fs_rel = get_uid()
                fs_abs = CoreConfig.fs_storage_root / fs_rel
                _is_temp = False
        else:
            _is_temp = False
        self.fs_abs = fs_abs
        ### create object and call storages
        self._in_memory = in_memory
        self._obj_st = ObjStorage(
            root=fs_abs / 'obj',
            DefaultKVClass=obj_kv,
            transient=self.in_memory
        )
        self._call_st = PartitionedCallStorage(
            root=fs_abs / 'call',
            default_kv_class=call_kv,
            transient=self.in_memory
        )
        self._call_graph_st = CallGraphStorage(root=fs_abs / 'call_graph')
        ### create relational storage
        if db_backend == 'psql':
            assert EnvConfig.has_psql
            if psql_name is None:
                psql_name = RelStorage.generate_db_name()
            rels_exists = PSQLRelStorage.exists_db(db_name=psql_name)
            self._rel_st = PSQLRelStorage(db_name=psql_name, 
                                          psql_config=CoreConfig.psql)
        elif db_backend == 'sqlite':
            # TODO: in-memory sqlite
            if sqlite_abs is None: # produce absolute path for sqlite
                if in_memory: # temp location
                    sqlite_abs = (Path(tempfile.gettempdir()) /
                                  get_uid() / 'db.sqlite')
                else:
                    assert fs_abs is not None
                    if sqlite_rel is None:
                        if _is_temp:
                            sqlite_dirname = get_uid()
                        else:
                            sqlite_dirname = 'rels'
                        sqlite_rel_path = Path(sqlite_dirname) / 'db.sqlite'
                    else:
                        sqlite_rel_path = Path(sqlite_rel)
                    assert not sqlite_rel_path.is_absolute()
                    sqlite_abs = fs_abs / sqlite_rel_path
            rels_exists = SQLiteRelStorage.exists_db(path=sqlite_abs)
            self._rel_st = SQLiteRelStorage(path=sqlite_abs)
        else:
            raise NotImplementedError()

        ### link adapters to storages
        self._val_adapter = ValAdapter(obj_storage=self.obj_st) 
        self._rel_adapter = RelAdapter(
            rel_storage=self.rel_st,
            val_adapter=self.val_adapter,
            call_graph_storage=self.call_graph_st
        )
        self.rel_adapter.init(first_time=(not rels_exists))
        self._call_adapter = CallAdapter(
            call_storage=self.call_st,
            val_adapter=self.val_adapter
        )
        self.block_subtransactions = False

        ### container of connected ops
        self._synchronizer = Synchronizer()

        ### deprecated: op tracking
        # if sig_tracking_abs is not None: 
        #     assert sig_tracking_defs_path is not None
        # self._sig_tracking_abs = sig_tracking_abs
        # self._sig_tracking_defs_path = sig_tracking_defs_path

    ############################################################################ 
    ### synchronization
    ############################################################################ 
    @property
    def synchronizer(self) -> 'Synchronizer':
        return self._synchronizer
    
    @transaction()
    def synchronize_many(self, funcops:TList['FuncUIBase'],
                         conn:Connection=None) -> TList[SimpleFunc]:
        ops = [f.op for f in funcops]
        (synced_ops, op_updates, created_ops,
         autoupdated_superops) = self.rel_adapter.synchronize_many(
             ops=ops, conn=conn)
        # show summary of changes
        for op_update in op_updates:
            if not op_update.empty:
                op_update.show()
            else:
                logging.info(f'RECONNECT OPERATION "{op_update.ui_name}"')
        for created_op in created_ops:
            logging.info(f'CREATE OPERATION "{created_op.ui_name}" (VERSION={created_op.version}):')
            logging.info(f'{created_op}')
        for f, synced_op in zip(funcops, synced_ops):
            self.synchronizer.connect(op=synced_op, func_ui=f)
            f.set_storage(storage=self)
        for ui_name, f in self.synchronizer.funcs.items():
            if ui_name in autoupdated_superops:
                f.invalidate(reason='A newer version of this operation '
                'was created automatically as a result of '
                'a version change of one of the operations called inside it')
        return synced_ops

    @transaction()
    def synchronize(self, funcop:'FuncUIBase', 
                    conn:Connection=None) -> SimpleFunc:
        """
        Issue a synchronized copy of this operation. Only synchronized operation
        objects can be used for storage-related actions.
        """
        res = self.synchronize_many(funcops=[funcop], conn=conn)
        return res
    
    @transaction()
    def rename_func(self, func_ui:'FuncUIBase', new_name:str, 
                    conn:Connection=None):
        """
        Rename an operation and
            - invalidate this interface;
            - disconnect from this storage.
            
        To use the operation, you must re-connect a version with the correct
        name.
        """
        op = func_ui.op
        original_name = op.ui_name
        version = op.version
        if not self.synchronizer.is_connected(op):
            raise ValueError()
        # update internal state
        self.rel_adapter.rename_op(op=op, new_name=new_name, conn=conn)
        # invalidate
        reason = '\n'.join([f'Operation was renamed from {original_name} to {new_name}.',
                            f'To use this operation, change the name in the function definition and re-connect it'])
        func_ui.invalidate(reason=reason)
        # disconnect
        self.synchronizer.disconnect(ui_name=original_name)
        logging.info(f'RENAME OPERATION {original_name} (VERSION={version}):\n{original_name} ---> {new_name}')

    @transaction()
    def rename_args(self, func_ui:'FuncUIBase', mapping:TDict[str, str], conn:Connection=None):
        """
        Rename arguments of an operation and
            - invalidate this interface;
            - disconnect from this storage.

        To use the operation, you must re-connect a version with the correct
        signature.
        """
        op = func_ui.op
        ui_name = op.ui_name
        if not self.synchronizer.is_connected(op):
            raise ValueError()
        # update internal state
        self.rel_adapter.rename_op_args(op=op, mapping=mapping, conn=conn)
        # invalidate
        reason = '\n'.join([
            f'\nOperation inputs were renamed by mapping {mapping}.',
            f'To use this operation, rename arguments in the function definition and re-connect it.'
        ])
        func_ui.invalidate(reason=reason)
        # disconnect
        self.synchronizer.disconnect(ui_name=ui_name)
        rename_summary = '\n'.join([f'{k}    --->    {v}' for k, v in mapping.items()])
        logging.info(f'RENAME ARGUMENTS OF OPERATION "{ui_name}" (VERSION={op.version}):\n{rename_summary}')
    
    @property
    def stored_signatures(self) -> TDict[TTuple[str, str], TList[BaseSignature]]:
        res = {}
        builtin_ui_names = [BuiltInClass().ui_name 
                            for BuiltInClass in BuiltinOpClasses]
        for (ui_name, version), sigmap in self.op_adapter.sigmaps.items():
            if ui_name not in builtin_ui_names:
                res[ui_name, version] = sigmap.source
        return res

    @property
    def connected_ops(self) -> TDict[str, SimpleFunc]:
        return self.synchronizer.ops

    def is_connected(self, f:'FuncUIBase') -> bool:
        return self.synchronizer.is_connected(f=f)
    
    ### deprecated: op tracking
    # @property
    # def is_tracked(self) -> bool:
    #     return (self._sig_tracking_abs is not None)

    ############################################################################ 
    ### composition
    ############################################################################ 
    @property
    def in_memory(self) -> bool:
        return self._in_memory

    def get_engine(self):
        return self.rel_st.get_engine()

    @property
    def call_st(self) -> PartitionedCallStorage:
        return self._call_st
    
    @property
    def call_graph_st(self) -> BaseCallGraphStorage:
        return self._call_graph_st
    
    @property
    def call_adapter(self) -> BaseCallAdapter:
        return self._call_adapter
    
    @property
    def op_adapter(self) -> BaseOpAdapter:
        return self.rel_adapter.op_adapter
    
    @property
    def obj_st(self) -> BaseObjStorage:
        return self._obj_st

    @property
    def rel_st(self) -> RelStorage:
        return self._rel_st
    
    @property
    def rel_adapter(self) -> BaseRelAdapter:
        return self._rel_adapter
    
    @property
    def val_adapter(self) -> BaseValAdapter:
        return self._val_adapter

    @property
    def value_impl_idx(self):
        return ValueIndex
    
    @property
    def op_impl_idx(self):
        return OpIndex

    ############################################################################ 
    ### convenience interfaces to objects
    ############################################################################ 
    def attach(self, vref:ValueRef, shallow:bool=False):
        self.val_adapter.attach(vref=vref, shallow=shallow)
    
    def where_is(self, vref:ValueRef) -> BaseObjLocation:
        return self.val_adapter.get_vref_location(vref=vref)
    
    def where_are(self, vrefs:TList[ValueRef]) -> TList[BaseObjLocation]:
        return [self.val_adapter.get_vref_location(vref=vref) for vref in vrefs]

    def get(self, loc:BaseObjLocation, lazy:bool=False) -> ValueRef:
        return self.val_adapter.get(loc=loc, lazy=lazy)
    
    def mget(self, locs:TList[BaseObjLocation], lazy:bool=False) -> TList[ValueRef]:
        return self.val_adapter.mget(locs=locs, lazy=lazy)

    ############################################################################ 
    ### buffers
    ############################################################################ 
    def make_buffer(self, name:str=None, transient:bool=True) -> 'Buffer':
        """
        Create a new buffer to hold results of computations. May be either
        in-memory or persisted.
        """
        if name is None:
            name = get_uid()
        call_st = PartitionedCallStorage(root=self.fs_abs / f'{name}_calls',
                                         transient=transient)
        obj_st = ObjStorage(root=self.fs_abs / f'{name}_objs',
                            transient=transient)
        return Buffer(call_st=call_st, obj_st=obj_st, name=name)
    
    def commit_buffer(self, buffer:'Buffer'):
        """
        Commit the call and value data from a buffer to persistent storage
        """
        logging.info(f'Committing buffer {buffer.name}...')
        call_st, val_adapter = buffer.call_st, buffer.val_adapter
        ### transfer calls
        call_locs = call_st.locs()
        detached_calls = call_st.mget(locs=call_locs)
        self.call_st.mset_if_not_exists(locs=call_locs, call_datas=detached_calls)
        call_st.mdelete(locs=call_locs)
        ### transfer objects
        obj_locs = val_adapter.obj_storage.locs()
        objs, metas = val_adapter.obj_storage.mget(locs=obj_locs)
        self.obj_st.mset(mapping={loc: obj for loc, obj in zip(obj_locs, objs)}, 
                         meta_mapping={loc: meta 
                                       for loc, meta in zip(obj_locs, metas)})
        val_adapter.obj_storage.mdelete(locs=obj_locs)
    
    ############################################################################ 
    ### queries
    ############################################################################ 
    def make_compiler(self, val_queries:TTuple[ValQuery,...], 
                      op_queries:TTuple[OpQuery,...]) -> BaseCompiler:
        return Compiler(rels_adapter=self.rel_adapter,
                        vals_adapter=self.val_adapter, 
                        val_queries=val_queries, op_queries=op_queries)
    
    ############################################################################ 
    ### committing
    ############################################################################ 
    @transaction()
    def commit(self, partition:str=CALLS.default_temp_partition, conn:Connection=None):
        call_storage = self.call_st
        assert isinstance(call_storage, PartitionedCallStorage)
        rels_adapter = self.rel_adapter
        locs = call_storage.locs(partitions=[partition])
        num_locs = len(locs)
        logging.debug(f'commit(): Committing {len(locs)} locations from partition {partition}...')
        if len(locs) != 0:
            detached_calls:TList[Call] = call_storage.mget(locs=locs, 
                                                           allow_fallback=False)
            #! filter away calls to unknown ops
            # get the known (internal name, version) combinations
            known_ops_data = [(self.op_adapter.ui_to_name[ui_name], version) 
                              for ui_name, version in self.op_adapter.sigmaps.keys()]
            locs_for_existing_ops = [
                loc for loc, call in zip(locs, detached_calls) if
                (call.op.name, call.op.version) in known_ops_data
            ]
            detached_calls_for_existing_ops = [
                call for call in detached_calls if
                (call.op.name, call.op.version) in known_ops_data
            ]
            logging.debug(f'commit(): Got call data')
            rels_adapter.upsert_calls(calls=detached_calls_for_existing_ops,
                                      conn=conn)
            new_locs = [
                loc.moved(new_partition=CALLS.main_partition) for
                loc in locs_for_existing_ops
            ]
            call_storage.mset_if_not_exists(
                locs=new_locs,
                call_datas=detached_calls_for_existing_ops
            )
            logging.debug(f'commit(): Locations moved to main partition')
            #! delete ALL calls (including those to unknown ops)
            call_storage.mdelete(locs=locs)
            logging.debug(f'commit(): Locations deleted from partition {partition}')
            if CoreConfig.verbose_commits:
                df = summarize_calls(
                    calls=detached_calls_for_existing_ops,
                    internal_to_ui=self.op_adapter.internal_to_ui
                )
                logging.info(f'SUMMARY OF COMMITED CALLS:\n{df}')
        logging.info(f'Committed {num_locs} calls from partition {partition}')
    
    ############################################################################ 
    ### provenance
    ############################################################################ 
    @transaction()
    def explain(self, vref:ValueRef, method:str=Prov.back_shortest, 
                conn:Connection=None):
        raise NotImplementedError()
        call_uids, _ = self.rel_adapter.dependency_traversal(vref_uids={vref.uid}, call_uids=None, method=method, conn=conn)
        locs = [PartitionedCallLocation(uid=uid) for uid in call_uids]
        calls = self.call_st.mget(locs=locs, allow_fallback=False)
        s = print_history(calls=calls, op_adapter=self.op_adapter, val_adapter=self.val_adapter)
        return Syntax(s, lexer='python')

    ############################################################################ 
    ### deletions
    ############################################################################ 
    @ask(question='Are you sure you want to drop all instance data for this storage?',
         desc_getter=lambda x: x.describe())
    def drop_instance_data(self, answer:bool=None):
        self.call_st.delete_all(answer=answer)
        self.obj_st.delete_all(answer=answer)
        self.rel_st.drop_all_rows(answer=answer)
    
    @transaction()
    def reset_func(self, f:'FuncUIBase', version:str=None, 
                   conn:Connection=None):
        """
        Delete all calls to this operation and their dependents (without
        removing the operation itself). 
        
        NOTE: 
            - this does not delete internal calls of a superop 
        """
        op = f.op
        if version is None:
            version = op.version
        ### delete calls from objects
        call_uids = self.rel_adapter.get_call_uids(op=op, version=version, 
                                                   conn=conn)
        partitions = self.call_st.lookup_partitions(uids=call_uids)
        locs = [PartitionedCallLocation(uid=uid, partition=partition) 
                for uid, partition in zip(call_uids, partitions) 
                if partition is not None]
        calls = self.call_st.mget(locs=locs, allow_fallback=False)
        self.delete_with_dependents(calls=calls, verbose=None, conn=conn)
        
    @transaction()
    def drop_func(self, f:'FuncUIBase', conn:Connection=None):
        op = f.op
        if not self.synchronizer.is_connected(f=op):
            raise ValueError('Can only drop synchronized operations')
        self.reset_func(f=f, conn=conn)
        self.rel_adapter.drop_op_relation(op=op, conn=conn)
        self.call_graph_st.delete_node(name=f.op.qualified_name)
        self.synchronizer.disconnect(ui_name=op.ui_name)
        reason = f'This operation was dropped from storage'
        f.invalidate(reason=reason)
        logging.info(f'DROP OPERATION {op.ui_name} (VERSION={op.version}):\n{op}')
    
    @transaction()
    def _drop_vrefs(self, locs:TList[BaseObjLocation],
                   verbose:bool=None, conn:Connection=None):
        """
        Drop these locations from relations and objects. 
        """
        if verbose is None:
            verbose = CoreConfig.verbose_deletes
        # delete from relations
        self.rel_adapter.drop_vrefs(locs=locs, conn=conn)
        # delete from objects
        self.obj_st.mdelete(locs=locs)
        summary_df = summarize_vref_locs(locs=locs)
        deletion_msg = f'SUMMARY OF DELETED VALUES:\n{summary_df}'
        logging.info(deletion_msg)

    @transaction()
    def _drop_calls(self, locs:TList[CallLocation], allow_fallback:bool=True,
                   verbose:bool=None, conn:Connection=None):
        """
        Given a list of call locations, remove them from both relations and call
        storage.
        """
        if verbose is None:
            verbose = CoreConfig.verbose_deletes
        detached_calls = self.call_st.mget(locs=locs, allow_fallback=allow_fallback)
        ### delete from relations
        calls_by_op = group_calls_by_op(calls=detached_calls, by='op')
        for op, calls in calls_by_op.items():
            self.rel_adapter.drop_calls(op=op, call_uids=[call.uid for call in calls], conn=conn)
        ### delete from call storage
        if allow_fallback:
            partitions = self.call_st.lookup_partitions(uids=[call.uid for call in detached_calls])
            actual_locs = [loc.moved(new_partition=partition) for loc, partition in zip(locs, partitions)]
        else:
            actual_locs = locs
        self.call_st.mdelete(locs=actual_locs)
        if verbose:
            summary_df = summarize_calls(calls=detached_calls, internal_to_ui=self.op_adapter.internal_to_ui)
            deletion_msg = f'SUMMARY OF DELETED CALLS:\n{summary_df}'
            logging.info(deletion_msg)
        
    def get_active_temp_partitions(self) -> TList[str]:
        partitions = [p for p in self.call_st.partitions() if p != CALLS.main_partition]
        return [p for p in partitions if not self.call_st.is_empty(partition=p)]
    
    @transaction()
    def delete_with_dependents(self, calls:TList[Call], 
                               verbose:bool=None, conn:Connection=None):
        # first, check if there are uncommitted calls 
        active_temp_partitions = self.get_active_temp_partitions()
        if active_temp_partitions:
            raise RuntimeError(f'Deletion not allowed: there are uncommitted '
                               f'calls (partitions: {active_temp_partitions}). '
                               'Commit these calls first')
        if not calls:
            logging.info(f'No calls to delete')
            return
        # remove duplicates
        calls, _, _ = extract_uniques(objs=calls,
                                      keys=[call.uid for call in calls])
        # this is the correct way to delete
        call_uids = [call.uid for call in calls]
        vref_deps_uids, call_deps_uids = self.rel_adapter.get_call_dependents(
            call_uids=set(call_uids), conn=conn)
        # drop calls
        loc_getter = lambda uid: self.call_adapter.get_location(
            uid=uid, metadata={'partition': CALLS.main_partition})
        call_locs = [loc_getter(uid) for uid in call_deps_uids]
        # drop calls
        mexist_mask = self.call_st.mexists(call_locs, allow_fallback=False)
        existing_locs = [loc for loc, exists in 
                         zip(call_locs, mexist_mask) if exists]
        self._drop_calls(locs=existing_locs, allow_fallback=False,
                         verbose=verbose, conn=conn)
        # delete vrefs
        vref_locs = self.rel_adapter.get_vref_locs_from_uids(
            vref_uids=list(vref_deps_uids), conn=conn)
        self._drop_vrefs(locs=vref_locs, verbose=verbose, conn=conn)
    
    @transaction()
    def commit_deletions(self, captured_calls:TList[Call], 
                         verbose:bool=None, conn:Connection=None):
        """
        Delete calls captured in a context, as well as (optionally) their
        outputs, IF they are orphaned.
        """
        self.delete_with_dependents(calls=captured_calls, verbose=verbose,
                                    conn=conn)

    def drop_uncommitted_calls(self):
        # todo: this is very inefficient. Instead, delete for each partition
        committed_uids = set(self.rel_adapter.get_committed_call_uids())
        call_locs = self.call_st.locs()
        uncommitted_locs = [loc for loc in call_locs 
                            if loc.uid not in committed_uids]
        logging.info(f'Deleting {len(uncommitted_locs)} uncommitted calls...')
        self.call_st.mdelete(locs=uncommitted_locs)
    
    def cleanup(self):
        self.rel_adapter.drop_rel_orphans()
        self.rel_adapter.drop_uncommitted_vrefs()
        self.drop_uncommitted_calls()

    ############################################################################ 
    ### verification
    ############################################################################ 
    def verify_rels(self):
        verify_rels(call_st=self.call_st, rel_adapter=self.rel_adapter)
    
    def verify_static(self):
        """
        Verify the static invariants for this storage
        """
        val_adapter = self.val_adapter
        call_adapter = self.call_adapter
        call_adapter.verify_signatures()
        val_adapter._verify_get()
        self.verify_rels()
    
    def describe(self) -> str:
        return f'Storage(fs_abs={self.fs_abs})'
    
    @ask(question='Are you sure?', desc_getter=lambda x: x.describe())
    def drop(self, answer:bool=None):
        self.rel_st.drop_db(answer=answer)
        self.obj_st.drop(answer=answer)
        self.call_st.drop(answer=answer)

    ############################################################################ 
    ### optimization
    ############################################################################ 
    def get_obj_kvs(self) -> TTuple[TDict[str, KVStore], TDict[str, KVStore]]:
        """
        Return (main kvs, meta kvs) dicts by partition.
        """
        obj_st:ObjStorage = self.obj_st
        res_main = {}
        res_meta = {}
        for partition in self.obj_st.partitions():
            if partition in obj_st._main_kvs.keys():
                main_kv = obj_st._main_kvs.get(k=partition)
                res_main[partition] = main_kv
            if partition in obj_st._meta_kvs.keys():
                meta_kv = obj_st._meta_kvs.get(k=partition)
                res_meta[partition] = meta_kv
        return res_main, res_meta
    
    def get_call_kvs(self) -> TDict[str, KVStore]:
        res = {}
        call_st:PartitionedCallStorage = self.call_st
        for partition in self.call_st.partitions():
            if partition in call_st._kvs.keys():
                call_kv = call_st._kvs.get(k=partition)
                res[partition] = call_kv
        return res
    
    def get_constituent_kvs(self) -> TList[KVStore]:
        main_kvs, meta_kvs = self.get_obj_kvs()
        call_kvs = self.get_call_kvs()
        return list(main_kvs.values()) + list(meta_kvs.values()) + list(call_kvs.values())
    
    def parallelize_kv(self, kv:KVStore):
        if isinstance(kv, JoblibStorage):
            kv.parallel = True
            kv.progress = True
        else:
            pass
    
    def unparallelize_kv(self, kv:KVStore):
        if isinstance(kv, JoblibStorage):
            kv.parallel = False
            kv.progress = False
    
    def unparallelize_all_kvs(self):
        for kv in self.get_constituent_kvs():
            self.unparallelize_kv(kv=kv)
    
    def parallelize_all_kvs(self):
        for kv in self.get_constituent_kvs():
            self.parallelize_kv(kv=kv)
        

def check_rels_equality(df_1:pd.DataFrame, df_2:pd.DataFrame):
    assert df_1.shape == df_2.shape
    assert set(df_1.columns) == set(df_2.columns)
    ord_cols = sorted(df_1.columns)
    # make sure the columns are aligned
    df_1 = df_1[ord_cols].set_index(UID_COL)
    df_2 = df_2[ord_cols].set_index(UID_COL)
    assert set(df_1.itertuples(index=True)) == set(df_2.itertuples(index=True))

def verify_rels(call_st:BaseCallStorage, rel_adapter:BaseRelAdapter):
    """
    Check that the content of relation storage equals the relations generated
    by all stored calls.
    """
    calls = call_st.mget(call_st.locs())
    # generate tables based on call storage
    tables_from_calls = rel_adapter.calls_to_tables(calls)
    # load tables directly from relation storage
    wideform_tables = rel_adapter.op_tables + rel_adapter.vref_tables
    tables_from_rels = {k: rel_adapter.rel_storage.fast_read(name=k) 
                        for k in wideform_tables}
    assert set(tables_from_calls.keys()).issubset(set(tables_from_rels.keys()))
    for k in tables_from_calls:
        rels_df = tables_from_rels[k]
        calls_df = tables_from_calls[k]
        check_rels_equality(df_1=rels_df, df_2=calls_df)
    for k in tables_from_rels:
        if k not in tables_from_calls:
            assert tables_from_rels[k].empty


class Buffer(object):
    """
    An in-memory combination of storage components for increased performance
    """
    def __init__(self, call_st:PartitionedCallStorage, obj_st:ObjStorage, 
                 name:str):
        self.name = name
        self.call_st = call_st
        self.obj_st = obj_st
        self.val_adapter = ValAdapter(obj_storage=obj_st)
        self.call_adapter = CallAdapter(call_storage=self.call_st, 
                                        val_adapter=self.val_adapter)
        

class CallBuffer(object):
    """
    Collects (detached) calls
    """
    def __init__(self):
        self._calls:TList[Call] = []
    
    @property
    def calls(self) -> TList[Call]:
        return self._calls
    
    def unique_calls(self) -> TList[Call]:
        res, _, _ = extract_uniques(objs=self.calls, keys=[call.uid for call in self.calls])
        return res
    
    def insert(self, call:Call):
        self._calls.append(call)
        
    def minsert(self, calls:TList[Call]):
        self._calls += calls
    
    def reset(self):
        self._calls = []
    

class DeletionBuffer(CallBuffer):
    """
    Holds detached calls to be deleted from relations and call storage.
    """
    def commit_deletions(self, storage:Storage):
        # remove duplicates
        captured_calls = list({call.uid: call for call in self._calls}.values())
        storage.commit_deletions(captured_calls=captured_calls)
        self.reset()


class FuncUIBase(ABC):

    @property
    @abstractmethod
    def op(self) -> SimpleFunc:
        raise NotImplementedError()
    
    @abstractmethod
    def set_op(self, op:SimpleFunc):
        raise NotImplementedError()
    
    @abstractmethod
    def invalidate(self, reason:str):
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_invalidated(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def set_storage(self, storage:Storage):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def storage(self) -> TOption[Storage]:
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> TAny:
        raise NotImplementedError()


class Synchronizer(object):
    """
    Keeps track of at most 1 connected version of each operation. 
    """
    def __init__(self):
        # ui_name -> op object
        self.ops:TDict[str, SimpleFunc] = {}
        self.funcs:TDict[str, FuncUIBase] = {}
    
    def connect(self, op:SimpleFunc, func_ui:FuncUIBase):
        if op.ui_name in self.ops:
            self.update(op)
        else:
            self.create(op)
        func_ui.set_op(op)
        self.funcs[op.ui_name] = func_ui
        assert self.is_connected(f=op)
    
    def create(self, op:SimpleFunc):
        assert op.ui_name not in self.ops
        self.ops[op.ui_name] = op
    
    def update(self, op:SimpleFunc):
        ui_name = op.ui_name
        assert ui_name in self.ops
        if op.version != self.ops[ui_name].version:
            cur_version = self.ops[ui_name].version
            cur_op = self.ops[ui_name]
            new_version = op.version
            self.ops[ui_name] = op
            lines = [
                f'CHANGE VERSION of operation {ui_name} FROM {cur_version} TO {new_version}',
                f'PREVIOUS OPERATION: {cur_op}',
                f'NEW OPERATION: {op}'
            ]
            logging.info('\n'.join(lines))
        else:
            self.ops[ui_name] = op
    
    @property
    def connected_names(self) -> TList[str]:
        return list(self.ops.keys())
    
    def connected_version(self, ui_name:str) -> str:
        if ui_name not in self.ops:
            raise NotImplementedError()
        return self.ops[ui_name].version
    
    def exists_by_value(self, op:SimpleFunc) -> bool:
        return (op.ui_name in self.ops and 
                self.connected_version(op.ui_name) == op.version)
    
    def is_connected(self, f:TUnion[SimpleFunc, FuncUIBase]) -> bool:
        if isinstance(f, FuncUIBase):
            if f.is_invalidated:
                return False
            op = f.op
        elif isinstance(f, SimpleFunc):
            op = f
        else:
            raise NotImplementedError()
        if not self.exists_by_value(op):
            return False
        return self.ops[op.ui_name] is op
    
    def disconnect(self, ui_name:str):
        del self.ops[ui_name]
    
    def _dump_state(self):
        data = {k: v for k, v in self.__dict__.items() 
                if k != 'funcs'}
        return copy.deepcopy(data)