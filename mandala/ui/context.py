from .storage import Storage, Buffer, DeletionBuffer, FuncUIBase, CallBuffer

from ..queries.tree import BaseQTree, QTree, induced_queries

from ..common_imports import *
from ..util.common_ut import (
    get_collection_atoms, rename_dict_keys, invert_dict, group_like
)
from ..core.bases import (
    ValueRef, Operation, BaseContext, GlobalContext,
    TVRefCollection, Call)
from ..core.config import CoreConfig, EnvConfig, MODES, CALLS
from ..storages.objects import BaseObjLocation
from ..storages.calls import PartitionedCallLocation
from ..queries.weaver_bases import traverse_all
from ..queries.rel_weaver import ValQuery, OpQuery 

from ..session import *

if EnvConfig.has_ray:
    import ray

################################################################################
### context
################################################################################
class ContextError(Exception):
    pass

### specifies the state 
STATE_KEYS = ('mode', 'tag', 'storage', 'partition', 'lazy', 'allow_calls',
              'autocommit', 'autodelete', 'buffer', 'cur_tree', 'force', 
              'disable_ray', 'captured_calls', )
UPDATE_KEYS = tuple(list(STATE_KEYS) + ['buffered'])
# settings that can be transfered to a global context
TRANSFERABLE_KEYS = ('mode', 'tag', 'storage', 'partition', 'lazy',
                     'allow_calls', 'autocommit', 'autodelete', 'force',
                     'disable_ray', 'captured_calls',)
# settings which if present prevent transfer 
NONTRANSFERABLE_KEYS = ('buffer', 'storage')

CONTEXT_DEFAULTS = {
        'mode': MODES.noop,
        'tag': 'main',
        'storage': None,
        'lazy': False,
        'allow_calls': True,
        'buffer': None,
        'cur_tree': None,
        'force': False,
        'disable_ray': False,
        'captured_calls': None,
    }

def get_default(k:str) -> TAny:
    """
    Use this because some defaults are dynamic
    """
    if k == 'autocommit':
        return CoreConfig.autocommit
    elif k == 'autodelete':
        return CoreConfig.autodelete
    elif k == 'partition':
        return CALLS.default_temp_partition
    else:
        return CONTEXT_DEFAULTS[k]
    

ADMISSIBLE_MODE_CHANGES = [
    (MODES.noop, MODES.transient),
    (MODES.noop, MODES.run),
    (MODES.run, MODES.transient),
    (MODES.run, MODES.query),
    (MODES.run, MODES.delete),
    (MODES.run, MODES.capture),
    (MODES.run, MODES.query_delete),
    (MODES.query, MODES.query_delete),
]

class Context(BaseContext):
    # class variable to use for customizing default behavior
    OVERRIDES = {}

    #! all args must default to None so we know what was given.
    def __init__(self, 
                 storage:Storage=None,
                 mode:str=None,
                 tag:str=None, 
                 partition:str=None, 
                 lazy:bool=None, 
                 allow_calls:bool=None,
                 autodelete:bool=None,
                 autocommit:bool=None,
                 buffer:Buffer=None, 
                 buffered:bool=None,
                 cur_tree:QTree=None,
                 force:bool=None,
                 disable_ray:bool=None, 
                 captured_calls:CallBuffer=None):
        if buffered:
            raise NotImplementedError()

        ### STATE KEYS
        assert all(k in STATE_KEYS for k in self.OVERRIDES.keys())
        self._init_key('mode', mode)
        self._init_key('tag', tag)
        self._init_key('storage', storage)
        self._init_key('partition', partition)
        self._init_key('lazy', lazy)
        self._init_key('allow_calls', allow_calls)
        self._init_key('autocommit', autocommit)
        self._init_key('autodelete', autodelete)
        self._init_key('force', force)
        self._init_key('disable_ray', disable_ray)
        self._init_key('captured_calls', captured_calls)
        self._buffer = self.get_buffer_state_update(buffered=buffered, 
                                                    buffer=buffer)

        self._updates = None
        kwargs = {
            'mode': mode,
            'tag': tag,
            'storage': storage,
            'partition': partition,
            'lazy': lazy,
            'allow_calls': allow_calls,
            'autocommit': autocommit,
            'autodelete': autodelete,
            'buffer': buffer,
            'cur_tree': cur_tree,
            'force': force,
            'disable_ray': disable_ray,
            'captured_calls': captured_calls
        }
        self.updates = {k: v for k, v in kwargs.items() if v is not None}
        self._updates_stack = []
        self._call_stack = []

        # the deletion buffer always stays the same object throughout the life
        # of a context
        self._deletion_buffer = DeletionBuffer()
        self._funcops = []

        ### query container
        self._queries = []
        if cur_tree is not None:
            self._qtree = cur_tree.get_root()
            self._cur_tree = cur_tree
        else:
            self._qtree = QTree(parent=None)
            self._cur_tree = self._qtree

    def _set_key(self, key:str, value:TAny):
        self.__dict__[f'_{key}'] = value
    
    def _init_key(self, key:str, value:TAny=None):
        default = get_default(k=key)
        override = self.OVERRIDES.get(key, None)
        if value is None:
            if override is not None:
                value = override
            else:
                value = default
        self._set_key(key=key, value=value)
    
    @staticmethod
    def update_state(state:TDict[str, TAny], 
                     updates:TDict[str, TAny]) -> TDict[str, TAny]:
        res = {}
        res.update({k: v for k, v in state.items() if v is not None})
        res.update({k: v for k, v in updates.items() if v is not None})
        return res
            
    ############################################################################ 
    ### context attributes should be protected from accidental changes
    ############################################################################ 
    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def tag(self) -> str:
        return self._tag
    
    @property
    def storage(self) -> TOption[Storage]:
        return self._storage
    
    @property
    def partition(self) -> str:
        return self._partition
    
    @property
    def lazy(self) -> bool:
        return self._lazy
    
    @property
    def allow_calls(self) -> bool:
        return self._allow_calls
    
    @property
    def autocommit(self) -> bool:
        return self._autocommit
    
    @property
    def autodelete(self) -> bool:
        return self._autodelete
    
    @property
    def buffered(self) -> bool:
        return self._buffer is not None
    
    @property
    def buffer(self) -> TOption[Buffer]:
        return self._buffer
    
    @property
    def call_stack(self) -> TList[str]:
        return self._call_stack
    
    def push_call(self, name:str):
        self._call_stack.append(name)
        
    def pop_call(self) -> str:
        return self._call_stack.pop()
    
    @property
    def cur_tree(self) -> BaseQTree:
        return self._cur_tree

    @property
    def force(self) -> bool:
        return self._force
    
    @property
    def disable_ray(self) -> bool:
        return self._disable_ray
    
    @property
    def captured_calls(self) -> CallBuffer:
        return self._captured_calls
    
    @property
    def deletion_buffer(self) -> DeletionBuffer:
        return self._deletion_buffer
    
    @property
    def updates(self) -> TDict[str, TAny]:
        return self._updates
    
    @updates.setter
    def updates(self, v:TDict[str, TAny]):
        self._validate_updates(updates=v)
        self._updates = v
    
    def _validate_updates(self, updates:TDict[str, TAny]):
        assert set(updates.keys()).issubset(set(UPDATE_KEYS))
        if 'mode' in updates:
            assert updates['mode'] in MODES._modes
            # TODO: type checks, etc.
    
    def get_buffer_state_update(self, buffered:bool=None, buffer:Buffer=None,
                                storage:Storage=None) -> TOption[Buffer]:
        if buffered or (buffer is not None):
            # there will be a buffer
            if self.buffer is not None:
                raise NotImplementedError('Changing buffer not supported')
            if buffer is None:
                assert storage is not None
                buffer = storage.make_buffer(transient=True)
        else:
            buffer = None
        return buffer
    
    @property
    def transferable_state(self) -> TDict[str, TAny]:
        """
        The part of the state that can be applied to an existing but different
        global context
        """
        res = {k: self.__dict__[f'_{k}'] for k in TRANSFERABLE_KEYS}
        return {k: v for k, v in res.items() if v is not None}
    
    @property
    def nontransferable_state(self) -> TDict[str, TAny]:
        res = {k: self.__dict__[f'_{k}'] for k in NONTRANSFERABLE_KEYS}
        return {k: v for k, v in res.items() if v is not None}
    
    ############################################################################ 
    ### descending
    ############################################################################ 
    def get_state_updates(self, descent_updates:TDict[str, TAny],
                          ) -> TDict[str, TAny]:
        state_updates = {k: v for k, v in descent_updates.items() 
                         if k in STATE_KEYS}
        buffered, buffer = (descent_updates.get('buffered', None),
                            descent_updates.get('buffer', None))
        if buffered or (buffer is not None):
            storage = (self.storage if self.storage is not None
                       else descent_updates['storage'])
            assert storage is not None
            buffer = self.get_buffer_state_update(buffered=buffered,
                                                  buffer=buffer, 
                                                  storage=storage)
            state_updates['buffer'] = buffer
        return state_updates
        
    def validate_state_updates(self, state_updates:TDict[str, TAny]):
        """
        Before making any updates, ensure they are admissible
        """
        cur_mode = self.mode
        if 'mode' in state_updates:
            new_mode = state_updates['mode']
            if cur_mode != new_mode:
                if (cur_mode, new_mode) not in ADMISSIBLE_MODE_CHANGES:
                    raise ContextError(f'Cannot change from mode={cur_mode} to mode={new_mode}')
    
    def _backup_state(self, keys:TIter[str]) -> TDict[str, TAny]:
        res = {}
        for k in keys:
            cur_v = self.__dict__[f'_{k}']
            if k == 'cur_tree': # this has some referential state
                res[k] = cur_v
            elif k == 'storage': # this too ;(
                res[k] = cur_v
            else:
                res[k] = copy.deepcopy(cur_v)
        return res

    def _descend(self, **updates):
        """
        Apply updates to the state of this context.
        
        Args:
            - updates: a dictionary of the non-trivial updates, i.e. ones where
            the value is not None
        """
        if not GlobalContext.exists():
            GlobalContext.set(context=self)
        ### verify update keys
        if not all(k in UPDATE_KEYS for k in updates.keys()):
            raise ValueError(updates.keys())
        ### produce state changes 
        state_updates = self.get_state_updates(descent_updates=updates)
        ### check state changes before running them
        self.validate_state_updates(state_updates=state_updates)
        ### perform update
        before_update = self._backup_state(keys=state_updates.keys())
        self._updates_stack.append(before_update)
        for k, v in state_updates.items():
            self.__dict__[f'_{k}'] = v
        if self.mode == MODES.define:
            assert self.storage is not None
            self._funcops = []
        logging.debug(f'Entering context {self}')
        
    ############################################################################ 
    ### ascending
    ############################################################################ 
    def finalize_state(self, 
                       ascent_updates:TDict[str, TAny]) -> TOption[Exception]:
        """
        Here, `ascent_updates` are the old values of the state fields that got
        changed by the descent.
        """
        exc = None
        cur_mode = self.mode
        if 'buffer' in ascent_updates and ascent_updates['buffer'] is None:
            # release buffer when enclosing context has no buffer
            self.commit_buffer()
        if self.mode == MODES.define:
            try:
                self.process_definitions()
            except Exception as e:
                exc = e
        elif self.mode == MODES.delete and self.autodelete:
            try:
                self.commit_deletions()
            except Exception as e:
                exc = e
        if self.mode == MODES.query_delete and self.autodelete:
            try:
                self.commit_query_deletions()
            except Exception as e:
                exc = e
        if (self.mode == MODES.run and 
            self.autocommit and 
            (self.partition != CALLS.main_partition)):
            try:
                self.commit()
            except Exception as e:
                exc = e
        if self._depth == 1:
            self.reset_queries()
        ### deal on mode-by-mode basis
        if 'mode' in ascent_updates:
            new_mode = ascent_updates['mode']
            query_modes = (MODES.query, MODES.query_delete)
            if (cur_mode in query_modes and new_mode not in query_modes):
                # release query container when existing outermost query-style
                # context 
                self.reset_queries()
        return exc 

    def process_definitions(self):
        storage = self.storage
        funcops = self._funcops
        storage.synchronize_many(funcops=funcops)
        
    def _do_ascend(self):
        if not self._updates_stack:
            raise RuntimeError('No updates to ascend from')
        ascent_updates = self._updates_stack.pop()
        for k, v in ascent_updates.items():
            self.__dict__[f'_{k}'] = v
        ### certain things are re-set when we ascend back to the top
        if self._depth == 0:
            # unlink from global if done
            GlobalContext.reset()
            # reset tree to initial state
            self.reset_tree()
            # reset queries to initial state
            self.reset_queries()
    
    def _ascend(self):
        logging.debug('Exiting context')
        if not self._updates_stack:
            raise RuntimeError('No context to ascend from')
        ascent_updates = self._updates_stack[-1]
        exc_option = self.finalize_state(ascent_updates=ascent_updates)
        self._do_ascend()
        if exc_option is not None:
            raise exc_option

    ############################################################################ 
    ### context manager interface
    ############################################################################ 
    @property
    def _depth(self) -> int:
        return len(self._updates_stack)
        
    def __call__(self, storage:Storage=None, mode:str=None, tag:str=None, 
                 partition:str=None, autodelete:bool=None,
                 autocommit:bool=None, lazy:bool=None, allow_calls:bool=None, 
                 buffered:bool=None, buffer:Buffer=None, 
                 force:bool=None, disable_ray:bool=None, 
                 captured_calls:CallBuffer=None) -> 'Context':
        kwargs = {
            'mode': mode,
            'tag': tag,
            'storage': storage,
            'partition': partition,
            'lazy': lazy,
            'allow_calls': allow_calls,
            'autocommit': autocommit,
            'autodelete': autodelete,
            'buffered': buffered,
            'buffer': buffer,
            'force': force,
            'disable_ray': disable_ray,
            'captured_calls': captured_calls,
        }
        if GlobalContext.exists():
            global_c = GlobalContext.get()
            if global_c is not self:
                if CoreConfig.enable_autonesting:
                    nontransferable_state = self.nontransferable_state
                    if nontransferable_state:
                        raise ContextError(f'Context settings {nontransferable_state.keys()} cannot be transferred to the global context')
                    state_to_transfer = Context.update_state(
                        state=self.transferable_state, updates=kwargs
                    )
                    if state_to_transfer.get('mode', None) == MODES.noop:
                        del state_to_transfer['mode']
                    return global_c(**state_to_transfer)
                else:
                    raise ContextError(f'There is a different global context {global_c} and autonesting is disabled')
        updates = {k: v for k, v in kwargs.items() if v is not None}
        self._validate_updates(updates=updates)
        self.updates = updates
        return self

    def branch(self) -> 'Context':
        new_tree = QTree()
        self._cur_tree.add_child(tree=new_tree)
        self.updates = {'cur_tree': new_tree}
        return self
    
    def __enter__(self) -> 'Context':
        self._descend(**self.updates)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._ascend()
        if exc_type:
            raise exc_type(exc_value).with_traceback(exc_traceback)
        return None
    
    ############################################################################ 
    def spawn(self) -> 'Context':
        return Context(**{k: getattr(self, k) for k in STATE_KEYS})
    
    def __repr__(self) -> str:
        interesting_keys = {k for k in STATE_KEYS 
                            if k != 'storage' and
                            getattr(self, k) != get_default(k)}
        if self._depth != 0:
            interesting_keys.add('_depth')
        data_str = ', '.join([f'{k}={getattr(self, k)}'
                              for k in interesting_keys])
        return f'Context({data_str})'
    
    ############################################################################ 
    ### queries
    ############################################################################ 
    def attach_query(self, query_obj:TUnion[ValQuery, OpQuery]):
        assert self.mode in (MODES.query, MODES.query_delete)
        self._queries.append(query_obj)
        self._cur_tree.add_qobj(qobj=query_obj)
    
    def reset_queries(self):
        self._queries = []
    
    def reset_tree(self):
        self._qtree = QTree(parent=None)
        self._cur_tree = self._qtree
    
    def capture_calls(self, func:FuncUIBase, call:Call) -> TList[Call]:
        # copy the inputs so that `call` is not modified
        inputs = {k: v.detached() for k, v in call.inputs.items()}
        ui_inputs = rename_dict_keys(
            dct=inputs,
            mapping=invert_dict(func.op.sig_map.fixed_inputs_map())
        )
        call_buffer = CallBuffer()
        with capture(storage=self.storage, captured_calls=call_buffer):
            func(**ui_inputs)
        res = call_buffer.unique_calls()
        call_buffer.reset()
        return res

    def _apply_col_names(self, df:pd.DataFrame,
                         val_queries:TTuple[ValQuery,...],
                         names:TList[str]=None):
        if names is None:
            display_names = [vq.display_name for vq in val_queries]
            for i, name in enumerate(display_names):
                if name is None:
                    display_names[i] = f'unnamed_{i}'
            assert all(isinstance(elt, str) for elt in display_names)
            names = display_names
        # apply column names in-place 
        if len(set(names)) != len(names):
            raise NotImplementedError()
        assert len(names) == len(df.columns)
        df.columns = names
    
    def qcalls(self, *val_queries:ValQuery):
        op_weaves:TTuple[OpQuery,...] = tuple()
        all_val_queries, all_op_queries = traverse_all(val_weaves=val_queries,
                                                       op_weaves=op_weaves)
        assert self.storage is not None
        self._compiler = self.storage.make_compiler(val_queries=all_val_queries,
                                                    op_queries=all_op_queries)
        query = self._compiler.query_calls()
        return self.storage.rel_adapter.rel_storage.fast_select(query=query)

    def commit_query_deletions(self, verbose:bool=None):
        assert self.storage is not None
        # get the high-level calls being deleted
        val_queries = tuple([obj for obj in self._queries
                             if isinstance(obj, ValQuery)])
        op_queries = tuple([obj for obj in self._queries 
                            if isinstance(obj, OpQuery)])
        all_val_queries, all_op_queries = traverse_all(val_weaves=val_queries,
                                                       op_weaves=op_queries)
        if (not all_val_queries) and (not all_op_queries):
            return
        self._compiler = self.storage.make_compiler(
            val_queries=all_val_queries,
            op_queries=all_op_queries
        )
        select_queries = tuple([op_query for op_query in op_queries 
                                if op_query.tags.get('delete', False)])
        call_uids = self._compiler.get_calls(select_queries=select_queries)
        partitions = self.storage.call_st.lookup_partitions(uids=call_uids)
        locs = [PartitionedCallLocation(uid=uid, partition=partition) 
                for uid, partition in zip(call_uids, partitions)]
        calls = self.storage.call_st.mget(locs=locs, allow_fallback=False)
        # group them by operation
        labels = [(call.op.name, call.op.version) for call in calls]
        calls_by_op = group_like(objs=calls, labels=labels)
        # figure out operations
        superop_func_dict = {}
        for name, version in calls_by_op.keys():
            for f in self.storage.synchronizer.funcs.values():
                if (f.op.version == version and
                    f.op.name == name and
                    f.op.is_super):
                    superop_func_dict[name, version] = f
        # capture calls
        all_calls = {}
        for (name, version), calls in calls_by_op.items():
            if (name, version) in superop_func_dict:
                f = superop_func_dict[name, version]
                for call in calls:
                    captured_calls = self.capture_calls(func=f, call=call)
                    for captured_call in captured_calls:
                        all_calls[captured_call.uid] = captured_call
            else:
                for call in calls:
                    all_calls[call.uid] = call
        # delete
        self.storage.delete_with_dependents(calls=calls, verbose=verbose)
    
    def qlocs(self, *val_queries:ValQuery, 
              names:TList[str]=None) -> pd.DataFrame:
        op_weaves:TTuple[OpQuery,...] = tuple()
        all_val_queries, all_op_queries = traverse_all(val_weaves=val_queries,
                                                       op_weaves=op_weaves)
        # branching step
        tree_qobjs_subset = induced_queries(root=self._qtree.get_root(),
                                            val_queries=list(val_queries))
        all_val_queries = [vq for vq in all_val_queries 
                           if vq in tree_qobjs_subset]
        all_op_queries = [opq for opq in all_op_queries 
                          if opq in tree_qobjs_subset]
        assert self.storage is not None
        self._compiler = self.storage.make_compiler(val_queries=all_val_queries,
                                                    op_queries=all_op_queries)
        locations_df = self._compiler.get_locations(select_queries=val_queries)
        self._apply_col_names(df=locations_df, names=names, val_queries=val_queries)
        return locations_df
    
    def locs_to_vrefs(self, locs_df:pd.DataFrame,
                      lazy:bool=False) -> pd.DataFrame:
        """
        Convert a locs table to a vref table
        """
        return self.storage.val_adapter.evaluate(df=locs_df,
                                                 unwrap_vrefs=False,
                                                 lazy=lazy)
    
    def locs_to_objs(self, locs_df:pd.DataFrame) -> pd.DataFrame:
        return self.storage.val_adapter.evaluate(df=locs_df)
    
    def qget(self, *val_queries:TAny, names:TList[str]=None,
             lazy:bool=False) -> pd.DataFrame:
        locations_df = self.qlocs(*val_queries)
        res = self.storage.val_adapter.evaluate(df=locations_df,
                                                unwrap_vrefs=False,
                                                lazy=lazy)
        self._apply_col_names(df=res, names=names, val_queries=val_queries)
        return res
    
    def qeval(self, *val_queries:TAny, names:TList[str]=None) -> pd.DataFrame:
        """
        Return table of unwrapped values for the given queries
        """
        locations_df = self.qlocs(*val_queries)
        res = self.storage.val_adapter.evaluate(df=locations_df)
        self._apply_col_names(df=res, names=names, val_queries=val_queries)
        return res
    
    def get_table(self, *val_queries:TAny,
                  names:TList[str]=None) -> pd.DataFrame:
        return self.qeval(*val_queries, names=names)
    
    def get_refs(self, *val_queries:TAny,
                 names:TList[str]=None) -> pd.DataFrame:
        return self.qget(*val_queries, names=names)
    
    ############################################################################ 
    ### commits
    ############################################################################ 
    def commit_buffer(self):
        assert self.buffer is not None
        self.storage.commit_buffer(buffer=self.buffer)

    def commit(self, buffer_first:bool=True):
        if self.storage is None:
            logging.warning(f'Calling commit from a context without a storage does nothing')
            return
        if buffer_first:
            if self.buffer is not None:
                self.commit_buffer()
        assert self.partition is not None
        assert self.partition != CALLS.main_partition
        self.storage.commit(partition=self.partition)
    
    def commit_deletions(self):
        assert self.storage is not None
        assert self._deletion_buffer is not None
        self._deletion_buffer.commit_deletions(storage=self.storage)
    
    ############################################################################ 
    def attach(self, vref:ValueRef, shallow:bool=False):
        self.storage.val_adapter.attach(vref=vref, shallow=shallow)
        
    def mattach(self, vref_collection:TVRefCollection, shallow:bool=False):
        vrefs = get_collection_atoms(collection=vref_collection)
        self.storage.val_adapter.mattach(vrefs=vrefs, shallow=shallow)
    
    def get(self, loc:BaseObjLocation) -> ValueRef:
        return self.storage.val_adapter.mget([loc])[0]
    
    def save(self, vref_collection:TVRefCollection):
        vrefs = get_collection_atoms(collection=vref_collection)
        self.storage.val_adapter.mset(vrefs=vrefs)
    
    def where_is(self, vref:ValueRef) -> BaseObjLocation:
        return self.storage.where_is(vref=vref)

    ############################################################################ 
    ### mocking ray
    ############################################################################ 
    def init_ray(self):
        assert EnvConfig.has_ray
        if self.disable_ray:
            logging.info('Skip ray init because disable_ray=True')
        else:
            ray.init(ignore_reinit_error=True)
    
    def ray_get(self, object_refs:TList[TAny]) -> TList[TAny]:
        assert EnvConfig.has_ray
        if self.disable_ray:
            return object_refs
        else:
            return ray.get(object_refs)
    
    def ray_put(self, obj:TAny) -> TAny:
        assert EnvConfig.has_ray
        if self.disable_ray:
            return obj
        else:
            return ray.put(obj)
    

class NoopContext(Context):
    OVERRIDES = {
        'mode': MODES.noop
    }


class RunContext(Context):
    OVERRIDES = {
        'mode': MODES.run
    }
        

class QueryContext(Context):
    OVERRIDES = {
        'mode': MODES.query
    }
        

class DeleteContext(Context):
    OVERRIDES = {
        'mode': MODES.delete,
        'allow_calls': False,
        'lazy': True,
        'disable_ray': True,
        'force': False,
    }


class TransientContext(Context):
    OVERRIDES = {
        'mode': MODES.transient
    }
    

class DefineContext(Context):
    OVERRIDES = {
        'mode': MODES.define,
        'allow_calls': False,
    }
    

class QueryDeleteContext(Context):
    OVERRIDES = {
        'mode': MODES.query_delete
    }



class RetraceContext(Context):
    OVERRIDES = {
        'mode': MODES.run,
        'allow_calls': False,
        'autocommit': False,
        'autodelete': False,
        'lazy': True,
        'disable_ray': True,
        'force': False,
    }

    
class CaptureContext(Context):
    OVERRIDES = {
        'mode': MODES.capture,
        'allow_calls': False,
        'autocommit': False,
        'autodelete': False,
        'lazy': True,
        'disable_ray': True,
        'force': False,
    }



context = Context()
noop = NoopContext()
run = RunContext()
query = QueryContext()
transient = TransientContext()
delete = DeleteContext()
qdelete = QueryDeleteContext()
define = DefineContext()
retrace = RetraceContext()
capture = CaptureContext()
