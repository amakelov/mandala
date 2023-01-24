from duckdb import DuckDBPyConnection as Connection
from abc import ABC, abstractmethod
import datetime
from pypika import Query, Table
import pyarrow.parquet as pq
from textwrap import indent
from typing import Literal

from ..storages.rel_impls.utils import Transactable, transaction
from ..storages.kv import InMemoryStorage, MultiProcInMemoryStorage, KVStore
from ..storages.rel_impls.duckdb_impl import DuckDBRelStorage
from ..storages.rels import RelAdapter, RemoteEventLogEntry
from ..storages.sigs import SigSyncer
from ..storages.remote_storage import RemoteStorage
from ..common_imports import *
from ..core.config import Config
from ..core.model import Ref, Call, FuncOp, make_delayed
from ..core.builtins_ import Builtins
from ..core.wrapping import wrap_dict, wrap_list, unwrap
from ..core.tps import Type
from ..core.sig import Signature, _get_arg_annotations, _get_return_annotations
from ..core.workflow import Workflow, CallStruct
from ..core.utils import get_uid
from ..core.deps import DependencyGraph, DepKey, OpKey, CallableNode, Tracer
from .viz import _get_colorized_diff, write_output
from .utils import MODES, debug_call

from ..core.weaver import (
    ValQuery,
    FuncQuery,
    traverse_all,
    computational_graph_to_dot,
    qwrap,
)
from ..core.compiler import Compiler, QueryGraph

from .utils import wrap_inputs, bind_inputs, format_as_outputs
from ..utils import ask_user

if Config.has_dask:
    from dask import delayed


class OnChange:
    ignore = "ignore"
    new_version = "new_version"
    ask = "ask"


class GlobalContext:
    current: Optional["Context"] = None


class Context:
    def __init__(
        self,
        storage: "Storage" = None,
        mode: str = MODES.run,
        lazy: bool = False,
        allow_calls: bool = True,
        debug_calls: bool = False,
        debug_truncate: Optional[int] = 20,
    ):
        self.storage = storage
        self.mode = mode
        self.lazy = lazy
        self.allow_calls = allow_calls
        self.debug_calls = debug_calls
        self.debug_truncate = debug_truncate
        self.updates = {}
        self._updates_stack = []
        self._call_structs = []
        self._defined_funcs: List["FuncInterface"] = []
        self._call_buffer: List[Call] = []

    def _backup_state(self, keys: Iterable[str]) -> Dict[str, Any]:
        res = {}
        for k in keys:
            cur_v = self.__dict__[f"{k}"]
            if k == "storage":  # gotta use a pointer
                res[k] = cur_v
            else:
                res[k] = copy.deepcopy(cur_v)
        return res

    def __enter__(self) -> "Context":
        if GlobalContext.current is None:
            GlobalContext.current = self
        ### verify update keys
        updates = self.updates
        if not all(
            k in ("storage", "mode", "lazy", "allow_calls", "debug_calls")
            for k in updates.keys()
        ):
            raise ValueError(updates.keys())
        if "mode" in updates.keys() and updates["mode"] not in MODES.all_:
            raise ValueError(updates["mode"])
        ### backup state
        before_update = self._backup_state(keys=updates.keys())
        self._updates_stack.append(before_update)
        ### apply updates
        for k, v in updates.items():
            if v is not None:
                self.__dict__[f"{k}"] = v
        # Load state from remote
        if self.storage is not None:
            # self.storage.sync_with_remote()
            self.storage.sync_from_remote()
        if (
            self.mode == MODES.run
            and self.storage is not None
            and self.storage.deps_root is not None
        ):
            self.storage.refresh_deps()
        return self

    def _undo_updates(self):
        """
        Roll back the updates from the current level
        """
        if not self._updates_stack:
            raise InternalError("No context to exit from")
        ascent_updates = self._updates_stack.pop()
        for k, v in ascent_updates.items():
            self.__dict__[f"{k}"] = v
        # unlink from global if done
        if len(self._updates_stack) == 0:
            GlobalContext.current = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        exc = None
        try:
            if self.mode == MODES.run:
                if self.storage is not None:
                    # commit calls from temp partition to main and tabulate them
                    if Config.autocommit:
                        self.storage.commit()
                    self.storage.sync_to_remote()
            elif self.mode == MODES.delete:
                print(f"Deleting {len(self._call_buffer)} calls...")
                self.storage.rel_adapter.delete_calls(calls=self._call_buffer)
                self._call_buffer = []
                self.storage.rel_adapter.cleanup()
            elif self.mode == MODES.query:
                pass
            elif self.mode == MODES.batch:
                executor = SimpleWorkflowExecutor()
                workflow = Workflow.from_call_structs(self._call_structs)
                calls = executor.execute(workflow=workflow, storage=self.storage)
                self.storage.commit(calls=calls)
            elif self.mode == MODES.define:
                storage = self.storage
                if storage.deps_root is not None:
                    new_func_ops = storage.refresh_deps()
                    for f in self._defined_funcs:
                        func = f.func_op.func
                        sig = f.func_op.sig
                        if sig.ui_name in new_func_ops:
                            f.func_op = new_func_ops[sig.ui_name]
                            f.func_op._set_func(func=func)
                            storage.synchronize_op(func_op=f.func_op)
                        else:
                            storage.synchronize(f=f)
                else:
                    for f in self._defined_funcs:
                        storage.synchronize(f=f)
                self._defined_funcs = []
            else:
                raise InternalError(self.mode)
        except Exception as e:
            exc = e
        self._undo_updates()
        if exc is not None:
            raise exc
        if exc_type:
            raise exc_type(exc_value).with_traceback(exc_traceback)
        return None

    def __call__(
        self,
        storage: Optional["Storage"] = None,
        allow_calls: bool = True,
        debug_calls: bool = False,
        **updates,
    ):
        self.updates = {
            "storage": storage,
            "allow_calls": allow_calls,
            "debug_calls": debug_calls,
            **updates,
        }
        return self

    def get_table(
        self,
        *queries: Any,
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        _engine: str = "sql",
        _filter_duplicates: bool = True,
        _visualize_steps_at: Optional[Path] = None,
    ) -> pd.DataFrame:
        #! important
        # We must sync any dirty cache elements to the DuckDB store before performing a query.
        # If we don't, we'll query a store that might be missing calls and objs.
        wrapped_queries = [qwrap(q) for q in queries]
        self.storage.commit()
        return self.storage.execute_query(
            select_queries=wrapped_queries,
            engine=_engine,
            values=values,
            filter_duplicates=_filter_duplicates,
            visualize_steps_at=_visualize_steps_at,
        )


class Storage(Transactable):
    """
    Groups together all the components of the storage system.

    Responsible for things that require multiple components to work together,
    e.g.
        - committing: moving calls from the "temporary" partition to the "main"
        partition. See also `CallStorage`.
        - synchronizing: connecting an operation with the storage and performing
        any necessary updates
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        root: Optional[Union[Path, RemoteStorage]] = None,
        timestamp: Optional[datetime.datetime] = None,
        multiproc: bool = False,
        call_cache: Optional[KVStore] = None,
        obj_cache: Optional[KVStore] = None,
        signatures: Optional[Dict[Tuple[str, int], Signature]] = None,
        _read_only: bool = False,
        ### dependency tracking config
        deps_root: Optional[Path] = None,
        on_change: str = "ask",
    ):
        self.root = root
        if call_cache is None:
            call_cache = MultiProcInMemoryStorage() if multiproc else InMemoryStorage()
        self.call_cache = call_cache
        if obj_cache is None:
            obj_cache = MultiProcInMemoryStorage() if multiproc else InMemoryStorage()
        self.obj_cache = obj_cache
        # all objects (inputs and outputs to operations, defaults) are saved here
        # stores the memoization tables
        if db_path is None and Config._persistent_storage_testing:
            # get a temp db path
            # generate a random filename
            db_name = f"db_{get_uid()}.db"
            db_path = Path(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f"../../temp_dbs/{db_name}",
                )
            ).resolve()
        self.db_path = db_path
        self.rel_storage = DuckDBRelStorage(
            address=None if db_path is None else str(db_path),
            _read_only=_read_only,
        )

        # if deps_root is not None, dependencies will be tracked
        if deps_root is not None:
            self.deps_root = Path(deps_root).absolute().resolve()
        else:
            self.deps_root = None
        self.on_change = on_change

        # manipulates the memoization tables
        self.rel_adapter = RelAdapter(
            rel_storage=self.rel_storage, deps_root=self.deps_root
        )
        self.sig_adapter = self.rel_adapter.sig_adapter
        self.sig_syncer = SigSyncer(sig_adapter=self.sig_adapter, root=self.root)
        if signatures is not None:
            self.sig_adapter.dump_state(state=signatures)
        self.last_timestamp = (
            timestamp if timestamp is not None else datetime.datetime.fromtimestamp(0)
        )

        # set up builtins
        for func_op in Builtins.OPS.values():
            self.synchronize_op(func_op=func_op)

    @property
    def in_memory(self) -> bool:
        return self.db_path is None

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
    ### read and write calls/objects
    ############################################################################
    @transaction()
    def call_exists(self, call_uid: str, conn: Optional[Connection] = None) -> bool:
        return self.call_cache.exists(call_uid) or self.rel_adapter.call_exists(
            call_uid, conn=conn
        )

    @transaction()
    def call_get(
        self, call_uid: str, lazy: bool, conn: Optional[Connection] = None
    ) -> Call:
        if self.call_cache.exists(call_uid):
            return self.call_cache.get(call_uid)
        else:
            lazy_call = self.rel_adapter.call_get_lazy(call_uid, conn=conn)
            if not lazy:
                # load the values of the inputs and outputs
                inputs = {
                    k: self.obj_get(v.uid, conn=conn)
                    for k, v in lazy_call.inputs.items()
                }
                outputs = [self.obj_get(v.uid, conn=conn) for v in lazy_call.outputs]
                call_without_outputs = lazy_call.set_input_values(inputs=inputs)
                call = call_without_outputs.set_output_values(outputs=outputs)
                return call
            else:
                return lazy_call

    def cache_call_and_objs(self, call: Call) -> None:
        for vref in itertools.chain(call.inputs.values(), call.outputs):
            self.cache_obj(vref.uid, vref)
        self.cache_call(call_uid=call.uid, call=call)

    def cache_call(self, call_uid: str, call: Call) -> None:
        self.call_cache.set(call_uid, call)

    @transaction()
    def obj_get(self, obj_uid: str, conn: Optional[Connection] = None) -> Ref:
        if self.obj_cache.exists(obj_uid):
            return self.obj_cache.get(obj_uid)
        return self.rel_adapter.obj_get(uid=obj_uid, conn=conn)

    def cache_obj(self, obj_uid: str, vref: Ref) -> None:
        self.obj_cache.set(obj_uid, vref)

    @transaction()
    def preload_objs(self, uids: list[str], conn: Optional[Connection] = None):
        """
        Put the objects with the given UIDs in the cache.
        """
        uids_not_in_cache = [uid for uid in uids if not self.obj_cache.exists(uid)]
        for uid, vref in zip(
            uids_not_in_cache,
            self.rel_adapter.obj_gets(uids=uids_not_in_cache, conn=conn),
        ):
            self.obj_cache.set(k=uid, v=vref)

    def evict_caches(self):
        for k in self.call_cache.keys():
            self.call_cache.delete(k=k)
        for k in self.obj_cache.keys():
            self.obj_cache.delete(k=k)

    @transaction()
    def commit(
        self, calls: Optional[list[Call]] = None, conn: Optional[Connection] = None
    ):
        """
        Flush calls and objs from the cache that haven't yet been written to DuckDB.
        """
        if calls is None:
            new_objs = {
                key: self.obj_cache.get(key) for key in self.obj_cache.dirty_entries
            }
            new_calls = [
                self.call_cache.get(key) for key in self.call_cache.dirty_entries
            ]
        else:
            new_objs = {}
            for call in calls:
                for vref in itertools.chain(call.inputs.values(), call.outputs):
                    new_objs[vref.uid] = vref
            new_calls = calls
        self.rel_adapter.obj_sets(new_objs, conn=conn)
        self.rel_adapter.upsert_calls(new_calls, conn=conn)
        if Config.evict_on_commit:
            self.evict_caches()
        # Remove dirty bits from cache.
        self.obj_cache.dirty_entries.clear()
        self.call_cache.dirty_entries.clear()

    @transaction()
    def cleanup(self, conn: Optional[Connection] = None):
        self.rel_adapter.cleanup(conn=conn)

    ############################################################################
    ### func synchronization
    ############################################################################
    @transaction()
    def synchronize_op(
        self,
        func_op: FuncOp,
        conn: Optional[Connection] = None,
    ):
        # first, pull the current data from the remote!
        self.sig_syncer.sync_from_remote(conn=conn)
        # this step also sends the signature to the remote
        new_sig = self.sig_syncer.sync_from_local(sig=func_op.sig, conn=conn)
        func_op.sig = new_sig
        # to send any default values that were created by adding inputs
        self.sync_to_remote()

    @transaction()
    def synchronize(
        self, f: Union["FuncInterface", Any], conn: Optional[Connection] = None
    ):
        if f.func_op.sig.version is not None and self.deps_root is not None:
            raise ValueError(
                "When automatically tracking dependencies, you cannot manually set the version of a function."
            )
        self.synchronize_op(func_op=f.func_op, conn=conn)
        f.is_synchronized = True

    ############################################################################
    ### func refactoring
    ############################################################################
    def _check_rename_precondition(self, func: "FuncInterface"):
        """
        In order to rename function data, the function must be synced with the
        storage, and the storage must be clean
        """
        if not func.is_synchronized:
            raise RuntimeError("Cannot rename while function is not synchronized.")
        if not self.is_clean:
            raise RuntimeError("Cannot rename while there is uncommited work.")

    @transaction()
    def rename_func(
        self, func: "FuncInterface", new_name: str, conn: Optional[Connection] = None
    ) -> Signature:
        """
        Rename a memoized function.

        What happens here:
            - check renaming preconditions
            - check there is no name clash with the new name
            - rename the memoization table
            - update signature object
            - invalidate the function (making it impossible to compute with it)
        """
        self._check_rename_precondition(func=func)
        sig = self.sig_syncer.sync_rename_sig(
            sig=func.func_op.sig, new_name=new_name, conn=conn
        )
        func.invalidate()
        return sig

    @transaction()
    def rename_arg(
        self,
        func: "FuncInterface",
        name: str,
        new_name: str,
        conn: Optional[Connection] = None,
    ) -> Signature:
        """
        Rename memoized function argument.

        What happens here:
            - check renaming preconditions
            - update signature object
            - rename table
            - invalidate the function (making it impossible to compute with it)
        """
        self._check_rename_precondition(func=func)
        sig = self.sig_syncer.sync_rename_input(
            sig=func.func_op.sig, input_name=name, new_input_name=new_name, conn=conn
        )
        func.invalidate()
        return sig

    ############################################################################
    ### remote sync operations
    ############################################################################
    @transaction()
    def bundle_to_remote(
        self, conn: Optional[Connection] = None
    ) -> RemoteEventLogEntry:
        """
        Collect the new calls according to the event log, and pack them into a
        dict of binary blobs to be sent off to the remote server.

        NOTE: this also renames tables and columns to their immutable internal
        names.
        """
        # Bundle event log and referenced calls into tables.
        event_log_df = self.rel_adapter.get_event_log(conn=conn)
        tables_with_changes = {}
        table_names_with_changes = event_log_df["table"].unique()

        event_log_table = Table(self.rel_adapter.EVENT_LOG_TABLE)
        for table_name in table_names_with_changes:
            table = Table(table_name)
            tables_with_changes[table_name] = self.rel_storage.execute_arrow(
                query=Query.from_(table)
                .join(event_log_table)
                .on(table[Config.uid_col] == event_log_table[Config.uid_col])
                .select(table.star),
                conn=conn,
            )
        # pass to internal names
        tables_with_changes = self.sig_adapter.rename_tables(
            tables_with_changes, to="internal", conn=conn
        )
        output = {}
        for table_name, table in tables_with_changes.items():
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            output[table_name] = buffer.getvalue()
        return output

    @transaction()
    def apply_from_remote(
        self, changes: list[RemoteEventLogEntry], conn: Optional[Connection] = None
    ):
        """
        Apply new calls from the remote server.

        NOTE: this also renames tables and columns to their UI names.
        """
        for raw_changeset in changes:
            changeset_data = {}
            for table_name, serialized_table in raw_changeset.items():
                buffer = io.BytesIO(serialized_table)
                deserialized_table = pq.read_table(buffer)
                changeset_data[table_name] = deserialized_table
            # pass to UI names
            changeset_data = self.sig_adapter.rename_tables(
                tables=changeset_data, to="ui", conn=conn
            )
            for table_name, deserialized_table in changeset_data.items():
                self.rel_storage.upsert(table_name, deserialized_table, conn=conn)
        # evaluated_tables = {k: self.rel_adapter.evaluate_call_table(ta=v, conn=conn) for k, v in data.items() if k != Config.vref_table}
        # logger.debug(f'Applied tables from remote: {evaluated_tables}')

    @transaction()
    def sync_from_remote(self, conn: Optional[Connection] = None):
        """
        Pull new calls from the remote server.

        Note that the server's schema (i.e. signatures) can be a super-schema of
        the local schema, but all local schema elements must be present in the
        remote schema, because this is enforced by how schema updates are
        performed.
        """
        if not isinstance(self.root, RemoteStorage):
            return
        # apply signature changes from the server first, because the new calls
        # from the server may depend on the new schema.
        self.sig_syncer.sync_from_remote(conn=conn)
        # next, pull new calls
        new_log_entries, timestamp = self.root.get_log_entries_since(
            self.last_timestamp
        )
        self.apply_from_remote(new_log_entries, conn=conn)
        self.last_timestamp = timestamp
        logger.debug("synced from remote")

    @transaction()
    def sync_to_remote(self, conn: Optional[Connection] = None):
        """
        Send calls to the remote server.

        As with `sync_from_remote`, the server may have a super-schema of the
        local schema. The current signatures are first pulled and applied to the
        local schema.
        """
        if not isinstance(self.root, RemoteStorage):
            # todo: there should be a way to completely ignore the event log
            # when there's no remote
            self.rel_adapter.clear_event_log(conn=conn)
        else:
            # collect new work and send it to the server
            changes = self.bundle_to_remote(conn=conn)
            self.root.save_event_log_entry(changes)
            # clear the event log only *after* the changes have been received
            self.rel_adapter.clear_event_log(conn=conn)
            logger.debug("synced to remote")

    @transaction()
    def sync_with_remote(self, conn: Optional[Connection] = None):
        if not isinstance(self.root, RemoteStorage):
            return
        self.sync_to_remote(conn=conn)
        self.sync_from_remote(conn=conn)

    @property
    def is_clean(self) -> bool:
        """
        Check that the storage has no uncommitted calls or objects.
        """
        return (
            self.call_cache.is_clean and self.obj_cache.is_clean
        )  # and self.rel_adapter.event_log_is_clean()

    ############################################################################
    ### spawning contexts
    ############################################################################
    def _nest(self, **updates) -> Context:
        if GlobalContext.current is not None:
            return GlobalContext.current(**updates)
        else:
            result = Context(**updates)
            GlobalContext.current = result
            return result

    def run(
        self,
        allow_calls: bool = True,
        debug_calls: bool = False,
        lazy: Optional[bool] = None,
        **updates,
    ) -> Context:
        # spawn context to execute or retrace calls
        lazy = not self.in_memory if lazy is None else lazy
        return self._nest(
            storage=self,
            allow_calls=allow_calls,
            debug_calls=debug_calls,
            mode=MODES.run,
            lazy=lazy,
            **updates,
        )

    def delete(self, **updates) -> Context:
        return self._nest(storage=self, mode=MODES.delete, **updates)

    def query(self, **updates) -> Context:
        # spawn a context to define a query
        return self._nest(
            storage=self,
            mode=MODES.query,
            **updates,
        )

    def batch(self, **updates) -> Context:
        # spawn a context to execute calls in batch
        return self._nest(
            storage=self,
            mode=MODES.batch,
            **updates,
        )

    def define(self, **updates) -> Context:
        # spawn a context to define ops. Needed for dependency tracking.
        return self._nest(
            storage=self,
            mode=MODES.define,
            **updates,
        )

    ############################################################################
    ### managing dependencies
    ############################################################################
    @transaction()
    def get_deps(
        self, func_interface: "FuncInterface", conn: Optional[Connection] = None
    ) -> DependencyGraph:
        sig = func_interface.sig
        all_deps = self.sig_adapter.deps_adapter.load_state(conn=conn)
        return all_deps.get_expanded(op_key=(sig.internal_name, sig.version))

    @transaction()
    def get_table(
        self,
        func_interface: Union["FuncInterface", Any],
        include_call_uid: bool = False,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        df = self.rel_storage.get_data(
            table=func_interface.func_op.sig.versioned_ui_name, conn=conn
        )
        df = self.rel_adapter.evaluate_call_table(ta=df, conn=conn)
        if not include_call_uid:
            df = df.drop(columns=[Config.uid_col])
        return df

    @transaction()
    def update_op_deps(
        self,
        func_op: FuncOp,
        new_deps: DependencyGraph,
        conn: Optional[Connection] = None,
    ):
        sig = func_op.sig
        current_dep_state = self.sig_adapter.deps_adapter.load_state(conn=conn)
        current_dep_state.update_op(
            op_key=(sig.internal_name, sig.version), graph=new_deps
        )
        self.sig_adapter.deps_adapter.dump_state(current_dep_state, conn=conn)

    @transaction()
    def refresh_deps(
        self, conn: Optional[Connection] = None
    ) -> Dict[Tuple[str, int], FuncOp]:
        on_change = self.on_change
        signatures = self.sig_adapter.load_state(conn=conn)
        old = self.sig_adapter.deps_adapter.load_state(conn=conn)
        global_graph = old.global_graph
        keys = global_graph.nodes.keys()
        new_deps = global_graph.load_current_state(keys=list(keys))
        missing_deps_by_module: Dict[str, List[DepKey]] = {}
        changed_deps_by_module: Dict[str, List[DepKey]] = {}
        for key, dep in global_graph.nodes.items():
            if key not in new_deps:
                # missing dependency
                missing_deps_by_module.setdefault(dep.module_name, []).append(key)
            elif (
                new_deps[key].comparable_representation()
                != dep.comparable_representation()
            ):
                # changed dependency
                changed_deps_by_module.setdefault(dep.module_name, []).append(key)
        deps_to_ops = old.get_deps_to_ops()
        actions: Dict[OpKey, List[str]] = {}
        question = "Choose an action to apply to all functions: [i]gnore, [n]ew version, [a]bort?"
        valid_options = ["i", "n", "a"]

        def get_action() -> str:
            if on_change == OnChange.ask:
                action = ask_user(question=question, valid_options=valid_options)
            elif on_change == OnChange.ignore:
                action = "i"
            elif on_change == OnChange.new_version:
                action = "n"
            else:
                raise ValueError(f"Invalid on_change={on_change}")
            return action

        action_decoder = {"i": "ignore", "n": "new version", "a": "abort"}

        def process_msg(msg: str):
            if on_change == OnChange.ask:
                print(msg)

        for module_name in set.union(
            set(missing_deps_by_module.keys()), set(changed_deps_by_module.keys())
        ):
            process_msg(f"FOUND CHANGES IN MODULE {module_name}:")
            all_keys = [
                (missing_key, "missing")
                for missing_key in missing_deps_by_module.get(module_name, [])
            ]
            all_keys.extend(
                [
                    (changed_key, "changed")
                    for changed_key in changed_deps_by_module.get(module_name, [])
                ]
            )
            for key, change_type in all_keys:
                affected_ops = deps_to_ops[key]
                affected_ops_ui_names = [
                    signatures[op_key].ui_name for op_key in affected_ops
                ]
                affected_ops_label = "[{}]".format(", ".join(affected_ops_ui_names))
                # fix above
                dependency_label = key[1]
                process_msg(
                    f"  {change_type.upper()} dependency {dependency_label} for memoized functions {affected_ops_label}"
                )
                if change_type == "changed":
                    current_rep = global_graph.nodes[key].diff_representation()
                    new_rep = new_deps[key].diff_representation()
                    style = (
                        "multiline"
                        if isinstance(global_graph.nodes[key], CallableNode)
                        else "inline"
                    )
                    process_msg(
                        indent(
                            _get_colorized_diff(
                                current=current_rep, new=new_rep, style=style
                            ),
                            prefix=4 * " ",
                        )
                    )
                action = get_action()
                if action == "a":
                    raise Exception("Aborting")
                for op_key in affected_ops:
                    actions.setdefault(op_key, []).append(action)
                process_msg(
                    f"Choice: {action_decoder[action]} for memoized functions {affected_ops_label}"
                )
        new_func_ops = {}  # ui_name -> new func op
        for op_key, op_actions in actions.items():
            if any(action == "n" for action in op_actions):
                sig = self.sig_adapter.load_state(conn=conn)[op_key]
                new_sig = sig.bump_version()
                new_func_op = FuncOp(sig=new_sig, func=None)
                self.synchronize_op(func_op=new_func_op, conn=conn)
                new_func_ops[sig.ui_name] = new_func_op
                print(f"  Created new version of {sig.ui_name}")
        global_graph.update_representations(nodes=new_deps)
        self.sig_adapter.deps_adapter.dump_state(old, conn=conn)
        return new_func_ops

    ############################################################################
    ### make calls in contexts
    ############################################################################
    @transaction()
    def _load_memoization_tables(
        self, evaluate: bool = False, conn: Optional[Connection] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get a dict of {versioned internal name: memoization table} for all
        functions. Note that memoization tables are labeled by UI arg names.
        """
        sigs = self.sig_adapter.load_state(conn=conn)
        ui_to_internal = {
            sig.versioned_ui_name: sig.versioned_internal_name for sig in sigs.values()
        }
        ui_call_data = self.rel_adapter.get_all_call_data(conn=conn)
        call_data = {ui_to_internal[k]: v for k, v in ui_call_data.items()}
        if evaluate:
            call_data = {
                k: self.rel_adapter.evaluate_call_table(v, conn=conn)
                for k, v in call_data.items()
            }
        return call_data

    @transaction()
    def execute_query(
        self,
        select_queries: List[ValQuery],
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        engine: str = "sql",
        filter_duplicates: bool = True,
        visualize_steps_at: Optional[Path] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """
        Execute the given queries and return the result as a pandas DataFrame.
        """
        if visualize_steps_at is not None:
            assert engine == "naive"
        if not select_queries:
            return pd.DataFrame()
        val_queries, func_queries = traverse_all(select_queries)
        if engine == "sql":
            compiler = Compiler(val_queries=val_queries, func_queries=func_queries)
            query = compiler.compile(
                select_queries=select_queries, filter_duplicates=filter_duplicates
            )
            df = self.rel_storage.execute_df(query=str(query), conn=conn)
        elif engine == "naive":
            val_copies, func_copies, select_copies = QueryGraph._copy_graph(
                val_queries, func_queries, select_queries=select_queries
            )
            memoization_tables = self._load_memoization_tables(conn=conn)
            tables = {
                f: memoization_tables[f.func_op.sig.versioned_internal_name]
                for f in func_copies
            }
            query_graph = QueryGraph(
                val_queries=val_copies,
                func_queries=func_copies,
                select_queries=select_copies,
                tables=tables,
                _table_evaluator=self.rel_adapter.evaluate_call_table,
                _visualize_steps_at=visualize_steps_at,
            )
            df = query_graph.solve()
            if filter_duplicates:
                df = df.drop_duplicates(keep="first")
        else:
            raise NotImplementedError()
        # now, evaluate the table
        if values in ("objs", "refs"):
            uids_to_collect = [
                item for _, column in df.items() for _, item in column.items()
            ]
            self.preload_objs(uids_to_collect, conn=conn)
            if values == "objs":
                result = df.applymap(lambda uid: unwrap(self.obj_get(uid, conn=conn)))
            else:
                result = df.applymap(lambda uid: self.obj_get(uid, conn=conn))
        elif values == "uids":
            result = df
        elif values == "lazy":
            result = df.applymap(lambda uid: Ref.from_uid(uid=uid))
        else:
            raise ValueError(
                f"Invalid value for `values`: {values}. Must be one of "
                "['objs', 'refs', 'uids', 'lazy']"
            )
        # finally, name the columns
        cols = [
            f"unnamed_{i}" if query.column_name is None else query.column_name
            for i, query in zip(range(len((result.columns))), select_queries)
        ]
        result.rename(columns=dict(zip(result.columns, cols)), inplace=True)
        return result

    @transaction()
    def visualize_query(
        self,
        *select_queries: List[ValQuery],
        how: str = "none",
        output_path: Optional[Path] = None,
        conn: Optional[Connection] = None,
    ) -> None:
        val_queries, func_queries = traverse_all(select_queries)
        memoization_tables = self._load_memoization_tables(evaluate=True, conn=conn)
        tables_by_fq = {
            fq: memoization_tables[fq.func_op.sig.versioned_internal_name]
            for fq in func_queries
        }
        dot_string = computational_graph_to_dot(
            val_queries=val_queries,
            func_queries=func_queries,
            layout="computational",
            memoization_tables=tables_by_fq,
        )
        output_ext = "svg" if how in ["browser"] else "png"
        write_output(
            dot_string=dot_string,
            output_ext=output_ext,
            output_path=output_path,
            show_how=how,
        )

    @transaction()
    def _process_call_found(
        self, call_uid: str, lazy: bool, conn: Connection
    ) -> Tuple[List[Ref], Call]:
        call = self.call_get(call_uid, lazy=lazy, conn=conn)
        if not lazy:
            self.preload_objs([v.uid for v in call.outputs], conn=conn)
            wrapped_outputs = [self.obj_get(v.uid, conn=conn) for v in call.outputs]
        else:
            wrapped_outputs = [v for v in call.outputs]
        return wrapped_outputs, call

    @transaction()
    def call_run(
        self,
        func_op: FuncOp,
        inputs: Dict[str, Union[Any, Ref]],
        allow_calls: bool = True,
        debug_calls: bool = False,
        lazy: bool = False,
        _collect_calls: bool = False,
        _recurse: bool = False,
        _call_buffer: Optional[List[Call]] = None,
        conn: Optional[Connection] = None,
    ) -> Tuple[List[Ref], Call]:
        wrapped_inputs, input_calls = wrap_dict(
            objs=inputs,
            annotations=func_op.input_annotations,
        )
        call_uid = func_op.get_call_uid(wrapped_inputs=wrapped_inputs)
        call_exists = self.call_exists(call_uid, conn=conn)
        memoized = call_exists
        # check if call UID exists in call storage
        if not call_exists or (_recurse and func_op.is_super):
            # compute op
            if not allow_calls and not call_exists:
                raise ValueError(
                    f"Call to {func_op.sig.ui_name} not found in call storage."
                )
            if not self.in_memory and not memoized and not func_op.is_super:
                self.rel_adapter.mattach(vrefs=list(wrapped_inputs.values()), conn=conn)
            if Config.autounwrap_inputs and (not func_op.is_super):
                raw_inputs = unwrap(obj=wrapped_inputs, through_collections=True)
            else:
                raw_inputs = wrapped_inputs
            outputs, dependency_state_option = func_op.compute(
                raw_inputs,
                deps_root=self.deps_root,
            )
            wrapped_outputs, output_calls = wrap_list(
                objs=outputs, annotations=func_op.output_annotations
            )
            call = Call(
                uid=call_uid,
                inputs=wrapped_inputs,
                outputs=wrapped_outputs,
                func_op=func_op,
            )
            if not memoized:
                for constituent_call in itertools.chain(
                    [call], input_calls, output_calls
                ):
                    self.cache_call_and_objs(call=constituent_call)
                # update dependencies only after the call has been stored
                if dependency_state_option is not None:
                    self.update_op_deps(
                        func_op=func_op, new_deps=dependency_state_option, conn=conn
                    )
        else:
            if sys.gettrace() is not None and self.deps_root is not None:
                data = Tracer.generate_terminal_data(
                    func=func_op.func,
                    internal_name=func_op.sig.internal_name,
                    version=func_op.sig.version,
                )
                Tracer.break_signal(data=data)
            wrapped_outputs, call = self._process_call_found(
                call_uid=call_uid, lazy=lazy, conn=conn
            )
        if debug_calls:
            debug_call(
                func_name=func_op.sig.ui_name,
                memoized=memoized,
                wrapped_inputs=wrapped_inputs,
                wrapped_outputs=wrapped_outputs,
            )
        if _collect_calls:
            _call_buffer.append(call)
        return wrapped_outputs, call

    # async def call_run_async(
    #     self, func_op: FuncOp, inputs: Dict[str, Union[Any, Ref]]
    # ) -> Tuple[List[Ref], Call]:
    #     wrapped_inputs = wrap_inputs(inputs)
    #     call_uid = func_op.get_call_uid(wrapped_inputs=wrapped_inputs)
    #     # check if call UID exists in call storage
    #     if self.call_exists(call_uid):
    #         return self._process_call_found(call_uid=call_uid)
    #     else:
    #         # compute op
    #         if Config.autounwrap_inputs:
    #             raw_inputs = {k: v.obj for k, v in wrapped_inputs.items()}
    #         else:
    #             raw_inputs = wrapped_inputs
    #         outputs, dependency_state = await func_op.compute_async(raw_inputs)
    #         return self._process_call_not_found(
    #             func_outputs=outputs,
    #             call_uid=call_uid,
    #             wrapped_inputs=wrapped_inputs,
    #             func_op=func_op,
    #         )

    def call_query(
        self, func_op: FuncOp, inputs: Dict[str, ValQuery]
    ) -> List[ValQuery]:
        assert all(isinstance(inp, ValQuery) for inp in inputs.values())
        fq = FuncQuery.link(inputs=inputs, func_op=func_op)
        return fq.outputs

    def call_batch(
        self, func_op: FuncOp, inputs: Dict[str, Ref]
    ) -> Tuple[List[Ref], CallStruct]:
        output_types = [Type.from_annotation(a) for a in func_op.output_annotations]
        outputs = [make_delayed(tp=tp) for tp in output_types]
        call_struct = CallStruct(func_op=func_op, inputs=inputs, outputs=outputs)
        return outputs, call_struct


class WorkflowExecutor(ABC):
    @abstractmethod
    def execute(self, workflow: Workflow, storage: Storage) -> List[Call]:
        pass


class SimpleWorkflowExecutor(WorkflowExecutor):
    def execute(self, workflow: Workflow, storage: Storage) -> List[Call]:
        result = []
        for op_node in workflow.op_nodes:
            call_structs = workflow.op_node_to_call_structs[op_node]
            for call_struct in call_structs:
                func_op, inputs, outputs = (
                    call_struct.func_op,
                    call_struct.inputs,
                    call_struct.outputs,
                )
                assert all([not inp.is_delayed() for inp in inputs.values()])
                vref_outputs, call = storage.call_run(
                    func_op=func_op,
                    inputs=inputs,
                )
                # overwrite things
                for output, vref_output in zip(outputs, vref_outputs):
                    output._obj = vref_output.obj
                    output.uid = vref_output.uid
                    output.in_memory = True
                result.append(call)
        # filter out repeated calls
        result = list({call.uid: call for call in result}.values())
        return result


class FuncInterface:
    """
    Wrapper around a memoized function.

    This is the object the `@op` decorator converts functions into.
    """

    def __init__(
        self,
        func_op: FuncOp,
        executor: str = "python",
    ):
        self.func_op = func_op
        self.__name__ = self.func_op.sig.ui_name
        self.is_synchronized = False
        self.is_invalidated = False
        self.executor = executor
        if (
            GlobalContext.current is not None
            and GlobalContext.current.mode == MODES.define
        ):
            GlobalContext.current._defined_funcs.append(self)

    @property
    def sig(self) -> Signature:
        return self.func_op.sig

    def __repr__(self) -> str:
        sig = self.func_op.sig
        if self.is_invalidated:
            # clearly distinguish stale functions
            return f"FuncInterface(func_name={sig.ui_name}, is_invalidated=True)"
        else:
            return f"FuncInterface(func_name={sig.ui_name}, version={sig.version})"

    def invalidate(self):
        self.is_invalidated = True
        self.is_synchronized = False

    def _preprocess_call(
        self, *args, **kwargs
    ) -> Tuple[Dict[str, Any], str, Storage, Context]:
        context = GlobalContext.current
        storage = context.storage
        if self.is_invalidated:
            raise RuntimeError(
                "This function has been invalidated due to a change in the signature, and cannot be called"
            )
        if not self.func_op.sig.has_internal_data:
            # synchronize if necessary
            storage.synchronize(self)
            # synchronize(func=self, storage=context.storage)
        inputs = bind_inputs(args, kwargs, mode=context.mode, func_op=self.func_op)
        mode = context.mode
        return inputs, mode, storage, context

    def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        context = GlobalContext.current
        if context is None:
            # mandala is completely disabled when not in a context
            return self.func_op.func(*args, **kwargs)
        inputs, mode, storage, context = self._preprocess_call(*args, **kwargs)
        if mode == MODES.run:
            if self.executor == "python":
                outputs, call = storage.call_run(
                    func_op=self.func_op,
                    inputs=inputs,
                    allow_calls=context.allow_calls,
                    debug_calls=context.debug_calls,
                    lazy=context.lazy,
                )
                return format_as_outputs(outputs=outputs)
            # elif self.executor == 'asyncio' or inspect.iscoroutinefunction(self.func_op.func):
            elif self.executor == "dask":
                assert (
                    not storage.rel_storage.in_memory
                ), "Dask executor only works with a persistent storage"

                def daskop_f(*args, __data__, **kwargs):
                    call_cache, obj_cache, db_path = __data__
                    temp_storage = Storage(db_path=db_path, _read_only=True)
                    temp_storage.call_cache = call_cache
                    temp_storage.obj_cache = obj_cache
                    inputs = bind_inputs(
                        func_op=self.func_op, args=args, kwargs=kwargs, mode=MODES.run
                    )
                    outputs, _ = temp_storage.call_run(
                        func_op=self.func_op, inputs=inputs, mode=mode
                    )
                    return format_as_outputs(outputs=outputs)

                __data__ = (storage.call_cache, storage.obj_cache, storage.db_path)
                nout = self.func_op.sig.n_outputs
                return delayed(daskop_f, nout=nout)(*args, __data__=__data__, **kwargs)
            else:
                raise NotImplementedError()
        elif mode == MODES.delete:
            outputs, call = storage.call_run(
                func_op=self.func_op,
                inputs=inputs,
                allow_calls=False,
                debug_calls=False,
                _collect_calls=True,
                _recurse=True,
                _call_buffer=context._call_buffer,
            )
            return format_as_outputs(outputs=outputs)
        elif mode == MODES.query:
            return format_as_outputs(
                outputs=storage.call_query(func_op=self.func_op, inputs=inputs)
            )
        elif mode == MODES.batch:
            assert self.executor == "python"
            wrapped_inputs = wrap_inputs(inputs)
            outputs, call_struct = storage.call_batch(
                func_op=self.func_op, inputs=wrapped_inputs
            )
            context._call_structs.append(call_struct)
            return format_as_outputs(outputs=outputs)
        else:
            raise ValueError()


class AsyncioFuncInterface(FuncInterface):
    async def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        context = GlobalContext.current
        if context is None:
            # mandala is completely disabled when not in a context
            return self.func_op.func(*args, **kwargs)
        inputs, mode, storage, context = self._preprocess_call(*args, **kwargs)
        if mode == MODES.run:

            async def async_f(*args, __data__, **kwargs):
                call_cache, obj_cache, db_path = __data__
                temp_storage = Storage(db_path=db_path, _read_only=True)
                temp_storage.call_cache = call_cache
                temp_storage.obj_cache = obj_cache
                inputs = bind_inputs(
                    func_op=self.func_op, args=args, kwargs=kwargs, mode=MODES.run
                )
                outputs, _ = await temp_storage.call_run_async(
                    func_op=self.func_op, inputs=inputs
                )
                return format_as_outputs(outputs=outputs)

            __data__ = (storage.call_cache, storage.obj_cache, storage.db_path)
            return await async_f(*args, __data__=__data__, **kwargs)
        else:
            return super().__call__(*args, **kwargs)