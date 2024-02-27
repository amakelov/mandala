import datetime
import tqdm
from typing import Literal
from collections import deque

from .viz import _get_colorized_diff
from .utils import MODES
from .remote_utils import RemoteManager
from . import contexts
from .context_cache import Cache

from ..common_imports import *
from ..core.config import Config, Provenance, dump_output_name
from ..core.model import Ref, Call, FuncOp, ValueRef, Delayed
from ..core.builtins_ import Builtins, ListRef, DictRef, SetRef
from ..core.wrapping import (
    unwrap,
    compare_dfs_as_relations,
)
from ..core.tps import Type, AnyType, ListType, DictType, SetType
from ..core.sig import Signature
from ..core.utils import get_uid, OpKey

from ..storages.rel_impls.utils import Transactable, transaction, Connection
from ..storages.kv import InMemoryStorage, MultiProcInMemoryStorage, KVCache

if Config.has_duckdb:
    from ..storages.rel_impls.duckdb_impl import DuckDBRelStorage
from ..storages.rel_impls.sqlite_impl import SQLiteRelStorage
from ..storages.rels import RelAdapter, VersionAdapter
from ..storages.sigs import SigSyncer
from ..storages.remote_storage import RemoteStorage
from ..deps.tracers import DecTracer
from ..deps.versioner import Versioner, CodeState
from ..deps.utils import get_dep_key_from_func, extract_func_obj
from ..deps.model import DepKey, TerminalData

from ..queries.workflow import CallStruct
from ..queries.weaver import (
    ValNode,
    CallNode,
    traverse_all,
)
from ..queries.viz import (
    visualize_graph,
    print_graph,
    get_names,
    extract_names_from_scope,
    GraphPrinter,
    ValueLoaderABC,
)
from ..queries.main import Querier
from ..queries.graphs import get_canonical_order, InducedSubgraph

from ..core.prov import propagate_struct_provenance


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
        db_backend: str = Config.db_backend,
        spillover_dir: Optional[Union[str, Path]] = None,
        spillover_threshold_mb: Optional[float] = None,
        root: Optional[Union[Path, RemoteStorage]] = None,
        timestamp: Optional[datetime.datetime] = None,
        multiproc: bool = False,
        evict_on_commit: bool = None,
        signatures: Optional[Dict[Tuple[str, int], Signature]] = None,
        _read_only: bool = False,
        ### dependency tracking config
        deps_path: Optional[Union[Path, str]] = None,
        deps_package: Optional[str] = None,
        track_methods: bool = True,
        _strict_deps: bool = True,  # for testing only
        tracer_impl: Optional[type] = None,
    ):
        self.root = root
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
        self.db_backend = db_backend
        self.evict_on_commit = (
            Config.evict_on_commit if evict_on_commit is None else evict_on_commit
        )
        if Config.has_duckdb and db_backend == "duckdb":
            DBImplementation = DuckDBRelStorage
        else:
            DBImplementation = SQLiteRelStorage
        self.rel_storage = DBImplementation(
            address=None if db_path is None else str(db_path),
            _read_only=_read_only,
        )

        # manipulates the memoization tables
        self.rel_adapter = RelAdapter(
            rel_storage=self.rel_storage,
            spillover_dir=Path(spillover_dir) if spillover_dir is not None else None,
            spillover_threshold_mb=spillover_threshold_mb,
        )

        self.cache = Cache(rel_adapter=self.rel_adapter)

        # self.versions_adapter = VersionAdapter(rel_adapter=rel_adapter)
        self.sig_adapter = self.rel_adapter.sig_adapter
        self.sig_syncer = SigSyncer(sig_adapter=self.sig_adapter, root=self.root)
        if signatures is not None:
            self.sig_adapter.dump_state(state=signatures)
        self.last_timestamp = (
            timestamp if timestamp is not None else datetime.datetime.fromtimestamp(0)
        )

        self.version_adapter = VersionAdapter(rel_adapter=self.rel_adapter)
        if deps_path is not None:
            deps_path = (
                Path(deps_path).absolute().resolve()
                if deps_path != "__main__"
                else "__main__"
            )
            roots = [] if deps_path == "__main__" else [deps_path]
            self._versioned = True
            current_versioner = self.version_adapter.load_state()
            if current_versioner is not None:
                if current_versioner.paths != roots:
                    raise ValueError(
                        f"Found existing versioner with roots {current_versioner.paths}, but "
                        f"was asked to use {roots}"
                    )
            else:
                versioner = Versioner(
                    paths=roots,
                    TracerCls=DecTracer if tracer_impl is None else tracer_impl,
                    strict=_strict_deps,
                    track_methods=track_methods,
                    package_name=deps_package,
                )
                self.version_adapter.dump_state(state=versioner)
        else:
            self._versioned = False

        if root is not None:
            self.remote_manager = RemoteManager(
                rel_adapter=self.rel_adapter,
                sig_adapter=self.sig_adapter,
                rel_storage=self.rel_storage,
                sig_syncer=self.sig_syncer,
                root=self.root,
            )
        else:
            self.remote_manager = None

        # set up builtins
        for func_op in Builtins.OPS.values():
            self.synchronize_op(func_op=func_op)

    @property
    def in_memory(self) -> bool:
        return self.db_path is None

    @transaction()
    def get_versioner(self, conn: Optional[Connection] = None) -> Versioner:
        result = self.version_adapter.load_state(conn=conn)
        if result is None:
            raise ValueError("This storage is not versioned.")
        return result

    @property
    def versioned(self) -> bool:
        return self._versioned

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    def unwrap(self, obj: Any) -> Any:
        with self.run():
            res = unwrap(obj)
        return res

    @transaction()
    def commit(
        self,
        calls: Optional[List[Call]] = None,
        versioner: Optional[Versioner] = None,
        conn: Optional[Connection] = None,
    ):
        """
        Flush calls and objs from the cache that haven't yet been written to the database.
        """
        self.cache.commit(
            calls=calls,
            versioner=versioner,
            version_adapter=self.version_adapter,
            conn=conn,
        )

    @transaction()
    def eval_df(
        self,
        full_uids_df: pd.DataFrame,
        drop_duplicates: bool = False,
        values: Literal["objs", "refs", "uids", "full_uids", "lazy"] = "objs",
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """
        - ! this function loads objects in the cache; this is probably not
        transparent to the user
        - Note that currently we pass the full UIDs as input, and thus we return
        values with causal UIDs. Maybe it is desirable in some settings to
        disable this behavior.
        """
        if values == "full_uids":
            return full_uids_df
        has_meta = set(Config.special_call_cols).issubset(full_uids_df.columns)
        inp_outp_cols = [
            col for col in full_uids_df.columns if col not in Config.special_call_cols
        ]
        uids_df = full_uids_df[inp_outp_cols].applymap(
            lambda uid: uid.rsplit(".", 1)[0]
        )
        if has_meta:
            uids_df[Config.special_call_cols] = full_uids_df[Config.special_call_cols]
        if values in ("objs", "refs"):
            uids_to_collect = [
                item for _, column in uids_df.items() for _, item in column.items()
            ]
            self.cache.preload_objs(uids_to_collect, conn=conn)
            if values == "objs":
                result = uids_df.applymap(lambda uid: unwrap(self.cache.obj_get(uid)))
            else:
                result = full_uids_df.applymap(
                    lambda full_uid: self.cache.obj_get(
                        obj_uid=full_uid.split(".")[0],
                        causal_uid=full_uid.split(".")[1],
                    )
                )
        elif values == "uids":
            result = uids_df
        elif values == "lazy":
            result = full_uids_df[inp_outp_cols].applymap(
                lambda full_uid: Ref.from_uid(
                    uid=full_uid.split(".")[0], causal_uid=full_uid.split(".")[1]
                )
            )
            if has_meta:
                result[Config.special_call_cols] = full_uids_df[
                    Config.special_call_cols
                ]
        else:
            raise ValueError(
                f"Invalid value for `values`: {values}. Must be one of "
                "['objs', 'refs', 'uids', 'lazy']"
            )
        if drop_duplicates:
            result = result.drop_duplicates()
        return result

    @transaction()
    def get_table(
        self,
        func_interface: Union["funcs.FuncInterface", Any],
        meta: bool = False,
        values: Literal["objs", "uids", "full_uids", "refs", "lazy"] = "objs",
        drop_duplicates: bool = False,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        full_uids_df = self.rel_storage.get_data(
            table=func_interface.func_op.sig.versioned_ui_name, conn=conn
        )
        if not meta:
            full_uids_df = full_uids_df.drop(columns=Config.special_call_cols)
        df = self.eval_df(
            full_uids_df=full_uids_df,
            values=values,
            drop_duplicates=drop_duplicates,
            conn=conn,
        )
        return df

    ############################################################################
    ### synchronization
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
        self, f: Union["funcs.FuncInterface", Any], conn: Optional[Connection] = None
    ):
        if f._is_invalidated:
            raise RuntimeError(
                "This function has been invalidated due to a change in the signature, and cannot be called"
            )
        # if f._is_synchronized:
        #     if f._storage_id != id(self):
        #         raise RuntimeError(
        #             "This function is already synchronized with a different storage object. Re-define the function to synchronize it with this storage object."
        #         )
        #     return
        self.synchronize_op(func_op=f.func_op, conn=conn)
        f._is_synchronized = True
        f._storage_id = id(self)

    ############################################################################
    ### versioning
    ############################################################################
    @transaction()
    def guess_code_state(
        self, versioner: Optional[Versioner] = None, conn: Optional[Connection] = None
    ) -> CodeState:
        if versioner is None:
            versioner = self.get_versioner(conn=conn)
        return versioner.guess_code_state()

    @transaction()
    def sync_code(
        self, conn: Optional[Connection] = None
    ) -> Tuple[Versioner, CodeState]:
        versioner = self.get_versioner(conn=conn)
        code_state = self.guess_code_state(versioner=versioner, conn=conn)
        versioner.sync_codebase(code_state=code_state)
        return versioner, code_state

    @transaction()
    def sync_component(
        self,
        component: types.FunctionType,
        is_semantic_change: Optional[bool],
        conn: Optional[Connection] = None,
    ):
        # low-level versioning
        dep_key = get_dep_key_from_func(func=component)
        versioner = self.get_versioner(conn=conn)
        code_state = self.guess_code_state(versioner=versioner, conn=conn)
        result = versioner.sync_component(
            component=dep_key,
            is_semantic_change=is_semantic_change,
            code_state=code_state,
        )
        self.version_adapter.dump_state(state=versioner, conn=conn)
        return result

    @transaction()
    def _show_version_data(
        self,
        f: Union[Callable, "funcs.FuncInterface"],
        deps: bool = True,
        meta: bool = False,
        plain: bool = False,
        compact: bool = False,
        conn: Optional[Connection] = None,
    ):
        # show the versions of a function, with/without its dependencies
        func = extract_func_obj(obj=f, strict=True)
        component = get_dep_key_from_func(func=func)
        versioner = self.get_versioner(conn=conn)
        if deps:
            versioner.show_versions(
                component=component,
                include_metadata=meta,
                plain=plain,
            )
        else:
            versioner.component_dags[component].show(
                compact=compact, plain=plain, include_metadata=meta
            )

    @transaction()
    def versions(
        self,
        f: Union[Callable, "funcs.FuncInterface"],
        meta: bool = False,
        plain: bool = False,
        conn: Optional[Connection] = None,
    ):
        self._show_version_data(
            f=f,
            deps=True,
            meta=meta,
            plain=plain,
            compact=False,
            conn=conn,
        )

    @transaction()
    def sources(
        self,
        f: Union[Callable, "funcs.FuncInterface"],
        meta: bool = False,
        plain: bool = False,
        compact: bool = False,
        conn: Optional[Connection] = None,
    ):
        func = extract_func_obj(obj=f, strict=True)
        component = get_dep_key_from_func(func=func)
        versioner = self.get_versioner(conn=conn)
        print(
            f"Revision history for the source code of function {component[1]} from module {component[0]} "
            '("===HEAD===" is the current version):'
        )
        versioner.component_dags[component].show(
            compact=compact, plain=plain, include_metadata=meta
        )

    @transaction()
    def code(
        self, version_id: str, meta: bool = False, conn: Optional[Connection] = None
    ):
        # show a copy-pastable version of the code for a given version id. Plain
        # by design.
        result = self.get_code(version_id=version_id, show=False, meta=meta, conn=conn)
        print(result)

    @transaction()
    def get_code(
        self,
        version_id: str,
        show: bool = True,
        meta: bool = False,
        conn: Optional[Connection] = None,
    ) -> str:
        versioner = self.get_versioner(conn=conn)
        for dag in versioner.component_dags.values():
            if version_id in dag.commits.keys():
                text = dag.get_content(commit=version_id)
                if show:
                    print(text)
                return text
        for (
            content_version,
            version,
        ) in versioner.get_flat_versions().items():
            if version_id == content_version:
                raw_string = versioner.present_dependencies(
                    commits=version.semantic_expansion,
                    include_metadata=meta,
                )
                if show:
                    print(raw_string)
                return raw_string
        raise ValueError(f"version id {version_id} not found")

    @transaction()
    def diff(
        self,
        id_1: str,
        id_2: str,
        context_lines: int = 2,
        conn: Optional[Connection] = None,
    ):
        code_1: str = self.get_code(version_id=id_1, show=False, conn=conn)
        code_2: str = self.get_code(version_id=id_2, show=False, conn=conn)
        print(
            _get_colorized_diff(current=code_1, new=code_2, context_lines=context_lines)
        )

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
                k: self.eval_df(full_uids_df=v, values="objs", conn=conn)
                for k, v in call_data.items()
            }
        return call_data

    @transaction()
    def get_compatible_semantic_versions(
        self,
        fqs: Set[CallNode],
        conn: Optional[Connection] = None,
    ) -> Tuple[Optional[Dict[OpKey, Set[str]]], Optional[Dict[DepKey, Set[str]]]]:
        if not self.versioned:
            return None, None
        if contexts.GlobalContext.current is not None:
            code_state = contexts.GlobalContext.current._code_state
        else:
            code_state = self.guess_code_state(
                versioner=self.get_versioner(), conn=conn
            )
        result_ops = {}
        result_deps = {}
        versioner = self.get_versioner(conn=conn)
        for func_query in fqs:
            sig = func_query.func_op.sig
            op_key = (sig.internal_name, sig.version)
            # dep_key = get_dep_key_from_func(func=func_query.func_op.func)
            dep_key = (func_query.func_op._module, func_query.func_op._qualname)
            if func_query.func_op._is_builtin:
                result_ops[op_key] = None
                result_deps[dep_key] = None
            else:
                versions = versioner.get_semantically_compatible_versions(
                    component=dep_key, code_state=code_state
                )
                result_ops[op_key] = set([v.semantic_version for v in versions])
                result_deps[dep_key] = result_ops[op_key]
        return result_ops, result_deps

    @transaction()
    def execute_query(
        self,
        selection: List[ValNode],
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        names: Dict[ValNode, str],
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        engine: Optional[Literal["sql", "naive", "_test"]] = None,
        local: bool = False,
        verbose: bool = True,
        drop_duplicates: bool = True,
        visualize_steps_at: Optional[Path] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """
        Execute the given queries and return the result as a pandas DataFrame.
        """
        if engine is None:
            engine = Config.query_engine

        def rename_cols(
            df: pd.DataFrame, selection: List[ValNode], names: Dict[ValNode, str]
        ):
            df.columns = [str(i) for i in range(len(df.columns))]
            cols = [names[query] for query in selection]
            df.rename(columns=dict(zip(df.columns, cols)), inplace=True)

        Querier.validate_query(vqs=vqs, fqs=fqs, selection=selection, names=names)
        context = contexts.GlobalContext.current
        if verbose:
            print(
                "Pattern-matching to the following computational graph (all constraints apply):"
            )
            print_graph(vqs=vqs, fqs=fqs, names=names, selection=selection)
        if visualize_steps_at is not None:
            assert engine == "naive"
        if engine in ["sql", "_test"]:
            version_constraints, _ = self.get_compatible_semantic_versions(
                fqs=fqs, conn=conn
            )
            call_uids = context._call_uids if local else None
            query = Querier.compile(
                selection=selection,
                vqs=vqs,
                fqs=fqs,
                version_constraints=version_constraints,
                filter_duplicates=drop_duplicates,
                call_uids=call_uids,
            )
            start = time.time()
            sql_uids_df = self.rel_storage.execute_df(query=str(query), conn=conn)
            end = time.time()
            logger.debug(f"SQL query took {round(end - start, 3)} seconds")
            rename_cols(df=sql_uids_df, selection=selection, names=names)
            uids_df = sql_uids_df
        if engine in ["naive", "_test"]:
            memoization_tables = self._load_memoization_tables(conn=conn)
            logger.debug("Executing query naively...")
            naive_uids_df = Querier.execute_naive(
                vqs=vqs,
                fqs=fqs,
                selection=selection,
                memoization_tables=memoization_tables,
                filter_duplicates=drop_duplicates,
                table_evaluator=self.eval_df,
                visualize_steps_at=visualize_steps_at,
            )
            rename_cols(df=naive_uids_df, selection=selection, names=names)
            uids_df = naive_uids_df
        if engine == "_test":
            outcome, reason = compare_dfs_as_relations(
                df_1=sql_uids_df, df_2=naive_uids_df, return_reason=True
            )
            assert outcome, reason
        return self.eval_df(full_uids_df=uids_df, values=values, conn=conn)

    def _get_graph_and_names(
        self,
        objs: Tuple[Union[Ref, ValNode]],
        direction: Literal["forward", "backward", "both"] = "both",
        scope: Optional[Dict[str, Any]] = None,
        project: bool = False,
    ):
        vqs = {obj.query if isinstance(obj, Ref) else obj for obj in objs}
        vqs, fqs = traverse_all(vqs=vqs, direction=direction)
        hints = extract_names_from_scope(scope=scope) if scope is not None else {}
        g = InducedSubgraph(vqs=vqs, fqs=fqs)
        if project:
            v_proj, f_proj, _ = g.project()
            proj_hints = {v_proj[vq]: hints[vq] for vq in v_proj if vq in hints}
            names = get_names(
                hints=proj_hints,
                canonical_order=get_canonical_order(
                    vqs=set(v_proj.values()), fqs=set(f_proj.values())
                ),
            )
            final_vqs = set(v_proj.values())
            final_fqs = set(f_proj.values())
        else:
            v_proj = {vq: vq for vq in vqs}
            f_proj = {fq: fq for fq in fqs}
            names = get_names(
                hints=hints,
                canonical_order=get_canonical_order(vqs=set(vqs), fqs=set(fqs)),
            )
            final_vqs = vqs
            final_fqs = fqs
        return final_vqs, final_fqs, names, v_proj, f_proj

    def draw_graph(
        self,
        *objs: Union[Ref, ValNode],
        traverse: Literal["forward", "backward", "both"] = "backward",
        project: bool = False,
        show_how: Literal["none", "browser", "inline", "open"] = "browser",
    ):
        scope = inspect.currentframe().f_back.f_locals
        vqs, fqs, names, v_proj, f_proj = self._get_graph_and_names(
            objs,
            direction=traverse,
            scope=scope,
            project=project,
        )
        visualize_graph(vqs, fqs, names=names, show_how=show_how)

    def print_graph(
        self,
        *objs: Union[Ref, ValNode],
        project: bool = False,
        traverse: Literal["forward", "backward", "both"] = "backward",
    ):
        scope = inspect.currentframe().f_back.f_locals
        vqs, fqs, names, v_proj, f_proj = self._get_graph_and_names(
            objs,
            direction=traverse,
            scope=scope,
            project=project,
        )
        print_graph(
            vqs=vqs,
            fqs=fqs,
            names=names,
            selection=[
                v_proj[obj.query] if isinstance(obj, Ref) else obj for obj in objs
            ],
        )

    @transaction()
    def similar(
        self,
        *objs: Union[Ref, ValNode],
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        context: bool = False,
        verbose: Optional[bool] = None,
        local: bool = False,
        drop_duplicates: bool = True,
        engine: Literal["sql", "naive", "_test"] = None,
        _visualize_steps_at: Optional[Path] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        scope = inspect.currentframe().f_back.f_back.f_locals
        return self.df(
            *objs,
            direction="backward",
            scope=scope,
            values=values,
            context=context,
            skip_objs=False,
            verbose=verbose,
            local=local,
            drop_duplicates=drop_duplicates,
            engine=engine,
            _visualize_steps_at=_visualize_steps_at,
            conn=conn,
        )

    @transaction()
    def df(
        self,
        *objs: Union[Ref, ValNode],
        direction: Literal["forward", "backward", "both"] = "both",
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        context: bool = False,
        skip_objs: bool = False,
        verbose: Optional[bool] = None,
        local: bool = False,
        drop_duplicates: bool = True,
        engine: Literal["sql", "naive", "_test"] = None,
        _visualize_steps_at: Optional[Path] = None,
        scope: Optional[Dict[str, Any]] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """
        Universal query method over computational graphs, both imperative and
        declarative.
        """
        if verbose is None:
            verbose = Config.verbose_queries
        if not all(isinstance(obj, (Ref, ValNode)) for obj in objs):
            raise ValueError(
                "All arguments to df() must be either `Ref`s or `ValQuery`s."
            )
        #! important
        # We must sync any dirty cache elements to the db before performing a query.
        # If we don't, we'll query a store that might be missing calls and objs.
        self.commit(versioner=None)
        selection = [obj.query if isinstance(obj, Ref) else obj for obj in objs]
        # deps = get_deps(nodes=set(selection))
        vqs, fqs = traverse_all(vqs=set(selection), direction=direction)
        if scope is None:
            scope = inspect.currentframe().f_back.f_back.f_locals
        name_hints = extract_names_from_scope(scope=scope)
        v_map, f_map, target_selection, target_names = Querier.prepare_projection_query(
            vqs=vqs, fqs=fqs, selection=selection, name_hints=name_hints
        )
        target_vqs, target_fqs = set(v_map.values()), set(f_map.values())
        if context:
            g = InducedSubgraph(vqs=target_vqs, fqs=target_fqs)
            _, _, topsort = g.canonicalize()
            target_selection = [vq for vq in topsort if isinstance(vq, ValNode)]
        df = self.execute_query(
            selection=target_selection,
            vqs=set(v_map.values()),
            fqs=set(f_map.values()),
            values=values,
            names=target_names,
            verbose=verbose,
            drop_duplicates=drop_duplicates,
            visualize_steps_at=_visualize_steps_at,
            engine=engine,
            local=local,
            conn=conn,
        )
        for col in df.columns:
            try:
                df = df.sort_values(by=col)
            except Exception:
                continue
        if skip_objs:
            # drop the dtypes that are objects
            df = df.select_dtypes(exclude=["object"])
        return df

    def _make_terminal_data(self, func_op: FuncOp, call: Call) -> TerminalData:
        terminal_data = TerminalData(
            op_internal_name=func_op.sig.internal_name,
            op_version=func_op.sig.version,
            call_content_version=call.content_version,
            call_semantic_version=call.semantic_version,
            dep_key=get_dep_key_from_func(func=func_op.func),
        )
        return terminal_data

    @transaction()
    def lookup_call(
        self,
        func_op: FuncOp,
        pre_call_uid: str,
        input_uids: Dict[str, str],
        input_causal_uids: Dict[str, str],
        code_state: Optional[CodeState] = None,
        versioner: Optional[Versioner] = None,
        conn: Optional[Connection] = None,
    ) -> Optional[Call]:
        """
        Return a *detached* call for the given function and inputs, if it
        exists.
        """
        if not self.versioned:
            semantic_version = None
        else:
            assert code_state is not None
            component = get_dep_key_from_func(func=func_op.func)
            lookup_outcome = versioner.lookup_call(
                component=component, pre_call_uid=pre_call_uid, code_state=code_state
            )
            if lookup_outcome is None:
                return
            else:
                _, semantic_version = lookup_outcome
        causal_uid = func_op.get_call_causal_uid(
            input_uids=input_uids,
            input_causal_uids=input_causal_uids,
            semantic_version=semantic_version,
        )
        if self.cache.call_exists(uid=causal_uid, by_causal=True):
            return self.cache.call_get(uid=causal_uid, by_causal=True, lazy=True)
        call_uid = func_op.get_call_uid(
            pre_call_uid=pre_call_uid, semantic_version=semantic_version
        )
        if self.cache.call_exists(uid=call_uid, by_causal=False):
            return self.cache.call_get(uid=call_uid, by_causal=False, lazy=True)
        return None

    def call_batch(
        self, func_op: FuncOp, inputs: Dict[str, Ref]
    ) -> Tuple[List[Ref], CallStruct]:
        output_types = [Type.from_annotation(a) for a in func_op.output_annotations]
        outputs = [make_delayed(tp=tp) for tp in output_types]
        call_struct = CallStruct(func_op=func_op, inputs=inputs, outputs=outputs)
        return outputs, call_struct

    ############################################################################
    ### low-level provenance interfaces
    ############################################################################
    def get_creators(
        self, refs: List[Ref], prov_df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Optional[Call]], List[Optional[str]]]:
        """
        Given some Refs, return the
         - calls that created them (there may be at most one such call per Ref), or None if there was no such call.
         - the output name under which the refs were created
        """
        if not refs:
            return [], []
        if prov_df is None:
            prov_df = self.rel_storage.get_data(Config.provenance_table)
            prov_df = propagate_struct_provenance(prov_df=prov_df)
        causal_uids = list([ref.causal_uid for ref in refs])
        assert all(x is not None for x in causal_uids)
        res_df = prov_df.query('causal in @causal_uids and direction_new == "output"')[
            ["causal", "call_causal", "name", "op_id"]
        ].set_index("causal")[["call_causal", "name", "op_id"]]
        if len(res_df) == 0:
            return [None] * len(refs), [None] * len(refs)
        causal_to_creator_call_uid = res_df["call_causal"].to_dict()
        causal_to_output_name = res_df["name"].to_dict()
        causal_to_op_id = res_df["op_id"].to_dict()
        op_groups = res_df.groupby("op_id")["call_causal"].apply(list).to_dict()
        call_causal_to_call = {}
        for op_id, call_causal_list in op_groups.items():
            internal_name, version = Signature.parse_versioned_name(
                versioned_name=op_id
            )
            versioned_ui_name = self.sig_adapter.load_state()[
                internal_name, version
            ].versioned_ui_name
            op_calls = self.cache.call_mget(
                uids=call_causal_list,
                by_causal=True,
                versioned_ui_name=versioned_ui_name,
            )
            call_causal_to_call.update({call.causal_uid: call for call in op_calls})
        calls = [
            call_causal_to_call[causal_to_creator_call_uid[causal_uid]]
            if causal_uid in causal_to_creator_call_uid
            else None
            for causal_uid in causal_uids
        ]
        # if not len(set(causal_to_op_id.values())) == 1:
        #     raise NotImplementedError(f"Creators of refs from different ops not supported; found ops: {set(causal_to_op_id.values())}")
        # internal_name, version = Signature.parse_versioned_name(versioned_name=causal_to_op_id[causal_uids[0]])
        # versioned_ui_name = self.sig_adapter.load_state()[internal_name, version].versioned_ui_name
        # calls = self.cache.call_mget(uids=[causal_to_creator_call_uid[causal_uid] for causal_uid in causal_uids], by_causal=True,
        #                              versioned_ui_name=versioned_ui_name)
        output_names = [
            causal_to_output_name[causal_uid]
            if causal_uid in causal_to_output_name
            else None
            for causal_uid in causal_uids
        ]
        return calls, output_names

    def get_consumers(
        self, refs: List[Ref], prov_df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[List[Call]], List[List[str]]]:
        """
        Given some Refs, return the
         - calls that use them (there may be multiple such calls per Ref), or an empty list if there were no such calls.
         - the input names under which the refs were used
        """
        if prov_df is None:
            prov_df = self.rel_storage.get_data(Config.provenance_table)
            prov_df = propagate_struct_provenance(prov_df=prov_df)
        causal_uids = [ref.causal_uid for ref in refs]
        assert all(x is not None for x in causal_uids)
        res_groups = prov_df.query(
            'causal in @causal_uids and direction_new == "input"'
        )[["causal", "call_causal", "name", "op_id"]]
        op_to_causal_to_call_uids_and_inp_names = defaultdict(dict)
        for causal, call_causal, name, op_id in res_groups.itertuples(index=False):
            if causal not in op_to_causal_to_call_uids_and_inp_names[op_id]:
                op_to_causal_to_call_uids_and_inp_names[op_id][causal] = []
            op_to_causal_to_call_uids_and_inp_names[op_id][causal].append(
                (call_causal, name)
            )
        op_id_to_versioned_ui_name = {}
        for op_id in op_to_causal_to_call_uids_and_inp_names.keys():
            internal_name, version = Signature.parse_versioned_name(
                versioned_name=op_id
            )
            op_id_to_versioned_ui_name[op_id] = self.sig_adapter.load_state()[
                internal_name, version
            ].versioned_ui_name
        op_to_causal_to_calls_and_inp_names = defaultdict(dict)
        for (
            op_id,
            causal_to_call_uids_and_inp_names,
        ) in op_to_causal_to_call_uids_and_inp_names.items():
            versioned_ui_name = op_id_to_versioned_ui_name[op_id]
            op_calls = self.cache.call_mget(
                uids=[
                    elt[0]
                    for v in causal_to_call_uids_and_inp_names.values()
                    for elt in v
                ],
                by_causal=True,
                versioned_ui_name=versioned_ui_name,
            )
            call_causal_to_call = {call.causal_uid: call for call in op_calls}
            op_to_causal_to_calls_and_inp_names[op_id] = {
                causal: [
                    (call_causal_to_call[call_causal], name)
                    for call_causal, name in call_causal_list
                ]
                for causal, call_causal_list in causal_to_call_uids_and_inp_names.items()
            }
        concat_lists = lambda l: [elt for sublist in l for elt in sublist]
        calls = [
            concat_lists(
                [
                    [
                        v[0]
                        for v in op_to_causal_to_calls_and_inp_names[op_id].get(
                            causal_uid, []
                        )
                    ]
                    for op_id in op_to_causal_to_calls_and_inp_names.keys()
                ]
            )
            for causal_uid in causal_uids
        ]
        input_names = [
            concat_lists(
                [
                    [
                        v[1]
                        for v in op_to_causal_to_calls_and_inp_names[op_id].get(
                            causal_uid, []
                        )
                    ]
                    for op_id in op_to_causal_to_calls_and_inp_names.keys()
                ]
            )
            for causal_uid in causal_uids
        ]
        return calls, input_names

    ############################################################################
    ### provenance
    ############################################################################
    @transaction()
    def prov(
        self,
        ref: Ref,
        conn: Optional[Connection] = None,
        uids_only: bool = False,
        debug: bool = False,
    ):
        prov_df = self.rel_storage.get_data(Config.provenance_table, conn=conn)
        prov_df = prov_df.set_index([Provenance.causal_uid, Provenance.direction])
        x = provenance.ProvHelpers(storage=self, prov_df=prov_df)
        val_nodes, call_nodes = x.get_graph(full_uid=ref.full_uid)
        show_sources_as = "values" if not uids_only else "uids"
        printer = GraphPrinter(
            vqs=val_nodes,
            fqs=call_nodes,
            names=None,
            value_loader=ValueLoader(storage=self),
        )
        print(printer.print_computational_graph(show_sources_as=show_sources_as))
        if debug:
            visualize_graph(
                vqs=val_nodes, fqs=call_nodes, names=None, show_how="browser"
            )

    ############################################################################
    ### spawning contexts
    ############################################################################
    def _nest(self, **updates) -> contexts.Context:
        if contexts.GlobalContext.current is not None:
            return contexts.GlobalContext.current(**updates)
        else:
            result = contexts.Context(**updates)
            contexts.GlobalContext.current = result
            return result

    def __call__(self, **updates) -> contexts.Context:
        return self.run(**updates)

    def run(
        self,
        allow_calls: bool = True,
        debug_calls: bool = False,
        attach_call_to_outputs: bool = False,
        recompute_transient: bool = False,
        lazy: Optional[bool] = None,
        **updates,
    ) -> contexts.Context:
        # spawn context to execute or retrace calls
        lazy = not self.in_memory if lazy is None else lazy
        return self._nest(
            storage=self,
            allow_calls=allow_calls,
            debug_calls=debug_calls,
            recompute_transient=recompute_transient,
            _attach_call_to_outputs=attach_call_to_outputs,
            mode=MODES.run,
            lazy=lazy,
            **updates,
        )

    def query(self, **updates) -> contexts.Context:
        # spawn a context to define a query
        return self._nest(
            storage=self,
            mode=MODES.query,
            **updates,
        )

    def batch(self, **updates) -> contexts.Context:
        # spawn a context to execute calls in batch
        return self._nest(
            storage=self,
            mode=MODES.batch,
            **updates,
        )

    def noop(self) -> contexts.Context:
        return self._nest(
            storage=self,
            mode=MODES.noop,
        )

    ############################################################################
    ### remote sync operations
    ############################################################################
    @transaction()
    def sync_from_remote(self, conn: Optional[Connection] = None):
        if self.remote_manager is not None:
            self.remote_manager.sync_from_remote(conn=conn)

    @transaction()
    def sync_to_remote(self, conn: Optional[Connection] = None):
        if self.remote_manager is not None:
            self.remote_manager.sync_to_remote(conn=conn)

    @transaction()
    def sync_with_remote(self, conn: Optional[Connection] = None):
        if self.remote_manager is not None:
            self.sync_to_remote(conn=conn)
            self.sync_from_remote(conn=conn)

    ############################################################################
    ### refactoring
    ############################################################################
    @property
    def is_clean(self) -> bool:
        """
        Check that the storage has no uncommitted calls or objects.
        """
        return (
            self.cache.call_cache_by_causal.is_clean and self.cache.obj_cache.is_clean
        )

    def _check_rename_precondition(self, func: "funcs.FuncInterface"):
        """
        In order to rename function data, the function must be synced with the
        storage, and the storage must be clean
        """
        if not func._is_synchronized:
            raise RuntimeError("Cannot rename while function is not synchronized.")
        if not self.is_clean:
            raise RuntimeError("Cannot rename while there is uncommited work.")

    @transaction()
    def rename_func(
        self,
        func: "funcs.FuncInterface",
        new_name: str,
        conn: Optional[Connection] = None,
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
        func: "funcs.FuncInterface",
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


from . import funcs
from . import provenance

FuncInterface = funcs.FuncInterface


TP_TO_CLS = {
    AnyType: ValueRef,
    ListType: ListRef,
    DictType: DictRef,
    SetType: SetRef,
}


def make_delayed(tp: Type) -> Ref:
    return TP_TO_CLS[type(tp)](uid="", obj=Delayed(), in_memory=False)


class ValueLoader(ValueLoaderABC):
    def __init__(self, storage: Storage):
        self.storage = storage

    def load_value(self, full_uid: str) -> Any:
        uid, _ = Ref.parse_full_uid(full_uid)
        return self.storage.rel_adapter.obj_get(uid=uid)
