from typing import Literal

from ..common_imports import *
from ..core.config import Config
from ..core.model import Call
from ..core.workflow import Workflow

from ..deps.versioner import CodeState, Versioner
from .utils import MODES

from ..core.weaver import (
    qwrap,
)


class GlobalContext:
    current: Optional["Context"] = None


class Context:
    def __init__(
        self,
        storage: "storage.Storage" = None,
        mode: str = MODES.run,
        lazy: bool = False,
        allow_calls: bool = True,
        debug_calls: bool = False,
        recompute_transient: bool = False,
        _attach_call_to_outputs: bool = False,  # for debugging
        debug_truncate: Optional[int] = 20,
    ):
        self.storage = storage
        self.mode = mode
        self.lazy = lazy
        self.allow_calls = allow_calls
        self.debug_calls = debug_calls
        self.recompute_transient = recompute_transient
        self._attach_call_to_outputs = _attach_call_to_outputs
        self.debug_truncate = debug_truncate
        self.updates = {}
        self._updates_stack = []
        self._call_structs = []
        self._defined_funcs: List["FuncInterface"] = []
        self._call_buffer: List[Call] = []
        self._code_state: CodeState = None
        self._cached_versioner: Versioner = None

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
        is_top = len(self._updates_stack) == 0
        ### verify update keys
        updates = self.updates
        if not all(
            k
            in (
                "storage",
                "mode",
                "lazy",
                "allow_calls",
                "debug_calls",
                "recompute_transient",
                "_attach_call_to_outputs",
            )
            for k in updates.keys()
        ):
            raise ValueError(updates.keys())
        if "mode" in updates.keys() and updates["mode"] not in MODES.all_:
            raise ValueError(updates["mode"])
        ### backup state
        before_update = self._backup_state(keys=updates.keys())
        # self._updates_stack.append(before_update)
        ### apply updates
        for k, v in updates.items():
            if v is not None:
                self.__dict__[f"{k}"] = v
        # Load state from remote
        if self.storage is not None:
            # self.storage.sync_with_remote()
            self.storage.sync_from_remote()
        if (
            self.mode in (MODES.run, MODES.query)
            and self.storage is not None
            and self.storage.versioned
        ):
            storage = self.storage
            if is_top:
                versioner, code_state = storage.sync_code()
                self._cached_versioner = versioner
                self._code_state = code_state
        # this is last so that any exceptions don't leave the context in an
        # inconsistent state
        self._updates_stack.append(before_update)
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
        if exc_type is None:
            if self.mode == MODES.run:
                if self.storage is not None:
                    # commit calls from temp partition to main and tabulate them
                    if Config.autocommit:
                        self.storage.commit(versioner=self._cached_versioner)
                    self.storage.sync_to_remote()
            elif self.mode == MODES.query:
                pass
            elif self.mode == MODES.batch:
                executor = SimpleWorkflowExecutor()
                workflow = Workflow.from_call_structs(self._call_structs)
                calls = executor.execute(workflow=workflow, storage=self.storage)
                self.storage.commit(calls=calls, versioner=self._cached_versioner)
            else:
                raise InternalError(self.mode)
        self._undo_updates()
        return None

    def __call__(
        self,
        storage: Optional["Storage"] = None,
        allow_calls: bool = True,
        debug_calls: bool = False,
        recompute_transient: bool = False,
        _attach_call_to_outputs: bool = False,
        **updates,
    ):
        self.updates = {
            "storage": storage,
            "allow_calls": allow_calls,
            "debug_calls": debug_calls,
            "recompute_transient": recompute_transient,
            "_attach_call_to_outputs": _attach_call_to_outputs,
            **updates,
        }
        return self

    def get_table(
        self,
        *queries: Any,
        values: Literal["objs", "refs", "uids", "lazy"] = "objs",
        constrain_versions: bool = True,
        _engine: str = "sql",
        _filter_duplicates: bool = True,
        _visualize_steps_at: Optional[Path] = None,
    ) -> pd.DataFrame:
        #! important
        # We must sync any dirty cache elements to the DuckDB store before performing a query.
        # If we don't, we'll query a store that might be missing calls and objs.
        wrapped_queries = [qwrap(q) for q in queries]
        self.storage.commit(versioner=None)
        return self.storage.execute_query(
            select_queries=wrapped_queries,
            engine=_engine,
            values=values,
            filter_duplicates=_filter_duplicates,
            visualize_steps_at=_visualize_steps_at,
            constrain_versions=constrain_versions,
        )


from . import storage
from .executors import SimpleWorkflowExecutor
