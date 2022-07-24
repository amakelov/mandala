from ..common_imports import *
from ..core.model import unwrap
from ..storages.main import Storage
from ..queries.weaver import ValQuery
from ..queries.compiler import traverse_all, solve_query, Compiler


class MODES:
    run = "run"
    query = "query"


class GlobalContext:
    current: Optional["Context"] = None


class Context:
    OVERRIDES = {}

    def __init__(
        self, storage: Storage = None, mode: str = MODES.run, lazy: bool = False
    ):
        self.storage = storage
        self.mode = self.OVERRIDES.get("mode", mode)
        self.lazy = self.OVERRIDES.get("lazy", lazy)
        self.updates = {}
        self._updates_stack = []

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
        if not all(k in ("storage", "mode", "lazy") for k in updates.keys()):
            raise ValueError(updates.keys())
        ### backup state
        before_update = self._backup_state(keys=updates.keys())
        self._updates_stack.append(before_update)
        ### apply updates
        for k, v in updates.items():
            self.__dict__[f"{k}"] = v
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self._updates_stack:
            raise RuntimeError("No context to exit from")
        # commit calls from temp partition to main and tabulate them
        self.storage.commit()
        # undo updates
        ascent_updates = self._updates_stack.pop()
        for k, v in ascent_updates.items():
            self.__dict__[f"{k}"] = v
        # unlink from global if done
        if len(self._updates_stack) == 0:
            GlobalContext.current = None
        if exc_type:
            raise exc_type(exc_value).with_traceback(exc_traceback)
        return None

    def __call__(self, storage: Storage, **updates):
        self.updates = {"storage": storage, **updates}
        return self

    def get_table(self, *queries: ValQuery) -> pd.DataFrame:
        # ! EXTREMELY IMPORTANT
        # We must sync any dirty cache elements to the DuckDB store before performing a query.
        # If we don't, we'll query a store that might be missing calls and objs.
        self.storage.commit()

        select_queries = list(queries)
        val_queries, func_queries = traverse_all(select_queries)
        compiler = Compiler(val_queries=val_queries, func_queries=func_queries)
        query = compiler.compile(select_queries=select_queries)
        df = self.storage.rel_storage.execute(query=str(query))
        # now, evaluate the table
        keys_to_collect = [
            item for _, column in df.iteritems() for _, item in column.iteritems()
        ]
        self.storage.preload_objs(keys_to_collect)
        result = df.applymap(lambda key: unwrap(self.storage.obj_get(key)))
        # finally, name the columns
        result.columns = [
            f"unnamed_{i}" if query.column_name is None else query.column_name
            for i, query in zip(range(len((result.columns))), queries)
        ]
        return result


class RunContext(Context):
    OVERRIDES = {"mode": MODES.run, "lazy": False}


class QueryContext(Context):
    OVERRIDES = {
        "mode": MODES.query,
    }


run = RunContext()
query = QueryContext()
