from .kv import InMemoryStorage
from .rel_impls.duckdb_impl import DuckDBRelStorage
from .rels import RelAdapter
from ..common_imports import *
from ..core.config import Config
from ..core.model import Call
from ..core.sig import Signature


class Storage:
    """
    Groups together all the components of the storage system.

    Responsible for things that require multiple components to work together,
    e.g.
        - committing: moving calls from the "temporary" partition to the "main"
        partition. See also `CallStorage`.
        - synchronizing: connecting an operation with the storage and performing
        any necessary updates
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root
        self.call_cache = InMemoryStorage()
        self.obj_cache = InMemoryStorage()
        # all objects (inputs and outputs to operations, defaults) are saved here
        # stores the memoization tables
        self.rel_storage = DuckDBRelStorage()
        # manipulates the memoization tables
        self.rel_adapter = RelAdapter(rel_storage=self.rel_storage)
        # stores the signatures of the operations connected to this storage
        # (name, version) -> signature
        self.sigs: Dict[Tuple[str, int], Signature] = {}

    def call_exists(self, call_uid: str) -> bool:
        return self.call_cache.exists(call_uid) or self.rel_adapter.call_exists(
            call_uid
        )

    def call_get(self, call_uid: str) -> Call:
        if self.call_cache.exists(call_uid):
            return self.call_cache.get(call_uid)
        else:
            return self.rel_adapter.call_get(call_uid)

    def call_set(self, call_uid: str, call: Call) -> None:
        self.call_cache.set(call_uid, call)

    def obj_get(self, obj_uid: str) -> Any:
        if self.obj_cache.exists(obj_uid):
            return self.obj_cache.get(obj_uid)
        return self.rel_adapter.obj_get(obj_uid)

    def obj_set(self, obj_uid: str, value: Any) -> None:
        self.obj_cache.set(obj_uid, value)

    def preload_objs(self, keys: list[str]):
        keys_not_in_cache = [key for key in keys if not self.obj_cache.exists(key)]
        for idx, row in self.rel_adapter.obj_gets(keys_not_in_cache).iterrows():
            self.obj_cache.set(k=row[Config.uid_col], v=row["value"])

    def evict_caches(self):
        for k in self.call_cache.keys():
            self.call_cache.delete(k=k)
        for k in self.obj_cache.keys():
            self.obj_cache.delete(k=k)

    def commit(self):
        """
        Flush calls and objs from the cache that haven't yet been written to DuckDB.
        """
        new_objs = {
            key: self.obj_cache.get(key) for key in self.obj_cache.dirty_entries
        }
        new_calls = [self.call_cache.get(key) for key in self.call_cache.dirty_entries]
        self.rel_adapter.obj_sets(new_objs)
        self.rel_adapter.upsert_calls(new_calls)
        if Config.evict_on_commit:
            self.evict_caches()

        # Remove dirty bits from cache.
        self.obj_cache.dirty_entries.clear()
        self.call_cache.dirty_entries.clear()

    def synchronize(self, sig: Signature) -> Signature:
        """
        Synchronize an op's signature with this storage.

        - If this is a new operation, it's just added to the storage.
        - If this is an existing operation,
            - if the new signature is compatible with the old one, it is updated
            and returned. TODO: if a new input is created, a new column is
            created in the relation for this op.
            - otherwise, an error is raised
        """
        if (sig.name, sig.version) not in self.sigs:
            res = copy.deepcopy(sig)
            self.sigs[(res.name, res.version)] = res
            # create relation
            columns = list(res.input_names) + [
                f"output_{i}" for i in range(res.n_outputs)
            ]
            columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
            self.rel_storage.create_relation(
                name=res.name, columns=columns, primary_key=Config.uid_col
            )
            return res
        else:
            current = self.sigs[(sig.name, sig.version)]
            res = current.update(new=sig)
            # TODO: update relation if a new input was created
            return res
