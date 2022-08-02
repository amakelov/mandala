from .kv import InMemoryStorage
from .rel_impls.duckdb_impl import DuckDBRelStorage
from .rels import RelAdapter
from .remote_storage import RemoteStorage, RemoteSyncManager
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

    def __init__(self, root: Optional[Union[Path, RemoteStorage]] = None):
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

        self.remote_sync_manager = None
        # manage remote storage
        if isinstance(root, RemoteStorage):
            self.remote_sync_manager = RemoteSyncManager(
                local_storage=self.rel_adapter, remote_storage=root
            )

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

    def sync_with_remote(self):
        if self.remote_sync_manager is not None:
            self.remote_sync_manager.sync_to_remote()
            self.remote_sync_manager.sync_from_remote()

    ############################################################################
    ### synchronization, renaming, refactoring
    ############################################################################
    @property
    def is_clean(self) -> bool:
        return self.call_cache.is_clean and self.obj_cache.is_clean

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
        conn = self.rel_adapter._get_connection()
        if not self.rel_adapter.has_signature(
            name=sig.name, version=sig.version, conn=conn
        ):
            new_sig = sig._generate_internal()
            self.rel_adapter.write_signature(sig=new_sig, conn=conn)
            # create relation
            columns = list(new_sig.input_names) + [
                f"output_{i}" for i in range(new_sig.n_outputs)
            ]
            columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
            self.rel_storage.create_relation(
                name=new_sig.versioned_name,
                columns=columns,
                primary_key=Config.uid_col,
                conn=conn,
            )
        else:
            current = self.rel_adapter.get_signature(
                name=sig.name, version=sig.version, conn=conn
            )
            new_sig, updates = current.update(new=sig)
            # create new inputs, if any
            for new_input, default_value in updates.items():
                default_uid = new_sig._new_input_defaults_uids[new_input]
                self.rel_storage.create_column(
                    relation=new_sig.versioned_name,
                    name=new_input,
                    default_value=default_uid,
                    conn=conn,
                )
                # insert the default in the objects *in the database*, if it's
                # not there already
                self.rel_adapter.obj_set(
                    key=default_uid, value=default_value, conn=conn
                )
            self.rel_adapter.write_signature(sig=new_sig, conn=conn)
        self.rel_adapter._end_transaction(conn=conn)
        return new_sig
