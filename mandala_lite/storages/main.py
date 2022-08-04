from duckdb import DuckDBPyConnection as Connection

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
            self.pull_signatures()
            self.remote_sync_manager.sync_from_remote()

    def pull_signatures(self):
        """
        Pull the current state of the signatures from the remote, make sure that
        they are compatible with the current ones, and then update or create
        according to the new signatures.
        """
        if self.remote_sync_manager is not None:
            assert isinstance(self.root, RemoteStorage)
            new_sigs = self.root.pull_signatures()
            new_sigs = self.remote_sync_manager.remote_storage.pull_signatures()
            self.synchronize_many(sigs=new_sigs, sync_signatures=False)

    def push_signatures(self):
        if self.remote_sync_manager is not None:
            assert isinstance(self.root, RemoteStorage)
            current_sigs = list(self.rel_adapter.signature_gets().values())
            self.root.push_signatures(current_sigs)

    ############################################################################
    ### synchronization, renaming, refactoring
    ############################################################################
    @property
    def is_clean(self) -> bool:
        return (
            self.call_cache.is_clean and self.obj_cache.is_clean
        )  # and self.rel_adapter.event_log_is_clean()

    def _rename_funcs(self, mapping: Dict[str, Tuple[str, str]], conn: Connection):
        """
        Rename many functions in a single transaction.

        NOTE: such a method is necessary to handle cases when some names were
        permuted.

        Args:
            mapping (Dict[str, Tuple[str, str]]): {internal name: (current ui name, new ui name)
        """
        current_ui_sigs = self.rel_adapter.signature_gets(conn=conn)
        current_ui_names = [elt[0] for elt in current_ui_sigs.keys()]
        name_mapping = {
            current_ui_name: new_ui_name
            for current_ui_name, new_ui_name in mapping.values()
        }
        new_ui_names = [
            name_mapping.get(ui_name, ui_name) for ui_name in current_ui_names
        ]
        if len(set(new_ui_names)) != len(new_ui_names):
            raise ValueError()
        # rename in signature object
        for internal_name, (current_ui_name, new_ui_name) in mapping.items():
            sigs = [
                sig
                for (ui_name, version), sig in current_ui_sigs.items()
                if ui_name == current_ui_name
            ]
            for sig in sigs:
                new_sig = sig.rename(new_name=new_ui_name)
                self.rel_adapter.signature_set(sig=new_sig, conn=conn)
                self.rel_storage.rename_relation(
                    name=sig.versioned_ui_name,
                    new_name=new_sig.versioned_ui_name,
                    conn=conn,
                )

    def _rename_args(
        self,
        internal_name: str,
        version: int,
        mapping: Dict[str, str],
        conn: Connection,
    ):
        """
        Rename the arguments of a function with internal name `internal_name`,
        accoding to the given mapping
        """
        current_internal_sigs = self.rel_adapter.signature_gets(
            use_ui_names=False, conn=conn
        )
        sig = current_internal_sigs[internal_name, version]
        self.rel_storage.rename_columns(
            relation=sig.versioned_ui_name, mapping=mapping, conn=conn
        )
        new_sig = sig.rename_inputs(mapping=mapping)
        self.rel_adapter.signature_set(sig=new_sig, conn=conn)

    def _create_function(self, sig: Signature, conn: Connection):
        print("This")
        print(self)
        """
        Given a signature with internal data that does not exist in this
        storage, this *both* puts a signature in the signature storage, and
        creates a memoization table in the database.
        """
        assert sig.has_internal_data
        internal_sigs = self.rel_adapter.signature_gets(use_ui_names=False, conn=conn)
        assert (sig.internal_name, sig.version) not in internal_sigs.keys()
        self.rel_adapter.signature_set(sig=sig, conn=conn)
        # create relation
        columns = list(sig.input_names) + [f"output_{i}" for i in range(sig.n_outputs)]
        columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
        self.rel_storage.create_relation(
            name=sig.versioned_ui_name,
            columns=columns,
            primary_key=Config.uid_col,
            conn=conn,
        )

    def _update_function(self, sig: Signature, conn: Connection):
        """
        Given a signature with or without internal data that exists in this
        storage, this *both* updates the memoization table in the database, and the
        signature in the signature storage.
        """
        ui_sigs = self.rel_adapter.signature_gets(use_ui_names=True, conn=conn)
        assert (sig.ui_name, sig.version) in ui_sigs.keys()
        current = ui_sigs[(sig.ui_name, sig.version)]
        # the `update` method also ensures that the signature is compatible
        new_sig, updates = current.update(new=sig)
        # create new inputs, if any
        for new_input, default_value in updates.items():
            internal_input_name = new_sig.ui_to_internal_input_map[new_input]
            default_uid = new_sig._new_input_defaults_uids[internal_input_name]
            self.rel_storage.create_column(
                relation=new_sig.versioned_ui_name,
                name=new_input,
                default_value=default_uid,
                conn=conn,
            )
            # insert the default in the objects *in the database*, if it's
            # not there already
            self.rel_adapter.obj_set(key=default_uid, value=default_value, conn=conn)
        self.rel_adapter.signature_set(sig=new_sig, conn=conn)

    def _create_new_version(
        self,
        sig: Signature,
        current_ui_sigs: Dict[Tuple[str, int], Signature],
        conn: Connection,
    ):
        ui_name, version = sig.ui_name, sig.version
        highest_current_version = max(
            [elt[1] for elt in current_ui_sigs.keys() if elt[0] == ui_name]
        )
        if not version == highest_current_version + 1:
            raise ValueError()
        internal_name = current_ui_sigs[
            (ui_name, highest_current_version)
        ].internal_name
        if sig.has_internal_data:
            raise NotImplementedError()
        new_sig = sig._generate_internal(internal_name=internal_name)
        self._create_function(sig=new_sig, conn=conn)

    def synchronize_many(
        self, sigs: List[Signature], sync_signatures: bool = True
    ) -> List[Signature]:
        """
        Universal method to synchronize many signatures and reject if a
        signature is incompatible.

        This handles everything:
            - any renamings for all signatures with internal data
            - generating new internal data for signatures that don't have it
            - updating, creating new verions
        """
        ### by default, we sync the signatures with remote before making any
        ### changes
        if sync_signatures:
            self.pull_signatures()
        conn = self.rel_adapter._get_connection()
        current_internal_sigs = self.rel_adapter.signature_gets(
            use_ui_names=False, conn=conn
        )
        current_ui_sigs = self.rel_adapter.signature_gets(use_ui_names=True, conn=conn)
        # figure out and apply function renamings
        function_renamings = {}  # internal name: (current, new)
        for sig in sigs:
            if not sig.has_internal_data:
                continue
            internal_name, version = sig.internal_name, sig.version
            if (internal_name, version) in current_internal_sigs.keys():
                current_sig = current_internal_sigs[internal_name, version]
                if sig.ui_name != current_sig.ui_name:
                    function_renamings[internal_name] = (
                        current_sig.ui_name,
                        sig.ui_name,
                    )
        self._rename_funcs(mapping=function_renamings, conn=conn)
        # figure out and apply arg renamings
        for sig in sigs:
            if not sig.has_internal_data:
                continue
            internal_name, version = sig.internal_name, sig.version
            if (internal_name, version) in current_internal_sigs.keys():
                current_sig = current_internal_sigs[internal_name, version]
                if sig.input_names != current_sig.input_names:
                    mapping = {}
                    current_int_to_ui = {
                        v: k for k, v in current_sig.ui_to_internal_input_map.items()
                    }
                    new_int_to_ui = {
                        v: k for k, v in sig.ui_to_internal_input_map.items()
                    }
                    for (
                        current_internal_arg,
                        current_ui_arg,
                    ) in current_int_to_ui.items():
                        new_ui_name = new_int_to_ui[current_internal_arg]
                        if new_ui_name != current_ui_arg:
                            mapping[current_ui_arg] = new_ui_name
                    self._rename_args(
                        internal_name, version=version, mapping=mapping, conn=conn
                    )
        #! reload current state after renaming
        current_internal_sigs = self.rel_adapter.signature_gets(
            use_ui_names=False, conn=conn
        )
        current_ui_sigs = self.rel_adapter.signature_gets(use_ui_names=True, conn=conn)
        # create/update new funcs
        for sig in sigs:
            if not sig.has_internal_data:
                ui_name, version = sig.ui_name, sig.version
                if (ui_name, version) in current_ui_sigs:
                    # updating an existing function in-place
                    self._update_function(sig=sig, conn=conn)
                elif ui_name in [elt[0] for elt in current_ui_sigs.keys()]:
                    # a new version of an existing function
                    self._create_new_version(
                        sig=sig, current_ui_sigs=current_ui_sigs, conn=conn
                    )
                else:
                    # create new function
                    if not sig.has_internal_data:
                        sig = sig._generate_internal()
                    self._create_function(sig=sig, conn=conn)
            else:
                internal_name, version = sig.internal_name, sig.version
                if (internal_name, version) in current_internal_sigs.keys():
                    self._update_function(sig=sig, conn=conn)
                elif internal_name in [elt[0] for elt in current_internal_sigs.keys()]:
                    self._create_new_version(
                        sig=sig, current_ui_sigs=current_ui_sigs, conn=conn
                    )
                else:
                    self._create_function(sig=sig, conn=conn)
        # finally, load the new signature objects
        current_ui_sigs = self.rel_adapter.signature_gets(use_ui_names=True, conn=conn)
        result = [current_ui_sigs[(sig.ui_name, sig.version)] for sig in sigs]
        self.rel_adapter._end_transaction(conn=conn)
        ### by default, we send the new signatures to the remote
        if sync_signatures:
            self.push_signatures()
        return result
