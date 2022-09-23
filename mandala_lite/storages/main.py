from duckdb import DuckDBPyConnection as Connection
import datetime
from pypika import Query, Table, Parameter
import pyarrow.parquet as pq

from .rel_impls.utils import Transactable, transaction
from .kv import InMemoryStorage
from .rel_impls.duckdb_impl import DuckDBRelStorage
from .rels import RelAdapter, RemoteEventLogEntry
from .sigs import SigAdapter
from .remote_storage import RemoteStorage
from ..common_imports import *
from ..core.config import Config, dump_output_name
from ..core.model import Call
from ..core.sig import Signature


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
        root: Optional[Union[Path, RemoteStorage]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        self.root = root
        self.call_cache = InMemoryStorage()
        self.obj_cache = InMemoryStorage()
        # all objects (inputs and outputs to operations, defaults) are saved here
        # stores the memoization tables
        self.rel_storage = DuckDBRelStorage()
        # manipulates the memoization tables
        self.rel_adapter = RelAdapter(rel_storage=self.rel_storage)
        # manipulates signatures
        self.sig_adapter = SigAdapter(
            rel_adapter=self.rel_adapter, sigs={}, root=self.root
        )
        # stores the signatures of the operations connected to this storage
        # (name, version) -> signature
        # self.sigs: Dict[Tuple[str, int], Signature] = {}

        # self.remote_sync_manager = None
        # # manage remote storage
        # if isinstance(root, RemoteStorage):
        #     self.remote_sync_manager = RemoteSyncManager(
        #         local_storage=self, remote_storage=root
        #     )
        self.last_timestamp = (
            timestamp if timestamp is not None else datetime.datetime.fromtimestamp(0)
        )

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
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
        return self.rel_adapter.obj_get(uid=obj_uid)

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
        # event_log_df = self.rel_storage.get_data(
        #     self.rel_adapter.EVENT_LOG_TABLE, conn=conn
        # )
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
            tables_with_changes, to="internal"
        )
        output = {}
        for table_name, table in tables_with_changes.items():
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            output[table_name] = buffer.getvalue()
        self.rel_adapter.clear_event_log(conn=conn)
        print(output)
        return output

    @transaction()
    def apply_from_remote(
        self, changes: list[RemoteEventLogEntry], conn: Optional[Connection] = None
    ):
        """
        Apply new calls from the remote server.

        NOTE: this also renames tables and columns to their UI names.
        """
        data = {}
        for raw_changeset in changes:
            for table_name in raw_changeset:
                buffer = io.BytesIO(raw_changeset[table_name])
                table = pq.read_table(buffer)
                data[table_name] = table
        # pass to UI names
        data = self.sig_adapter.rename_tables(tables=data, to="ui")
        for table_name, table in data.items():
            self.rel_storage.upsert(table_name, table, conn=conn)

    def sync_from_remote(self):
        """
        Pull new calls from the remote server.

        Note that the server's schema (i.e. signatures) can be a super-schema of
        the local schema, but all local schema elements must be present in the
        remote schema, because this is enforced by how schema updates are
        performed.
        """
        if not isinstance(self.root, RemoteStorage):
            return
        # apply signature changes from the server
        self.sig_adapter.sync_from_remote()
        # next, pull new calls
        new_log_entries, timestamp = self.root.get_log_entries_since(
            self.last_timestamp
        )
        self.apply_from_remote(new_log_entries)
        self.last_timestamp = timestamp

    def sync_to_remote(self):
        """
        Send calls to the remote server.

        As with `sync_from_remote`, the server may have a super-schema of the
        local schema.
        """
        if not isinstance(self.root, RemoteStorage):
            # todo: there should be a way to completely ignore the event log
            # when there's no remote
            self.rel_adapter.clear_event_log()
        else:
            # apply signature changes from the server
            self.sig_adapter.sync_from_remote()
            changes = self.bundle_to_remote()
            self.root.save_event_log_entry(changes)

    def sync_with_remote(self):
        if not isinstance(self.root, RemoteStorage):
            return
        self.sync_to_remote()
        self.sync_from_remote()

    ############################################################################
    ### signature sync, renaming, refactoring
    ############################################################################
    @property
    def is_clean(self) -> bool:
        """
        Check that the storage has no uncommitted calls or objects.
        """
        return (
            self.call_cache.is_clean and self.obj_cache.is_clean
        )  # and self.rel_adapter.event_log_is_clean()
