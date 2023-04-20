from pypika import Table, Query
import pyarrow.parquet as pq

from ..common_imports import *
from ..core.config import Config
from ..storages.rels import RelAdapter, RemoteEventLogEntry
from ..storages.rel_impls.bases import RelStorage
from ..storages.sigs import SigSyncer, SigAdapter
from ..storages.remote_storage import RemoteStorage
from ..storages.rel_impls.utils import Transactable, transaction, Connection


class RemoteManager(Transactable):
    def __init__(
        self,
        rel_adapter: RelAdapter,
        sig_adapter: SigAdapter,
        rel_storage: RelStorage,
        sig_syncer: SigSyncer,
        root: RemoteStorage,
    ):
        self.rel_adapter = rel_adapter
        self.sig_adapter = sig_adapter
        self.rel_storage = rel_storage
        self.sig_syncer = sig_syncer
        self.root = root

    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

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
        self, changes: List[RemoteEventLogEntry], conn: Optional[Connection] = None
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
