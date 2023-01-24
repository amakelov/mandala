import abc
import datetime
from abc import abstractmethod
from typing import Optional

from ..common_imports import *
from ..core.sig import Signature

from mandala.storages.rels import RemoteEventLogEntry, RelAdapter


class RemoteStorage(abc.ABC):
    @abstractmethod
    def save_event_log_entry(self, entry: RemoteEventLogEntry):
        raise NotImplementedError()

    @abstractmethod
    def get_log_entries_since(
        self, timestamp: datetime.datetime
    ) -> Tuple[List[RemoteEventLogEntry], datetime.datetime]:
        raise NotImplementedError()

    @abstractmethod
    def pull_signatures(self) -> Dict[Tuple[str, int], Signature]:
        raise NotImplementedError()

    @abstractmethod
    def push_signatures(self, new_sigs: Dict[Tuple[str, int], Signature]) -> None:
        raise NotImplementedError()


# class RemoteSyncManager:
#     def __init__(
#         self,
#         local_storage: RelAdapter,
#         remote_storage: RemoteStorage,
#         timestamp: Optional[datetime.datetime] = None,
#     ):
#         self.local_storage = local_storage
#         self.remote_storage = remote_storage
#         self.last_timestamp = (
#             timestamp if timestamp is not None else datetime.datetime.fromtimestamp(0)
#         )
#
#     def sync_from_remote(self):
#         new_log_entries, timestamp = self.remote_storage.get_log_entries_since(
#             self.last_timestamp
#         )
#         self.local_storage.apply_from_remote(new_log_entries)
#         self.last_timestamp = timestamp
#
#     def sync_to_remote(self):
#         changes = self.local_storage.bundle_to_remote()
#         self.remote_storage.save_event_log_entry(changes)
#
