# from ...core.sig import Signature
# from ...common_imports import *
# import mongomock
# import datetime
#
# from mandala.storages.rels import RemoteEventLogEntry
# from mandala.storages.remote_storage import RemoteStorage
#
#
# class MongoMockRemoteStorage(RemoteStorage):
#     def __init__(self, db_name: str, client: mongomock.MongoClient):
#         self.db_name = db_name
#         self.client = client
#         self.log = client.experiment_data[self.db_name].event_log
#         self.sigs: Dict[Tuple[str, int], Signature] = {}
#
#     def pull_signatures(self) -> Dict[Tuple[str, int], Signature]:
#         return self.sigs
#
#     def push_signatures(self, new_sigs: Dict[Tuple[str, int], Signature]) -> None:
#         current_internal_sigs = self.sigs
#         for (internal_name, version), new_sig in new_sigs.items():
#             if (internal_name, version) in current_internal_sigs:
#                 current_sig = current_internal_sigs[(internal_name, version)]
#                 if not current_sig.is_compatible(new_sig):
#                     raise ValueError(
#                         f"Signature {internal_name}:{version} is incompatible with {new_sig}"
#                     )
#         self.sigs = new_sigs
#
#     def save_event_log_entry(self, entry: RemoteEventLogEntry):
#         response = self.log.insert_one({"tables": entry})
#         assert response.acknowledged
#         # Set the timestamp based on the server time.
#         self.log.update_one(
#             {"_id": response.inserted_id},
#             {"$currentDate": {"timestamp": {"$type": "date"}}},
#         )
#
#     def get_log_entries_since(
#         self, timestamp: datetime.datetime
#     ) -> Tuple[List[RemoteEventLogEntry], datetime.datetime]:
#         logger.debug(f"Getting log entries since {timestamp}")
#         entries = []
#         last_timestamp = datetime.datetime.fromtimestamp(0)
#         for entry in self.log.find({"timestamp": {"$gt": timestamp}}):
#             entries.append(entry["tables"])
#             if entry["timestamp"] > last_timestamp:
#                 last_timestamp = entry["timestamp"]
#         return entries, last_timestamp
#
