import datetime

import pymongo

from mandala_lite.storages.rels import RemoteEventLogEntry
from mandala_lite.storages.remote_storage import RemoteStorage


class MongoRemoteStorage(RemoteStorage):
    def __init__(self, db_name: str, client: pymongo.MongoClient):
        self.db_name = db_name
        self.client = client
        self.log = client[self.db_name].event_log

    def save_event_log_entry(self, entry: RemoteEventLogEntry):
        response = self.log.insert_one(entry)
        assert response.acknowledged
        # Set the timestamp based on the server time.
        self.log.update(
            {"_id": response.inserted_id},
            {"$currentDate": {"timestamp": {"$type" "date"}}},
        )

    def get_log_entries_since(
        self, timestamp: datetime.datetime
    ) -> list[RemoteEventLogEntry]:
        return [entry for entry in self.log.find({"timestamp": {"$gt": timestamp}})]
