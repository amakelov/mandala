import datetime
from typing import Tuple, List

import pymongo

from mandala.storages.rels import RemoteEventLogEntry
from mandala.storages.remote_storage import RemoteStorage


class MongoRemoteStorage(RemoteStorage):
    def __init__(self, db_name: str, client: pymongo.MongoClient):
        self.db_name = db_name
        self.client = client
        self.log = client.experiment_data[self.db_name].event_log

    def save_event_log_entry(self, entry: RemoteEventLogEntry):
        response = self.log.insert_one({"tables": entry})
        assert response.acknowledged
        # Set the timestamp based on the server time.
        self.log.update_one(
            {"_id": response.inserted_id},
            {"$currentDate": {"timestamp": {"$type": "date"}}},
        )

    def get_log_entries_since(
        self, timestamp: datetime.datetime
    ) -> Tuple[List[RemoteEventLogEntry], datetime.datetime]:
        entries = []
        last_timestamp = datetime.datetime.fromtimestamp(0)
        for entry in self.log.find({"timestamp": {"$gt": timestamp}}):
            entries.append(entry["tables"])
            if entry["timestamp"] > last_timestamp:
                last_timestamp = entry["timestamp"]
        return entries, last_timestamp
