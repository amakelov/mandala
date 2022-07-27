import mongomock

from mandala_lite.all import *
from mandala_lite.storages.remote_impls.mongo_mock import MongoMockRemoteStorage


def test_remote():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
    # create a *single* (mock) remote database
    client = mongomock.MongoClient()

    # create multiple storages connected to it
    storage_1 = Storage(root=MongoMockRemoteStorage(db_name="test", client=client))
    storage_2 = Storage(root=MongoMockRemoteStorage(db_name="test", client=client))

    # do work with one storage
    @op(storage_1)
    def add(x: int, y: int = 42) -> int:
        return x + y

    add(23, 42)
    storage_1.sync_with_remote()

    # sync with the other storage
    @op(storage_2)
    def add(x: int, y: int = 42) -> int:
        return x + y

    storage_2.sync_with_remote()

    # verify equality of relational data
    data_1 = storage_1.rel_storage.get_all_data()
    data_2 = storage_2.rel_storage.get_all_data()
    assert data_1.keys() == data_2.keys()
    assert all({k: (data_1[k]) == data_2[k] for k in data_1})
