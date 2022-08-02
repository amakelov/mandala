import mongomock

from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_remote():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
    # create a *single* (mock) remote database
    client = mongomock.MongoClient()

    # create multiple storages connected to it
    storage_1 = Storage(root=MongoMockRemoteStorage(db_name="test", client=client))
    storage_2 = Storage(root=MongoMockRemoteStorage(db_name="test", client=client))

    # do work with one storage
    @op
    def add(x: int, y: int = 42) -> int:
        return x + y

    with run(storage_1):
        add(23, 42)
    storage_1.sync_with_remote()

    # sync with the other storage
    storage_2.sync_with_remote()

    # verify equality of relational data
    data_1 = storage_1.rel_storage.get_all_data()
    data_2 = storage_2.rel_storage.get_all_data()
    assert data_1.keys() == data_2.keys()
    # TODO: test for schema equality once we implement this in sync
    assert all(
        {k: (data_1[k]) == data_2[k] for k in data_1 if k != Config.schema_table}
    )
