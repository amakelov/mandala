import mongomock

from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_remote_simple():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
    # create a *single* (mock) remote database
    client = mongomock.MongoClient()

    # create multiple storages connected to it
    root = MongoMockRemoteStorage(db_name="test", client=client)
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    def show_tables():  # for debugging
        print(storage_1.rel_adapter.get_call_tables())
        print(storage_2.rel_adapter.get_call_tables())

    def check_all_invariants():
        check_invariants(storage=storage_1)
        check_invariants(storage=storage_2)

    # do work with one storage
    @op
    def add(x: int, y: int = 42) -> int:
        return x + y

    with run(storage_1):
        add(23, 42)
    storage_1.sync_with_remote()
    check_all_invariants()

    # sync with the other storage
    storage_2.sync_with_remote()
    check_all_invariants()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)

    # verify equality of relational data
    compare_data(storage_1=storage_1, storage_2=storage_2)


def test_create_func():
    """
    Unit test for function creation in a multi-user setting.

    Specifically, test a scenario where:
    - one user defines a function
    - another user defines the same function under the same name

    Expected behavior:
    - the first creator wins: both functions are assigned the UID issued to
    the first creator.
    """
    client = mongomock.MongoClient()
    root = MongoMockRemoteStorage(db_name="test", client=client)
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    @op(ui_name="f")
    def f_1(x: int) -> int:
        return x + 1

    @op(ui_name="f")
    def f_2(x: int) -> int:
        return x + 1

    synchronize(func=f_1, storage=storage_1)
    synchronize(func=f_2, storage=storage_2)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)


def test_add_input():
    """
    Unit test for adding an input to a function in a multi-user setting.

    Specifically, test a scenario where:
    - one user defines a function
    - another user defines the same function
    - the first user adds an input to the function
    - the second user tries to send calls to the initial function  to the server

    Expected behavior:
    - when the second user commits the new calls:
        - the new signature is pulled from the server and applied
        - the call objects are entered into the updated table (which now has a
          default value for the column with the new input)
        - the tables sent to the remote server are of the correct signature
    - on the remote, everything looks as if the second user had defined the
      function with the new input from the start
    """
    client = mongomock.MongoClient()
    root = MongoMockRemoteStorage(db_name="test", client=client)
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    @op(ui_name="inc")
    def inc_1(x: int) -> int:
        return x + 1

    @op(ui_name="inc")
    def inc_2(x: int) -> int:
        return x + 1

    synchronize(func=inc_1, storage=storage_1)
    synchronize(func=inc_2, storage=storage_2)

    @op(ui_name="inc")
    def inc_1(x: int, how_many_times: int = 1) -> int:
        return x + how_many_times

    synchronize(func=inc_1, storage=storage_1)
    assert not signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    with run(storage_2):
        for i in range(10):
            inc_2(i)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)


def _test_rename_func():
    """
    Unit test for renaming a function in a multi-user setting.

    Specifically, test a scenario where:
    - one user defines a function `f`
    - another user defines the same function
    - the first user renames the function to `g`
    - unbeknownst to that, the second user still sees `f` and uses it in
    computations.

    Expected behavior:
    - calls to the current `f` go into the `g` table on the remote
    """
    client = mongomock.MongoClient()
    root = MongoMockRemoteStorage(db_name="test", client=client)
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    @op(ui_name="f")
    def f_1(x: int) -> int:
        return x + 1

    @op(ui_name="f")
    def f_2(x: int) -> int:
        return x + 1

    synchronize(func=f_1, storage=storage_1)
    synchronize(func=f_2, storage=storage_2)

    rename_func(storage=storage_1, func=f_1, new_name="g")

    with run(storage_2):
        f_2(23)

    storage_1.sync_with_remote()


def test_remote_lots_of_stuff():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
    # create a *single* (mock) remote database
    client = mongomock.MongoClient()
    root = MongoMockRemoteStorage(db_name="test", client=client)

    # create multiple storages connected to it
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    def check_all_invariants():
        check_invariants(storage=storage_1)
        check_invariants(storage=storage_2)

    ### do stuff with storage 1
    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with run(storage_1):
        for i in range(20, 25):
            j = inc(x=i)
            final = add(i, j)

    ### do stuff with storage 2
    @op
    def mult(x: int, y: int) -> int:
        return x * y

    with run(storage_2):
        for i, j in zip(range(20, 25), range(20, 25)):
            k = mult(i, j)

    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert compare_data(storage_1=storage_1, storage_2=storage_2)

    ### now, rename a function in storage 1!
    rename_func(storage=storage_1, func=inc, new_name="inc_new")

    @op
    def inc_new(x: int) -> int:
        return x + 1

    synchronize(func=inc_new, storage=storage_1)

    ### rename argument too
    rename_arg(storage=storage_1, func=inc_new, name="x", new_name="x_new")

    @op
    def inc_new(x_new: int) -> int:
        return x_new + 1

    synchronize(func=inc_new, storage=storage_1)

    # do work with the renamed function in storage 1
    with run(storage_1):
        for i in range(20, 30):
            j = inc_new(x_new=i)
            final = add(i, j)

    # now sync stuff
    storage_2.sync_with_remote()

    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert compare_data(storage_1=storage_1, storage_2=storage_2)

    # check we can do work with the renamed function in storage_2
    with run(storage_2):
        for i in range(20, 40):
            j = inc_new(x_new=i)
            final = add(i, j)
    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert compare_data(storage_1=storage_1, storage_2=storage_2)

    # do some versioning stuff
    @op(version=1)
    def add(x: int, y: int, z: int) -> int:
        return x + y + z

    with run(storage_2):
        add(x=1, y=2, z=3)

    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert compare_data(storage_1=storage_1, storage_2=storage_2)
