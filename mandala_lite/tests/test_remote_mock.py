import mongomock

from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_disjoint_funcs():
    """
    Test a basic scenario where two users define two different functions and do
    some work with each, then sync their results.

    Expected behavior: the two storages end up in the same state.
    """
    client = mongomock.MongoClient()
    root = MongoMockRemoteStorage(db_name="test", client=client)

    # create multiple storages connected to it
    storage_1 = Storage(root=root)
    storage_2 = Storage(root=root)

    ### do work with storage 1
    @op
    def inc(x: int) -> int:
        return x + 1

    with storage_1.run():
        inc(23)

    ### do work with storage 2
    @op
    def mult(x: int, y: int) -> int:
        return x * y

    with storage_2.run():
        mult(23, 42)

    # synchronize storage_1 with the new work
    storage_1.sync_with_remote()

    # verify that both storages have the same state
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)


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

    storage_1.synchronize(f=f_1)
    storage_2.synchronize(f=f_2)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)


def test_add_input():
    """
    Unit test for adding an input to a function in a multi-user setting.

    Specifically, test a scenario where:
    - users 1 and 2 agree on the definition of a function `f`
    - user 1 adds an input to the function
    - user 2 tries to send calls to the initial function to the server

    Expected behavior:
    - calls to the old variant of the function work as expected:
        - user 2 is able to commit their calls and send them to the server
        - user 1 is able to then load these new calls into their local storage
        - the two storages end up in the same state
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

    storage_1.synchronize(f=inc_1)
    storage_2.synchronize(f=inc_2)

    @op(ui_name="inc")
    def inc_1(x: int, how_many_times: int = 1) -> int:
        return x + how_many_times

    storage_1.synchronize(f=inc_1)
    assert not signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    with storage_2.run():
        inc_2(23)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert not data_is_equal(storage_1=storage_1, storage_2=storage_2)
    storage_1.sync_from_remote()
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)


def test_rename_func():
    """
    Unit test for renaming a function in a multi-user setting.

    Specifically, test a scenario where:
    - users 1 and 2 agree on the definition of a function `f`
    - the user 1 renames the function to `g`
    - unbeknownst to that, the user 2 still sees `f` and uses it in
    computations.

    Expected behavior:
    - calls to the old variant of the function work as expected:
        - user 2 is able to commit their calls and send them to the server
        - user 1 is able to then load these new calls into their local storage
        - the two storages end up in the same state

    ! Possible confusion:
        - if user 2 then re-synchronizes the old variant of the function called
          `f`, this will create a new function.
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

    storage_1.synchronize(f=f_1)
    storage_2.synchronize(f=f_2)

    storage_1.rename_func(func=f_1, new_name="g")

    assert not signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    with storage_2.run():
        f_2(23)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert not data_is_equal(storage_1=storage_1, storage_2=storage_2)
    storage_1.sync_from_remote()
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)


def test_rename_input():
    """
    Analogous to `test_rename_func`.
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

    storage_1.synchronize(f=f_1)
    storage_2.synchronize(f=f_2)

    storage_1.rename_arg(func=f_1, name="x", new_name="y")

    assert not signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    with storage_2.run():
        f_2(23)
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert not data_is_equal(storage_1=storage_1, storage_2=storage_2)
    storage_1.sync_from_remote()
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)


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

    with storage_1.run():
        for i in range(20, 25):
            j = inc(x=i)
            final = add(i, j)

    ### do stuff with storage 2
    @op
    def mult(x: int, y: int) -> int:
        return x * y

    with storage_2.run():
        for i, j in zip(range(20, 25), range(20, 25)):
            k = mult(i, j)

    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    storage_2.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)

    ### now, rename a function in storage 1!
    storage_1.rename_func(func=inc, new_name="inc_new")

    @op
    def inc_new(x: int) -> int:
        return x + 1

    storage_1.synchronize(f=inc_new)

    ### rename argument too
    storage_1.rename_arg(func=inc_new, name="x", new_name="x_new")

    @op
    def inc_new(x_new: int) -> int:
        return x_new + 1

    storage_1.synchronize(f=inc_new)

    # do work with the renamed function in storage 1
    with storage_1.run():
        for i in range(20, 30):
            j = inc_new(x_new=i)
            final = add(i, j)

    # now sync stuff
    storage_2.sync_with_remote()

    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)

    # check we can do work with the renamed function in storage_2
    with storage_2.run():
        for i in range(20, 40):
            j = inc_new(x_new=i)
            final = add(i, j)
    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)

    # do some versioning stuff
    @op(version_or_func=1)
    def add(x: int, y: int, z: int) -> int:
        return x + y + z

    with storage_2.run():
        add(x=1, y=2, z=3)

    storage_2.sync_with_remote()
    storage_1.sync_with_remote()
    assert signatures_are_equal(storage_1=storage_1, storage_2=storage_2)
    assert data_is_equal(storage_1=storage_1, storage_2=storage_2)
