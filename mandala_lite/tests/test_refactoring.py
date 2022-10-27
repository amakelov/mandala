from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_add_input():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True

    storage = Storage()

    ############################################################################
    ### check that old calls are preserved
    ############################################################################
    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        x = inc(23)

    @op
    def inc(x: int, y=42) -> int:
        return x + y

    with storage.run():
        x = inc(23)

    assert inc.func_op.sig.input_names == {"x", "y"}
    with storage.run():
        df = inc.get_table()
    assert df.shape == (1, 4)

    ############################################################################
    ### check that defaults can be overridden
    ############################################################################
    @op
    def add_many(x: int) -> int:
        return x + 1

    synchronize(func=add_many, storage=storage)

    @op
    def add_many(x: int, y: int = 23, z: int = 42) -> int:
        return x + y + z

    synchronize(func=add_many, storage=storage)

    with storage.run():
        add_many(0)
        add_many(0, 1)
        add_many(0, 1, 2)

    with storage.query() as q:
        x, y, z = Q(), Q(), Q()
        w = add_many(x, y, z)
        df = q.get_table(x, y, z, w)

    rows = set(tuple(row) for row in df.values.tolist())
    assert rows == {(0, 1, 2, 3), (0, 1, 42, 43), (0, 23, 42, 65)}

    ############################################################################
    ### check that queries work with defaults
    ############################################################################
    with storage.query() as q:
        x = Q()
        w = add_many(x)
        df = q.get_table(x, w)

    ############################################################################
    ### check that invalid ways to add an input are not allowed
    ############################################################################

    ### no default
    @op
    def no_default(x: int) -> int:
        return x + 1

    synchronize(func=no_default, storage=storage)

    try:

        @op
        def no_default(x: int, y: int) -> int:
            return x + y

        synchronize(func=no_default, storage=storage)
        assert False
    except:
        assert True


def test_add_input_bug():
    """
    The issue was that the sync logic was expecting to see the UIDs for the
    defaults upon re-synchronizing the updated version.
    """
    storage = Storage()

    @op
    def f() -> int:
        return 1

    with storage.run():
        f()

    @op
    def f(x: int = 23) -> int:
        return x

    with storage.run():
        f()

    @op
    def f(x: int = 23) -> int:
        return x

    with storage.run():
        f()


def test_default_change():
    """
    Changing default values is not allowed for @ops
    """
    storage = Storage()

    @op
    def f(x: int = 23) -> int:
        return x

    with storage.run():
        a = f()

    @op
    def f(x: int = 42) -> int:
        return x

    try:
        with storage.run():
            b = f()
        assert False
    except:
        assert True


def test_func_renaming():

    storage = Storage()

    ############################################################################
    ### unit test
    ############################################################################
    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        x = inc(23)

    rename_func(storage=storage, func=inc, new_name="inc_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc_new(x: int) -> int:
        return x + 1

    with storage.run():
        inc_new(23)

    with storage.run():
        df = inc_new.get_table()
    # make sure the call was not new
    assert df.shape == (1, 3)
    # make sure we did not create a new function
    assert len(storage.sig_adapter.load_state()) == 1

    ############################################################################
    ### check that name collisions are not allowed
    ############################################################################
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def new_inc(x: int) -> int:
        return x + 1

    synchronize(func=inc, storage=storage)
    synchronize(func=new_inc, storage=storage)

    try:
        rename_func(storage=storage, func=inc, new_name="inc_new")
        assert False
    except:
        assert True

    ############################################################################
    ### permute names
    ############################################################################
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def new_inc(x: int) -> int:
        return x + 1

    synchronize(func=inc, storage=storage)
    synchronize(func=new_inc, storage=storage)

    rename_func(storage=storage, func=inc, new_name="temp")
    rename_func(storage=storage, func=new_inc, new_name="inc")

    @op
    def temp(x: int) -> int:
        return x + 1

    @op
    def inc(x: int) -> int:
        return x + 1

    synchronize(func=temp, storage=storage)
    synchronize(func=inc, storage=storage)

    rename_func(storage=storage, func=temp, new_name="new_inc")


def test_arg_renaming():
    storage = Storage()

    ############################################################################
    ### unit test
    ############################################################################
    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        x = inc(23)

    rename_arg(storage=storage, func=inc, name="x", new_name="x_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc(x_new: int) -> int:
        return x_new + 1

    with storage.run():
        inc(23)

    with storage.run():
        df = inc.get_table()
    # make sure the call was not new
    assert df.shape == (1, 3)
    # make sure we did not create a new function
    assert len(storage.sig_adapter.load_state()) == 1

    ############################################################################
    ### check collisions are not allowed
    ############################################################################
    @op
    def add(x: int, y: int) -> int:
        return x + y

    synchronize(func=add, storage=storage)
    try:
        rename_arg(storage=storage, func=add, name="x", new_name="y")
        assert False
    except:
        assert True


def test_versions():
    storage = Storage()

    ############################################################################
    ### unit test
    ############################################################################

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        inc(23)

    @op(version=1)
    def inc(x: int) -> int:
        return x + 1

    # check unsynchronized functions work for this
    storage.sig_adapter.get_versions(sig=inc.func_op.sig)

    with storage.run():
        inc(23)

    assert len(storage.sig_adapter.load_state()) == 2
    call_table_names = storage.rel_adapter.get_call_tables()
    assert len(call_table_names) == 2
    for table_name in call_table_names:
        df = storage.rel_storage.get_data(table=table_name)
        assert df.shape[0] == 1


def test_renaming_failures_1():
    """
    Try to do a rename on a function that was invalidated
    """
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    synchronize(func=inc, storage=storage)

    rename_func(storage=storage, func=inc, new_name="inc_new")
    try:
        rename_func(storage=storage, func=inc, new_name="inc_other")
        assert False
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    synchronize(func=add, storage=storage)

    rename_arg(storage=storage, func=add, name="x", new_name="z")
    try:
        rename_arg(storage=storage, func=add, name="y", new_name="w")
        assert False
    except:
        assert True


def test_renaming_failures_2():
    """
    Try renaming a function to a name that already exists for another function
    """
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    for f in (inc, add):
        synchronize(func=f, storage=storage)

    try:
        rename_func(storage=storage, func=inc, new_name="add")
        assert False
    except:
        assert True


def test_renaming_inside_context_1():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    synchronize(inc, storage=storage)

    try:
        with storage.run():
            rename_func(storage=storage, func=inc, new_name="inc_new")
            inc(23)
        assert False
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    synchronize(add, storage=storage)

    try:
        with storage.run():
            rename_arg(storage=storage, func=add, name="x", new_name="z")
            add(23, 42)
        assert False
    except:
        assert True


def test_renaming_inside_context_2():
    """
    Like the previous one, but with uncommitted work
    """
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    synchronize(inc, storage=storage)

    try:
        with storage.run():
            inc(23)
            rename_func(storage=storage, func=inc, new_name="inc_new")
        assert False
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    synchronize(add, storage=storage)

    try:
        with storage.run():
            add(23, 42)
            rename_arg(storage=storage, func=add, name="x", new_name="z")
        assert False
    except:
        assert True


def test_other_refactoring_failures():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    synchronize(inc, storage=storage)

    @op
    def inc(y: int) -> int:
        return y + 1

    try:
        synchronize(inc, storage)
        assert False
    except:
        assert True
