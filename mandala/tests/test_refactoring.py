from mandala.all import *
from mandala.tests.utils import *


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
    def inc(x: int, y=1) -> int:
        return x + y

    with storage.run():
        x = inc(23)

    assert inc.func_op.sig.input_names == {"x", "y"}
    df = storage.get_table(inc)
    assert all(c in df.columns for c in ["x", "y"])
    assert df.shape[0] == 1

    ############################################################################
    ### check that defaults can be overridden
    ############################################################################
    @op
    def add_many(x: int) -> int:
        return x + 1

    storage.synchronize(f=add_many)

    @op
    def add_many(x: int, y: int = 23, z: int = 42) -> int:
        return x + y + z

    storage.synchronize(f=add_many)

    with storage.run():
        add_many(0)
        add_many(0, 1)
        add_many(0, 1, 2)

    with storage.query():
        x, y, z = Q(), Q(), Q()
        w = add_many(x, y, z)
        df = storage.df(x, y, z, w)

    rows = set(tuple(row) for row in df.values.tolist())
    assert rows == {(0, 1, 2, 3), (0, 1, 42, 43), (0, 23, 42, 65)}

    ############################################################################
    ### check that queries work with defaults
    ############################################################################
    with storage.query() as q:
        x = Q()
        w = add_many(x)
        df = storage.df(x, w)

    ############################################################################
    ### check that invalid ways to add an input are not allowed
    ############################################################################

    ### no default
    @op
    def no_default(x: int) -> int:
        return x + 1

    storage.synchronize(f=no_default)

    try:

        @op
        def no_default(x: int, y: int) -> int:
            return x + y

        storage.synchronize(f=no_default)
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

    storage.rename_func(func=inc, new_name="inc_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc_new(x: int) -> int:
        return x + 1

    with storage.run():
        inc_new(23)

    df = storage.get_table(inc_new)
    # make sure the call was not new
    assert df.shape[0] == 1
    # make sure we did not create a new function
    sigs = [v for k, v in storage.sig_adapter.load_state().items() if not v.is_builtin]
    assert len(sigs) == 1

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

    storage.synchronize(f=inc)
    storage.synchronize(f=new_inc)

    try:
        storage.rename_func(func=inc, new_name="new_inc")
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

    storage.synchronize(f=inc)
    storage.synchronize(f=new_inc)

    storage.rename_func(func=inc, new_name="temp")
    storage.rename_func(func=new_inc, new_name="inc")

    @op
    def temp(x: int) -> int:
        return x + 1

    @op
    def inc(x: int) -> int:
        return x + 1

    storage.synchronize(f=temp)
    storage.synchronize(f=inc)

    storage.rename_func(func=temp, new_name="new_inc")


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

    storage.rename_arg(func=inc, name="x", new_name="x_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc(x_new: int) -> int:
        return x_new + 1

    with storage.run():
        inc(23)

    df = storage.get_table(inc)
    # make sure the call was not new
    assert df.shape[0] == 1
    # make sure we did not create a new function
    sigs = [v for k, v in storage.sig_adapter.load_state().items() if not v.is_builtin]
    assert len(sigs) == 1

    ############################################################################
    ### check collisions are not allowed
    ############################################################################
    @op
    def add(x: int, y: int) -> int:
        return x + y

    storage.synchronize(f=add)
    try:
        storage.rename_arg(func=add, name="x", new_name="y")
    except:
        assert True


def test_renaming_failures_1():
    """
    Try to do a rename on a function that was invalidated
    """
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    storage.synchronize(f=inc)

    storage.rename_func(func=inc, new_name="inc_new")
    try:
        storage.rename_func(func=inc, new_name="inc_other")
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    storage.synchronize(f=add)

    storage.rename_arg(func=add, name="x", new_name="z")
    try:
        storage.rename_arg(func=add, name="y", new_name="w")
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
        storage.synchronize(f=f)

    try:
        storage.rename_func(func=inc, new_name="add")
    except:
        assert True


def test_renaming_inside_context_1():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    storage.synchronize(f=inc)

    try:
        with storage.run():
            storage.rename_func(func=inc, new_name="inc_new")
            inc(23)
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    storage.synchronize(f=add)

    try:
        with storage.run():
            storage.rename_arg(func=add, name="x", new_name="z")
            add(23, 42)
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

    storage.synchronize(f=inc)

    try:
        with storage.run():
            inc(23)
            storage.rename_func(func=inc, new_name="inc_new")
    except:
        assert True

    @op
    def add(x: int, y: int) -> int:
        return x + y

    storage.synchronize(f=add)

    try:
        with storage.run():
            add(23, 42)
            storage.rename_arg(func=add, name="x", new_name="z")
    except:
        assert True


def test_other_refactoring_failures():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    storage.synchronize(f=inc)

    @op
    def inc(y: int) -> int:
        return y + 1

    try:
        storage.synchronize(f=inc)
    except:
        assert True
