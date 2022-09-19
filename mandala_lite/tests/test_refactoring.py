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

    with run(storage):
        x = inc(23)

    @op
    def inc(x: int, y=42) -> int:
        return x + y

    with run(storage):
        x = inc(23)

    assert inc.op.sig.input_names == {"x", "y"}
    with run(storage):
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

    with run(storage):
        add_many(0)
        add_many(0, 1)
        add_many(0, 1, 2)

    with query(storage) as q:
        x, y, z = Q(), Q(), Q()
        w = add_many(x, y, z)
        df = q.get_table(x, y, z, w)

    rows = set(tuple(row) for row in df.values.tolist())
    assert rows == {(0, 1, 2, 3), (0, 1, 42, 43), (0, 23, 42, 65)}

    ############################################################################
    ### check that queries work with defaults
    ############################################################################
    with query(storage) as q:
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


def test_func_renaming():

    storage = Storage()

    ############################################################################
    ### unit test
    ############################################################################
    @op
    def inc(x: int) -> int:
        return x + 1

    with run(storage):
        x = inc(23)

    rename_func(storage=storage, func=inc, new_name="inc_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc_new(x: int) -> int:
        return x + 1

    with run(storage):
        inc_new(23)

    with run(storage):
        df = inc_new.get_table()
    # make sure the call was not new
    assert df.shape == (1, 3)
    # make sure we did not create a new function
    assert len(storage.sig_adapter.sigs) == 1

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

    with run(storage):
        x = inc(23)

    rename_arg(storage=storage, func=inc, name="x", new_name="x_new")
    assert inc.is_invalidated

    # define correct function
    @op
    def inc(x_new: int) -> int:
        return x_new + 1

    with run(storage):
        inc(23)

    with run(storage):
        df = inc.get_table()
    # make sure the call was not new
    assert df.shape == (1, 3)
    # make sure we did not create a new function
    assert len(storage.sig_adapter.sigs) == 1

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

    with run(storage):
        inc(23)

    @op(version=1)
    def inc(x: int) -> int:
        return x + 1

    with run(storage):
        inc(23)

    assert len(storage.sig_adapter.sigs) == 2
    call_table_names = storage.rel_adapter.get_call_tables()
    assert len(call_table_names) == 2
    for table_name in call_table_names:
        df = storage.rel_storage.get_data(table=table_name)
        assert df.shape[0] == 1
