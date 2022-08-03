from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_add_input():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True

    storage = Storage()

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


def test_func_renaming():

    storage = Storage()

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
    assert len(storage.rel_adapter.load_signatures()) == 1


def test_arg_renaming():
    storage = Storage()

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
    assert len(storage.rel_adapter.load_signatures()) == 1


def test_versions():
    storage = Storage()

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

    assert len(storage.rel_adapter.load_signatures()) == 2
    call_table_names = storage.rel_adapter.get_call_tables()
    assert len(call_table_names) == 2
    for table_name in call_table_names:
        df = storage.rel_storage.get_data(table=table_name)
        assert df.shape[0] == 1
