from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_unit():
    storage = Storage()

    ############################################################################
    ### unit test
    ############################################################################

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        inc(23)

    @op(version_or_func=1)
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


def test_autoversion():
    """
    Idiomatic use of the library assumes versions keep moving forward and you
    don't go back to compute with older versions. This tests the
    `autoversion=True` setting which implicitly synchronizes with the latest
    version of the function.
    """
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    storage.synchronize(inc)

    @op(version_or_func=1)
    def inc(x: int) -> int:
        return x + 2

    storage.synchronize(inc)

    @op(autoversion=True)
    def inc(x: int) -> int:
        return x + 2

    storage.synchronize(inc)
    assert inc.func_op.sig.version == 1
