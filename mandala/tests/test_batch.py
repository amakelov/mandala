from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_unit():

    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.batch():
        y = inc(23)

    assert unwrap(y) == 24
    assert y.uid is not None
    all_data = storage.rel_storage.get_all_data()
    assert all_data[Config.vref_table].shape[0] == 2
    assert all_data[inc.func_op.sig.versioned_ui_name].shape[0] == 1
