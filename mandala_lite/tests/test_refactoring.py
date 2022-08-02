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
