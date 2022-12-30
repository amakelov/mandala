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

    @op(version=1)
    def inc(x: int) -> int:
        return x + 1

    # check unsynchronized functions work for this
    storage.sig_adapter.get_versions(sig=inc.func_op.sig)

    with storage.run():
        inc(23)

    sigs = [v for k, v in storage.sig_adapter.load_state().items() if not v.is_builtin]
    assert len(sigs) == 2
