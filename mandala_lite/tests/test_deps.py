from mandala_lite.all import *
from mandala_lite.tests.utils import *
from .sample_codebase.b import *
from .sample_codebase.nested.c import *

CONST_1 = 23
CONST_2 = True
MODULE_PATH = os.path.dirname(__file__)


def helper_1(x):
    return x + CONST_1


def helper_2(x):
    return helper_1(x) + 1


@op
def inc(x: int) -> int:
    return helper_1(x) + CONST_2


@op
def add(x: int, y: int) -> int:
    return helper_2(x)


def test_unit():

    storage = Storage(deps_root=Path(MODULE_PATH))

    with storage.run():
        x = 23
        y = inc(x)
        z = add(x, y)

    deps = storage.sig_adapter.deps_adapter.load_state()
    module_address = "mandala_lite.tests.test_deps"
    inc_sig: Signature = inc.func_op.sig
    inc_deps = deps[inc_sig.internal_name, inc_sig.version]
    assert set(inc_deps.globals_.keys()) == {module_address}
    assert set(inc_deps.globals_[module_address].keys()) == {"CONST_1", "CONST_2"}
