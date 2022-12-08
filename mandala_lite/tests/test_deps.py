from mandala_lite.all import *
from mandala_lite.tests.utils import *
from .sample_codebase.b import *
from .sample_codebase.nested.c import *

###! setup at top level to simulate a user's codebase
CONST_1 = 23
CONST_2 = True
MODULE_PATH = os.path.dirname(__file__)
STORAGE_PATH = Path(os.path.join(MODULE_PATH, "output/test_deps.db"))
storage = Storage(deps_root=Path(MODULE_PATH), db_path=STORAGE_PATH)


def cleanup():
    if STORAGE_PATH.exists():
        os.remove(STORAGE_PATH)


def helper_1(x):
    a = 23
    return x + CONST_1


def helper_2(x):
    return helper_1(x) + 1


with storage.define():

    @op
    def inc(x: int) -> int:
        return helper_1(x) + CONST_2

    @op
    def add(x: int, y: int) -> int:
        return helper_2(x)

    @op
    def mul(x: int, y: int) -> int:
        return x * y


def test_unit():

    with storage.run():
        x = 23
        y = inc(x)
        z = add(x, y)
        w = mul(x, y)

    deps = storage.sig_adapter.deps_adapter.load_state()
    module_address = "mandala_lite.tests.test_deps"
    inc_sig: Signature = inc.func_op.sig
    inc_deps = deps[inc_sig.internal_name, inc_sig.version]
    assert set(inc_deps.globals_.keys()) == {module_address}
    assert set(inc_deps.globals_[module_address].keys()) == {"CONST_1", "CONST_2"}

    cleanup()
