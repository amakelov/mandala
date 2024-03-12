from mandala.all import *
from mandala.tests.utils import *
from mandala.queries.weaver import *


def test_construction():

    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    @op
    def f(x: int, a: int = 1) -> Tuple[int, int]:
        return x + a, x - a

    with storage.run():
        things = [add(i, i + 1) for i in range(10)]
        other_things = [f(i) for i in range(10)]

    rf1 = ComputationFrame.from_refs(things, storage=storage)
    rf2 = ComputationFrame.from_refs([t[0] for t in other_things], storage=storage)
    rf3 = ComputationFrame.from_op(func=add, storage=storage)
    rf4 = ComputationFrame.from_op(func=f, storage=storage)


def test_back():
    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    @op
    def f(x: int, a: int = 1) -> Tuple[int, int]:
        return x + a, x - a

    @op
    def mul(x: int, y: int) -> int:
        return x * y

    with storage.run():
        cs = []
        for i in range(10):
            a = add(i, i + 1)
            b = f(a)
            c = mul(b[0], b[1])
            cs.append(c)

    rf1 = ComputationFrame.from_refs(cs, storage=storage, name="c")
    rf1.back("c")
    rf1.back("c", inplace=True)
    rf1.back()


def test_evals():
    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    @op
    def f(x: int, a: int = 1) -> Tuple[int, int]:
        return x + a, x - a

    @op
    def mul(x: int, y: int) -> int:
        return x * y

    with storage.run():
        cs = []
        for i in range(10):
            a = add(i, i + 1)
            b = f(a)
            c = mul(b[0], b[1])
            cs.append(c)

    rf = ComputationFrame.from_refs(cs, storage=storage, name="c")

    rf[["c"]]
    rf[list(rf.var_nodes.keys())]
    df = rf.eval("c")
    assert len(df) == 10
    df = rf.eval()
    assert len(df) == 10

    rf = ComputationFrame.from_op(func=add, storage=storage)
    sub_rf = rf[rf.eval("x") < 5]
    assert len(sub_rf) == 5


def test_deletion():

    storage = Storage()

    @op
    def inc(x: int) -> int:
        print("hey!")
        return x + 1

    with storage.run():
        for i in range(10):
            inc(i)

    # check the number of rows goes down by the expected amount
    rf = ComputationFrame.from_op(func=inc, storage=storage)
    rf[rf.eval("x") < 5].delete(delete_dependents=True, ask=False)
    rf = ComputationFrame.from_op(func=inc, storage=storage)
    assert len(rf) == 5

    # re-compute the calls
    storage.cache.evict_all()
    with storage.run():
        for i in range(10):
            inc(i)
