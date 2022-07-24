from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_basics():
    storage = Storage()

    @op(storage=storage)
    def inc(x: int) -> int:
        return x + 1

    @op(storage=storage)
    def add(x: int, y: int) -> int:
        return x + y

    with run(storage):
        for i in range(20, 25):
            j = inc(i)
            final = add(i, j)

    with query(storage) as q:
        i = Q().named("i")
        j = inc(i).named("j")
        final = add(i, j)
        df = q.get_table(i, j, final)
        assert set(df["i"]) == {i for i in range(20, 25)}
        assert all(df["j"] == df["i"] + 1)
    check_invariants(storage)
