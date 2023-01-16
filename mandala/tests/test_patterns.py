from mandala.all import *
from mandala.tests.utils import *


def test_context_nesting():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    @op
    def mul(x: int, y: int) -> int:
        return x * y

    with storage.run():
        for i in range(10):
            j = inc(i)
            k = add(i, j)
            l = mul(i, k)

    ### run -> query -> run composition
    with storage.run():
        i = 7
        with storage.query() as q:
            j = inc(i)
            k = add(i, j)
            df = q.get_table(i, j, k, values="objs")
            assert len(df) == 1
            with storage.run():
                for i, j, k in df.itertuples(index=False):
                    l = mul(i, k)
    assert unwrap(l) == 7 * (7 + 8)
