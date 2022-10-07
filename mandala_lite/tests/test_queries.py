from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_basics():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        for i in range(20, 25):
            j = inc(i)
            final = add(i, j)

    with storage.query() as q:
        vref = Q().named("vref")
        df = q.get_table(vref)
        assert set(df["vref"]) == set(range(20, 26)) | set(range(41, 50, 2))

    with storage.query() as q:
        i = Q().named("i")
        j = inc(i).named("j")
        final = add(i, j)
        df = q.get_table(i, j, final)
        assert set(df["i"]) == {i for i in range(20, 25)}
        assert all(df["j"] == df["i"] + 1)
    check_invariants(storage)


def test_superops_basic():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False

    storage = Storage()

    @op
    def inc(x: int) -> int:
        return unwrap(x) + 1

    @op
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)

    @op
    def inc_n_times(x: int, n: int) -> int:
        for i in range(unwrap(n)):
            x = inc(x)
        return x

    with storage.run():
        n = add(x=wrap(23), y=wrap(42))
        a = inc_n_times(x=wrap(23), n=n)

    with storage.query() as q:
        x, y = Q().named("x"), Q().named("y")
        n = add(x, y).named("n")
        z = Q().named("z")
        a = inc_n_times(x=z, n=n).named("a")
        df = q.get_table(x, y, n, a, z)
        assert df.values.tolist() == [[23, 42, 65, 88, 23]]


def test_superops_multilevel():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False

    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)

    @op
    def add_many(xs: List[int], ys: List[int]) -> List[int]:
        result = []
        for x in unwrap(xs):
            for y in unwrap(ys):
                result.append(add(wrap(x), wrap(y)))
        return result

    with storage.run():
        result = add_many(xs=wrap([1, 2, 3]), ys=wrap([4, 5, 6]))
    check_invariants(storage=storage)

    # individual adds
    with storage.query() as q:
        x, y = Q().named("x"), Q().named("y")
        z = add(x, y).named("z")
        df = q.get_table(x, y, z)
    assert all(df["x"] + df["y"] == df["z"])
    assert set(zip(df["x"].values.tolist(), df["y"].values.tolist())) == set(
        itertools.product([1, 2, 3], [4, 5, 6])
    )

    # end-to-end
    with storage.query() as q:
        xs, ys = Q().named("xs"), Q().named("ys")
        result = add_many(xs=xs, ys=ys).named("result")
        df = q.get_table(xs, ys, result)
    assert df["xs"].item() == [1, 2, 3]
    assert df["ys"].item() == [4, 5, 6]
    assert [unwrap(x) for x in df["result"].item()] == [
        a + b for a in [1, 2, 3] for b in [4, 5, 6]
    ]

    # two levels of nesting
    @op
    def add_many_many(xs: List[int], ys: List[int], zs: List[int]) -> List[int]:
        intermediate = add_many(xs, ys)
        final = add_many(intermediate, zs)
        return final

    with storage.run():
        a = wrap([1, 2, 3])
        b = wrap([4, 5, 6])
        c = wrap([7, 8, 9])
        d = add_many_many(xs=a, ys=b, zs=c)

    with storage.query() as q:
        xs, ys, zs = Q().named("xs"), Q().named("ys"), Q().named("zs")
        intermediate = add_many(xs, ys).named("intermediate")
        final = add_many(intermediate, zs).named("final")
        df = q.get_table(xs, ys, intermediate, final)
        assert len(df["intermediate"].item()) == 9
        assert len(df["final"].item()) == 27


def test_weird():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True

    storage = Storage()

    @op
    def a(f: int, g: int):
        return

    @op
    def b(k: int, l: int):
        return

    @op
    def c(h: int, i: int) -> int:
        return h + i

    @op
    def d(j: int) -> Tuple[int, int, int]:
        return j, j, j

    @op
    def e(m: int) -> int:
        return m + 1

    for f in [a, b, c, d, e]:
        synchronize(f, storage)

    with storage.run():
        var_0 = 23
        var_1 = 42
        a(var_0, var_0)
        b(var_0, var_0)
        a(f=var_0, g=var_0)
        var_2 = c(h=var_0, i=var_0)
        a(f=var_0, g=var_0)
        var_3, var_4, var_5 = d(j=var_1)
        b(k=var_4, l=var_1)
        var_6 = e(m=var_2)

    with storage.query() as q:
        var_0 = Q()
        var_1 = Q()
        a(f=var_0, g=var_0)
        b(k=var_0, l=var_0)
        a(f=var_0, g=var_0)
        var_2 = c(h=var_0, i=var_0)
        a(f=var_0, g=var_0)
        var_3, var_4, var_5 = d(j=var_1)
        b(k=var_4, l=var_1)
        var_6 = e(m=var_2)
        df = q.get_table(var_0, var_1, var_2, var_3, var_4, var_5, var_6)
