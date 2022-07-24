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


def test_superops_basic():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False

    storage = Storage()
    
    @op(storage=storage)
    def inc(x: int) -> int:
        return unwrap(x) + 1

    @op(storage=storage)
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)

    @op(storage=storage)
    def inc_n_times(x: int, n: int) -> int:
        for i in range(unwrap(n)):
            x = inc(x)
        return x

    with run(storage=storage):
        n = add(x=wrap(23), y=wrap(42))
        a = inc_n_times(x=wrap(23), n=n)
    
    with query(storage) as q:
        x, y = Q().named("x"), Q().named("y")
        n = add(x, y).named("n")
        z = Q().named('z')
        a = inc_n_times(x=z, n=n).named("a")
        df = q.get_table(x, y, n, a, z)
        assert df.values.tolist() == [[23, 42, 65, 88, 23]]
    

def test_superops_multilevel():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False

    storage = Storage()

    @op(storage)
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)
    
    @op(storage)
    def add_many(xs:List[int], ys:List[int]) -> List[int]:
        result = []
        for x in unwrap(xs):
            for y in unwrap(ys):
                result.append(add(wrap(x), wrap(y)))
        return result
    
    with run(storage):
        result = add_many(xs=wrap([1, 2, 3]), ys=wrap([4, 5, 6]))
    check_invariants(storage=storage)

    # individual adds
    with query(storage) as q:
        x, y = Q().named("x"), Q().named("y")
        z = add(x, y).named('z')
        df = q.get_table(x, y, z)
    assert all(df['x'] + df['y'] == df['z'])
    assert set(zip(df['x'].values.tolist(), df['y'].values.tolist())) == set(itertools.product([1, 2, 3], [4, 5, 6]))
    
    # end-to-end
    with query(storage) as q:
        xs, ys = Q().named("xs"), Q().named("ys")
        result = add_many(xs=xs, ys=ys).named('result')
        df = q.get_table(xs, ys, result)
    assert df['xs'].item() == [1, 2, 3]
    assert df['ys'].item() == [4, 5, 6]
    assert [unwrap(x) for x in df['result'].item()] == [a + b for a in [1, 2, 3] for b in [4, 5, 6]]
    
    # two levels of nesting
    @op(storage)
    def add_many_many(xs:List[int], ys:List[int], zs:List[int]) -> List[int]:
        intermediate = add_many(xs, ys)
        final = add_many(intermediate, zs)
        return final
    
    with run(storage):
        a = wrap([1, 2, 3])
        b = wrap([4, 5, 6])
        c = wrap([7, 8, 9])
        d = add_many_many(xs=a, ys=b, zs=c)
    
    with query(storage) as q:
        xs, ys, zs = Q().named("xs"), Q().named("ys"), Q().named("zs")
        intermediate = add_many(xs, ys).named('intermediate')
        final = add_many(intermediate, zs).named('final')
        df = q.get_table(xs, ys, intermediate, final)
        assert len(df['intermediate'].item()) == 9
        assert len(df['final'].item()) == 27