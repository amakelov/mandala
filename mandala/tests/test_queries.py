from mandala.all import *
from mandala.tests.utils import *
from mandala.core.weaver import visualize_computational_graph, traverse_all

OUTPUT_ROOT = Path(__file__).parent / "output/"


def test_empty():
    storage = Storage()

    with storage.query() as q:
        df = q.get_table()

    assert df.empty


def test_visualization():
    storage = Storage()

    @op
    def f(x: int, y: int) -> Tuple[int, int, int]:
        return x, y, x + y

    @op
    def g() -> int:
        return 42

    @op
    def h(z: int, w: int):
        pass

    for func in [f, g, h]:
        storage.synchronize(func)

    with storage.query():
        a = g().named("a")
        b = Q().named("b")
        c, d, e = f(x=a, y=b)
        c.named("c"), d.named("d"), e.named("e")
        h(z=c, w=d)
        h(z=a, w=b)
        x, y, z = f(x=d, y=e)
        x.named("x"), y.named("y"), z.named("z")
        storage.visualize_query(a, b, c, d, e)
        # vqs, fqs = traverse_all([a, b, c, d, e])
        # visualize_computational_graph(
        #     val_queries=vqs,
        #     func_queries=fqs,
        #     output_path=OUTPUT_ROOT / "test_visualization.svg",
        # )


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
            final = add(i, y=j)

    with storage.query() as q:
        vref = Q().named("vref")
        df = q.get_table(vref)
        assert set(df["vref"]) == set(range(20, 26)) | set(range(41, 50, 2))

    with storage.query() as q:
        i = Q().named("i")
        j = inc(i).named("j")
        final = add(i, y=j).named("final")
        df = q.get_table(i, j, final)
        assert set(df["i"]) == {i for i in range(20, 25)}
        assert all(df["j"] == df["i"] + 1)
        df_refs = q.get_table(i, j, final, values="refs")
        df_uids = q.get_table(i, j, final, values="uids")
        df_lazy = q.get_table(i, j, final, values="lazy")
        assert compare_dfs_as_relations(df_refs.applymap(lambda x: x.uid), df_uids)
        assert compare_dfs_as_relations(
            df_refs.applymap(lambda x: x.detached()), df_lazy
        )
        assert compare_dfs_as_relations(df_refs.applymap(unwrap), df)
        assert compare_dfs_as_relations(df_lazy.applymap(unwrap), df)

    check_invariants(storage)
    vqs, fqs = traverse_all([i, j, final])
    visualize_computational_graph(
        val_queries=vqs, func_queries=fqs, output_path=OUTPUT_ROOT / "test_basics.svg"
    )

    with storage.query() as q:
        i = Q().named("i")
        j = inc(i).named("j")
        final = add(i, j).named("final")
        df_naive = q.get_table(
            i, j, final, _engine="naive", _visualize_steps_at=OUTPUT_ROOT
        )
        assert compare_dfs_as_relations(df, df_naive)


def test_exceptions():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return unwrap(x) + 1

    try:
        # passing raw values in queries should (currently) raise an error
        with storage.query():
            y = inc(23)
        assert False
    except Exception:
        assert True


def test_filter_duplicates():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        for x in range(5):
            for y in range(5):
                z = inc(x)
                w = add(z, y)

    with storage.query() as q:
        x = Q().named("x")
        y = Q().named("y")
        z = inc(x).named("z")
        w = add(z, y).named("w")
        df_1 = q.get_table(x, z, _filter_duplicates=True)
        df_2 = q.get_table(x, z, _filter_duplicates=False)
        df_3 = q.get_table(
            x,
            z,
            _filter_duplicates=True,
            _engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )
        df_4 = q.get_table(
            x,
            z,
            _filter_duplicates=False,
            _engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )

    assert len(df_1) == 5
    assert len(df_2) == 25
    assert compare_dfs_as_relations(df_1, df_3)
    assert compare_dfs_as_relations(df_2, df_4)


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

    with storage.query() as q:
        x, y = Q().named("x"), Q().named("y")
        n = add(x, y).named("n")
        z = Q().named("z")
        a = inc_n_times(x=z, n=n).named("a")
        df_naive = q.get_table(
            x, y, n, a, z, _engine="naive", _visualize_steps_at=OUTPUT_ROOT
        )
        assert compare_dfs_as_relations(df, df_naive)


def test_superops_multilevel():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False

    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)

    @op
    def add_many(xs: Any, ys: Any) -> Any:
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
    with storage.query() as q:
        xs, ys = Q().named("xs"), Q().named("ys")
        result = add_many(xs=xs, ys=ys).named("result")
        df_naive = q.get_table(
            xs, ys, result, _engine="naive", _visualize_steps_at=OUTPUT_ROOT
        )
        assert compare_dfs_as_relations(df, df_naive)

    # two levels of nesting
    @op
    def add_many_many(xs: Any, ys: Any, zs: Any) -> Any:
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
    with storage.query() as q:
        xs, ys, zs = Q().named("xs"), Q().named("ys"), Q().named("zs")
        intermediate = add_many(xs, ys).named("intermediate")
        final = add_many(intermediate, zs).named("final")
        df_naive = q.get_table(
            xs,
            ys,
            intermediate,
            final,
            _engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )
        assert compare_dfs_as_relations(df, df_naive)


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
        storage.synchronize(f)

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
        df_naive = q.get_table(
            var_0,
            var_1,
            var_2,
            var_3,
            var_4,
            var_5,
            var_6,
            _engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )
        assert compare_dfs_as_relations(df, df_naive)


def test_required_args():
    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        for x in [1, 2, 3]:
            for y in [4, 5, 6]:
                add(x, y)

    with storage.query() as q:
        x = Q().named("x")
        result = add(x=x).named("result")
        df = q.get_table(x, result)
        assert df.shape == (9, 2)
