from mandala.all import *
from mandala.tests.utils import *
from mandala.queries.viz import visualize_graph
from mandala.queries.weaver import *
from mandala.queries.graphs import *
from mandala.queries.viz import *

OUTPUT_ROOT = Path(__file__).parent / "output/"


def get_graph(vqs: Set[ValQuery]) -> InducedSubgraph:
    vqs, fqs = traverse_all(vqs=vqs, direction="both")
    return InducedSubgraph(vqs=vqs, fqs=fqs)


def test_empty():
    storage = Storage()

    with storage.query():
        try:
            df = storage.df()
        except:
            pass


def test_queries_static_builder():
    """
    A one-stop test for all of the following cases:
        - queries to match data structures incl. nested structures
        - queries to match elements of data structures
        - wrapping raw objects into queries
        - merging the query graph from a run block and a query block

    """
    storage = Storage()

    @op
    def f(x) -> int:
        return x + 1

    @op
    def g(x, y) -> Tuple[int, int]:
        return x + y, x * y

    @op
    def avg(numbers: list) -> float:
        return sum(numbers) / len(numbers)

    @op
    def avg_dict(numbers: dict) -> float:
        return sum(numbers.values()) / len(numbers)

    @op
    def avg_set(numbers: set) -> float:
        return sum(numbers) / len(numbers)

    @op
    def repeat(x, n) -> list:
        return [x] * n

    @op
    def dictify(x, y) -> dict:
        return {"x": x, "y": y}

    @op
    def nested_dictify(x, y) -> Dict[str, List[int]]:
        return {"x": [x, x], "y": [y, y]}

    def get_graph_1():
        with storage.query():
            x = Q()
            y = f(x)
            z, w = g(y, x)
            ### constructive stuff
            lst_1 = qwrap([z, w, {Q(): x, ...: ...}, {x, y, ...}, ...])
            a_1 = avg(lst_1)
            dct_1 = qwrap({Q(): z, ...: ...})
            a_2 = avg_dict(dct_1)
            st_1 = qwrap({x, y, ...})
            a_3 = avg_set(st_1)
            ### destructive stuff
            lst_2 = repeat(x, Q())
            elt_0 = lst_2[Q()]
            dct_2 = dictify(x, y)
            val_0 = dct_2[Q()]
            dct_3 = nested_dictify(x, y)
            val_1 = dct_3[Q()]
            elt_1 = val_1[Q()]
        res = get_graph(vqs={elt_1, val_0, elt_0, a_3, a_2, a_1})
        return res

    def get_graph_2():
        with storage.query():
            x = Q()
            y = f(x)
            z, w = g(y, x)
            helper_1 = BuiltinQueries.DictQ(dct={Q(): x})
            helper_2 = BuiltinQueries.SetQ(elts={x, y})
            lst_1 = BuiltinQueries.ListQ(elts=[z, w, helper_1, helper_2])
            a_1 = avg(lst_1)
            dct_1 = BuiltinQueries.DictQ(dct={Q(): z})
            a_2 = avg_dict(dct_1)
            st_1 = BuiltinQueries.SetQ(elts={x, y})
            a_3 = avg_set(st_1)
            lst_2 = repeat(x, Q())
            elt_0 = lst_2[Q()]
            dct_2 = dictify(x, y)
            val_0 = dct_2[Q()]
            dct_3 = nested_dictify(x, y)
            val_1 = dct_3[Q()]
            elt_1 = val_1[Q()]
        res = get_graph(vqs={elt_1, val_0, elt_0, a_3, a_2, a_1})
        return res

    g_1, g_2 = get_graph_1(), get_graph_2()
    assert InducedSubgraph.are_canonically_isomorphic(g_1, g_2)


def test_queries_visualization():
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
        storage.draw(a, b, c, d, e, traverse="both", show_how="none")
        storage.print(a, b, c, d, e, traverse="both")


@pytest.mark.parametrize("storage", generate_storages())
def test_queries_basics(storage):
    Config.query_engine = "_test"

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

    with storage.query():
        i = Q().named("i")
        j = inc(i).named("j")
        final = add(i, y=j).named("final")
        df = storage.df(i, j, final)
        assert set(df["i"]) == {i for i in range(20, 25)}
        assert all(df["j"] == df["i"] + 1)
        df_refs = storage.df(i, j, final, values="refs")
        df_uids = storage.df(i, j, final, values="uids")
        df_lazy = storage.df(i, j, final, values="lazy")
        assert compare_dfs_as_relations(df_refs.applymap(lambda x: x.uid), df_uids)
        A = df_refs.applymap(lambda x: x.detached())
        B = df_lazy
        print(A)
        print(B)
        assert compare_dfs_as_relations(
            df_refs.applymap(lambda x: x.detached()), df_lazy
        )
        assert compare_dfs_as_relations(df_refs.applymap(unwrap), df)
        assert compare_dfs_as_relations(df_lazy.applymap(unwrap), df)

    check_invariants(storage)
    vqs, fqs = traverse_all([i, j, final])
    visualize_graph(
        vqs=vqs, fqs=fqs, output_path=OUTPUT_ROOT / "test_basics.svg", names=None
    )


def test_queries_exceptions():
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


def test_queries_filter_duplicates():
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

    with storage.query():
        x = Q().named("x")
        y = Q().named("y")
        z = inc(x).named("z")
        w = add(z, y).named("w")
        df_1 = storage.df(x, z, drop_duplicates=True, include_context=False)
        df_2 = storage.df(x, z, drop_duplicates=False, include_context=False)
        df_3 = storage.df(
            x,
            z,
            drop_duplicates=True,
            include_context=False,
            engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )
        df_4 = storage.df(
            x,
            z,
            drop_duplicates=False,
            include_context=False,
            engine="naive",
            _visualize_steps_at=OUTPUT_ROOT,
        )

    assert len(df_1) == 5
    assert len(df_2) == 25
    assert compare_dfs_as_relations(df_1, df_3)
    assert compare_dfs_as_relations(df_2, df_4)


def test_queries_superops_multilevel():
    Config.autowrap_inputs = False
    Config.autounwrap_inputs = False
    Config.query_engine = "_test"

    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return unwrap(x) + unwrap(y)

    @op
    def add_many(xs: Any, ys: Any) -> Any:
        result = []
        for x in unwrap(xs):
            for y in unwrap(ys):
                result.append(add(wrap_atom(x), wrap_atom(y)))
        return result

    with storage.run():
        result = add_many(xs=wrap_atom([1, 2, 3]), ys=wrap_atom([4, 5, 6]))
    check_invariants(storage=storage)

    # individual adds
    with storage.query():
        x, y = Q().named("x"), Q().named("y")
        z = add(x, y).named("z")
        df = storage.df(x, y, z)
    assert all(df["x"] + df["y"] == df["z"])
    assert set(zip(df["x"].values.tolist(), df["y"].values.tolist())) == set(
        itertools.product([1, 2, 3], [4, 5, 6])
    )

    # end-to-end
    with storage.query():
        xs, ys = Q().named("xs"), Q().named("ys")
        result = add_many(xs=xs, ys=ys).named("result")
        df = storage.df(xs, ys, result)
    assert df["xs"].item() == [1, 2, 3]
    assert df["ys"].item() == [4, 5, 6]
    assert [unwrap(x) for x in df["result"].item()] == [
        a + b for a in [1, 2, 3] for b in [4, 5, 6]
    ]

    # two levels of nesting
    @op
    def add_many_many(xs: Any, ys: Any, zs: Any) -> Any:
        intermediate = add_many(xs, ys)
        final = add_many(intermediate, zs)
        return final

    with storage.run():
        a = wrap_atom([1, 2, 3])
        b = wrap_atom([4, 5, 6])
        c = wrap_atom([7, 8, 9])
        d = add_many_many(xs=a, ys=b, zs=c)

    with storage.query():
        xs, ys, zs = Q().named("xs"), Q().named("ys"), Q().named("zs")
        intermediate = add_many(xs, ys).named("intermediate")
        final = add_many(intermediate, zs).named("final")
        df = storage.df(xs, ys, intermediate, final)
        assert len(df["intermediate"].item()) == 9
        assert len(df["final"].item()) == 27


def test_queries_weird():
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
    Config.query_engine = "_test"

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

    with storage.query():
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
        df = storage.df(var_0, var_1, var_2, var_3, var_4, var_5, var_6)


def test_queries_required_args():
    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        for x in [1, 2, 3]:
            for y in [4, 5, 6]:
                add(x, y)

    with storage.query():
        x = Q().named("x")
        result = add(x=x).named("result")
        df = storage.df(x, result)
        assert df.shape == (9, 2)
