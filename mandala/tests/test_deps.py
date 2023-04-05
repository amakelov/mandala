from mandala.all import *
from mandala.tests.utils import *
from mandala.deps.shallow_versions import DAG
import numpy as np


def test_dag():
    d = DAG(content_type="code")
    try:
        d.commit("something")
    except AssertionError:
        pass

    content_hash_1 = d.init(initial_content="something")
    assert len(d.commits) == 1
    assert d.head == content_hash_1
    content_hash_2 = d.commit(content="something else", is_semantic_change=True)
    assert (
        d.commits[content_hash_2].semantic_hash
        != d.commits[content_hash_1].semantic_hash
    )
    assert d.head == content_hash_2
    content_hash_3 = d.commit(content="something else #2", is_semantic_change=False)
    assert (
        d.commits[content_hash_3].semantic_hash
        == d.commits[content_hash_2].semantic_hash
    )

    content_hash_4 = d.sync(content="something else")
    assert content_hash_4 == content_hash_2
    assert d.head == content_hash_2

    d.show()


MODULE_NAME = "mandala.tests.test_deps"
DEPS_PACKAGE = "mandala.tests"
DEPS_PATH = Path(__file__).absolute().resolve()
# MODULE_NAME = '__main__'


def _test_version_reprs(storage: Storage):
    for dag in storage.get_versioner().component_dags.values():
        for compact in [True, False]:
            dag.show(compact=compact)
    for version in storage.get_versioner().get_flat_versions().values():
        storage.get_versioner().present_dependencies(commits=version.semantic_expansion)
    storage.get_versioner().global_topology.show(path=generate_path(ext=".png"))
    repr(storage.get_versioner().global_topology)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_unit(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )

    # to be able to import this name
    global f_1, A

    A = 42

    @op
    def f_1(x) -> int:
        return 23 + A

    with storage.run():
        f_1(1)

    vs = storage.get_versioner()
    f_1_versions = vs.versions[MODULE_NAME, "f_1"]
    assert len(f_1_versions) == 1
    version = f_1_versions[list(f_1_versions.keys())[0]]
    assert set(version.support) == {(MODULE_NAME, "f_1"), (MODULE_NAME, "A")}
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_libraries(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    if tracer_impl is SysTracer:
        track = lambda x: x
    else:
        from mandala.deps.tracers.dec_impl import track

    global f_2, f_1

    @track
    def f_1(x) -> int:
        return 23

    # use functions from libraries to make sure we don't trace them
    @op
    def f_2(x) -> int:
        df = pd.DataFrame({"a": [1, 2, 3]})
        array = np.array([1, 2, 3])
        x = array.mean() + np.random.uniform()
        return f_1(x)

    with storage.run():
        f_2(1)

    vs = storage.get_versioner()
    f_2_versions = vs.versions[MODULE_NAME, "f_2"]
    assert len(f_2_versions) == 1
    version = f_2_versions[list(f_2_versions.keys())[0]]
    assert set(version.support) == {(MODULE_NAME, "f_2"), (MODULE_NAME, "f_1")}
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_deps(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    global dep_1, f_2, A
    if tracer_impl is SysTracer:
        track = lambda x: x
    else:
        from mandala.deps.tracers.dec_impl import track

    A = 42

    @track
    def dep_1(x) -> int:
        return 23

    @track
    @op
    def f_2(x) -> int:
        return dep_1(x) + A

    with storage.run():
        f_2(1)

    vs = storage.get_versioner()
    f_2_versions = vs.versions[MODULE_NAME, "f_2"]
    assert len(f_2_versions) == 1
    version = f_2_versions[list(f_2_versions.keys())[0]]
    assert set(version.support) == {
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "dep_1"),
        (MODULE_NAME, "A"),
    }
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_changes(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )

    global f

    @op
    def f(x) -> int:
        return x + 1

    with storage.run():
        f(1)
    commit_1 = storage.sync_component(
        component=f,
        is_semantic_change=None,
    )

    @op
    def f(x) -> int:
        return x + 2

    commit_2 = storage.sync_component(component=f, is_semantic_change=True)
    assert commit_1 != commit_2
    with storage.run():
        f(1)

    @op
    def f(x) -> int:
        return x + 1

    # confirm we reverted to the previous version
    commit_3 = storage.sync_component(
        component=f,
        is_semantic_change=None,
    )
    assert commit_3 == commit_1
    with storage.run(allow_calls=False):
        f(1)

    # create a new branch
    @op
    def f(x) -> int:
        return x + 3

    commit_4 = storage.sync_component(
        component=f,
        is_semantic_change=True,
    )
    assert commit_4 not in (commit_1, commit_2)
    with storage.run():
        f(1)

    f_versions = storage.get_versioner().versions[MODULE_NAME, "f"]
    assert len(f_versions) == 3
    semantic_versions = [v.semantic_version for v in f_versions.values()]
    assert len(set(semantic_versions)) == 3
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_superops(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    Config.enable_ref_magics = True

    global f_1, f_2, f_3

    @op
    def f_1(x) -> int:
        return x + 1

    @superop
    def f_2(x) -> int:
        return f_1(x) + 1

    with storage.run():
        f_1(1)

    with storage.run(attach_call_to_outputs=True):
        a = f_2(1)
        call: Call = a._call
    version = storage.get_versioner().versions[MODULE_NAME, "f_2"][call.content_version]
    assert set(version.support) == {(MODULE_NAME, "f_1"), (MODULE_NAME, "f_2")}

    @superop
    def f_3(x) -> int:
        return f_2(x) + 1

    with storage.run(attach_call_to_outputs=True):
        a = f_3(1)
        call: Call = a._call
    version = storage.get_versioner().versions[MODULE_NAME, "f_3"][call.content_version]
    assert set(version.support) == {
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "f_3"),
    }
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_dependency_patterns(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    global A, B, f_1, f_2, f_3, f_4, f_5, f_6
    if tracer_impl is SysTracer:
        track = lambda x: x
    else:
        from mandala.deps.tracers.dec_impl import track

    # global vars
    A = 23
    B = [1, 2, 3]

    # using a global var
    @track
    def f_1(x) -> int:
        return x + A

    # calling another function
    @track
    def f_2(x) -> int:
        return f_1(x) + B[0]

    # different dependencies per call
    @op
    def f_3(x) -> int:
        if x % 2 == 0:
            return f_2(2 * x)
        else:
            return f_1(x + 1)

    with storage.run(attach_call_to_outputs=True):
        x = f_3(0)
        call: Call = x._call
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
        (MODULE_NAME, "f_1"),
    }
    with storage.run(attach_call_to_outputs=True):
        x = f_3(1)
        call: Call = x._call
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
    }

    # using a lambda
    @op
    def f_4(x) -> int:
        f = lambda y: f_1(y) + B[0]
        return f(x)

    # make sure the call in the lambda is detected
    with storage.run(attach_call_to_outputs=True):
        x = f_4(10)
        call: Call = x._call
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_4"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
    }

    # using comprehensions and generators
    @superop
    def f_5(x) -> int:
        x = unwrap(x)
        a = [f_1(y) for y in range(x)]
        b = {f_2(y) for y in range(x)}
        c = {y: f_3(y) for y in range(x)}
        return sum(unwrap(f_4(y)) for y in range(x))

    with storage.run():
        f_5(10)

    f_5_versions = storage.get_versioner().versions[MODULE_NAME, "f_5"]
    assert len(f_5_versions) == 1
    version = f_5_versions[list(f_5_versions.keys())[0]]
    assert set(version.support) == {
        (MODULE_NAME, "f_5"),
        (MODULE_NAME, "f_4"),
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
    }

    # nested comprehensions and generators
    @superop
    def f_6(x) -> int:
        x = unwrap(x)
        # nested list comprehension
        a = sum([sum([f_1(y) for y in range(x)]) for z in range(x)])
        # nested comprehension with generator
        b = sum(sum(f_2(y) for y in range(x)) for z in range(unwrap(f_3(x))))
        return a + b

    with storage.run():
        f_6(2)

    f_6_versions = storage.get_versioner().versions[MODULE_NAME, "f_6"]
    assert len(f_6_versions) == 1
    version = f_6_versions[list(f_6_versions.keys())[0]]
    assert set(version.support) == {
        (MODULE_NAME, "f_6"),
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
    }
    _test_version_reprs(storage=storage)
    storage.versions(f_6)
    storage.sources(f_6)
    storage.get_code(version_id=version.content_version)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_recursion(tracer_impl):
    ### mutual recursion
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    Config.enable_ref_magics = True
    global s_1, s_2

    @superop
    def s_1(x) -> int:
        if x == 0:
            return 0
        else:
            return s_2(x - 1) + 1

    @superop
    def s_2(x) -> int:
        if x == 0:
            return 0
        else:
            return s_1(x - 1) + 1

    with storage.run(attach_call_to_outputs=True):
        a = s_1(0)
        call_1 = a._call
        b = s_1(1)
        call_2 = b._call
    version_1 = storage.get_versioner().get_flat_versions()[call_1.content_version]
    assert version_1.support == {(MODULE_NAME, "s_1")}
    version_2 = storage.get_versioner().get_flat_versions()[call_2.content_version]
    assert version_2.support == {(MODULE_NAME, "s_1"), (MODULE_NAME, "s_2")}
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_queries_multiple_versions(tracer_impl):
    Config.query_engine = "sql"  # doesn't work yet with the naive engine
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )

    global f_1, f_2, f_3

    ### define an op with multiple semantically-compatible versions
    @op
    def f_1(x) -> int:
        return x + 2

    @op
    def f_2(x) -> int:
        return x + 3

    @op
    def f_3(x) -> int:
        if x % 2 == 0:
            return f_2(2 * x)
        else:
            return f_1(x + 1)

    with storage.run():
        for i in range(10):  # make sure both versions are used
            f_3(i)

    with storage.query():
        i = Q()
        j = f_3(i)
        df_1 = storage.df(i.named("i"), j.named("j"))
    assert df_1.shape == (10, 2)
    assert (df_1["j"] == df_1["i"].apply(f_3)).all()

    # change one of the dependencies semantically and check that the query
    # result is what remains from the other branch
    @op
    def f_1(x) -> int:
        return x + 4

    storage.sync_component(
        f_1,
        is_semantic_change=True,
    )

    with storage.query():
        i = Q()
        j = f_3(i)
        df_2 = storage.df(i.named("i"), j.named("j"))
    assert df_2.shape == (5, 2)
    assert sorted(df_2["i"].values.tolist()) == [0, 2, 4, 6, 8]
    assert (df_2["j"] == df_2["i"].apply(f_3)).all()

    ### go back to the old version and check that the query result is recovered
    @op
    def f_1(x) -> int:
        return x + 2

    storage.sync_component(
        f_1,
        is_semantic_change=None,
    )

    with storage.query():
        i = Q()
        j = f_3(i)
        df_3 = storage.df(i.named("i"), j.named("j"))
    assert (df_1 == df_3).all().all()


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_memoized_tracking(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )
    if tracer_impl is SysTracer:
        track = lambda x: x
    else:
        from mandala.deps.tracers.dec_impl import track

    global f_1, f_2, f, g

    @track
    def f_1(x):
        return x + 1

    @track
    def f_2(x):
        return x + 2

    @op
    def f(x) -> int:
        if x % 2 == 0:
            return f_1(x)
        else:
            return f_2(x)

    @superop
    def g(x) -> List[int]:
        return [f(i) for i in range(unwrap(x))]

    with storage.run():
        for i in range(10):
            f(i)

    with storage.run():
        g(10)


@pytest.mark.parametrize("tracer_impl", [SysTracer, DecTracer])
def test_transient(tracer_impl):
    global f

    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )

    @op
    def f(x) -> int:
        return Transient(x + 1)

    with storage.run():
        a = f(42)
    with storage.run(recompute_transient=True):
        a = f(42)


@pytest.mark.parametrize("tracer_impl", [DecTracer, SysTracer])
def test_queries_unit(tracer_impl):
    Config.query_engine = "sql"  # doesn't work yet with the naive engine

    storage = Storage(
        deps_path=DEPS_PATH, deps_package=DEPS_PACKAGE, tracer_impl=tracer_impl
    )

    global g_1

    ### create an op, run it and check the query result
    @op
    def g_1(x) -> int:
        return x + 2

    with storage.run():
        [g_1(i) for i in range(10)]

    with storage.query():
        i = Q()
        j = g_1(i)
        df = storage.df(i.named("i"), j.named("j"))

    assert df.shape == (10, 2)
    assert (df["j"] == df["i"] + 2).all()

    ### change the op semantically and check that the query result is empty
    @op
    def g_1(x) -> int:
        return x + 3

    storage.sync_component(
        component=g_1,
        is_semantic_change=True,
    )

    with storage.query():
        i = Q()
        j = g_1(i)
        df = storage.df(i.named("i"), j.named("j"))

    assert df.empty

    ### run the op again and check that the query result is correct for the new version
    with storage.run():
        [g_1(i) for i in range(10)]

    with storage.query():
        i = Q()
        j = g_1(i)
        df = storage.df(i.named("i"), j.named("j"))

    assert df.shape == (10, 2)
    assert (df["j"] == df["i"] + 3).all()

    ### go back to the old version and check that the query result is correct
    @op
    def g_1(x) -> int:
        return x + 2

    storage.sync_component(
        component=g_1,
        is_semantic_change=None,
    )
    with storage.query():
        i = Q()
        j = g_1(i)
        df = storage.df(i.named("i"), j.named("j"))
    assert df.shape == (10, 2)
    assert (df["j"] == df["i"] + 2).all()
