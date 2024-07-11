from mandala.imports import *
from mandala.deps.shallow_versions import DAG
from mandala.deps.tracers import DecTracer, SysTracer
from pathlib import Path
import os
import uuid
import pytest

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

def generate_path(ext: str) -> Path:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/output")
    fname = str(uuid.uuid4()) + ext
    return output_dir / fname

# MODULE_NAME = "mandala.tests.test_versioning"
# DEPS_PACKAGE = "mandala.tests"
DEPS_PATH = Path(__file__).parent.absolute().resolve()
MODULE_NAME = "test_versioning"
# MODULE_NAME = '__main__'


def _test_version_reprs(storage: Storage):
    for dag in storage.get_versioner().component_dags.values():
        for compact in [True, False]:
            dag.show(compact=compact)
    for version in storage.get_versioner().get_flat_versions().values():
        storage.get_versioner().present_dependencies(commits=version.semantic_expansion)
    storage.get_versioner().global_topology.show(path=generate_path(ext=".png"))
    repr(storage.get_versioner().global_topology)


@pytest.mark.parametrize("tracer_impl", [DecTracer])
def test_unit(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, tracer_impl=tracer_impl
    )

    # to be able to import this name
    global f_1, A

    A = 42

    @op
    def f_1(x) -> int:
        return 23 + A

    with storage:
        f_1(1)

    vs = storage.get_versioner()
    print(vs.versions)
    f_1_versions = vs.versions[MODULE_NAME, "f_1"]
    assert len(f_1_versions) == 1
    version = f_1_versions[list(f_1_versions.keys())[0]]
    assert set(version.support) == {(MODULE_NAME, "f_1"), (MODULE_NAME, "A")}
    _test_version_reprs(storage=storage )

@pytest.mark.parametrize("tracer_impl", [DecTracer])
def test_deps(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, tracer_impl=tracer_impl
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

    with storage:
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


@pytest.mark.parametrize("tracer_impl", [DecTracer])
def test_changes(tracer_impl):
    storage = Storage(
        deps_path=DEPS_PATH, tracer_impl=tracer_impl
    )

    global f

    @op
    def f(x) -> int:
        return x + 1

    with storage:
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
    with storage:
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
    with storage:
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
    with storage:
        f(1)

    f_versions = storage.get_versioner().versions[MODULE_NAME, "f"]
    assert len(f_versions) == 3
    semantic_versions = [v.semantic_version for v in f_versions.values()]
    assert len(set(semantic_versions)) == 3
    _test_version_reprs(storage=storage)


@pytest.mark.parametrize("tracer_impl", [DecTracer])
def _test_dependency_patterns(tracer_impl):
    # this test is borked currently
    storage = Storage(
        deps_path=DEPS_PATH, tracer_impl=tracer_impl
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
    @track
    @op
    def f_3(x) -> int:
        if x % 2 == 0:
            return f_2(2 * x)
        else:
            return f_1(x + 1)

    with storage:
        x = f_3(0)
    
    call = storage.get_ref_creator(x)
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_2"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
        (MODULE_NAME, "f_1"),
    }
    with storage:
        x = f_3(1)
    call = storage.get_ref_creator(x)
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_3"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
    }

    # using a lambda
    @track
    @op
    def f_4(x) -> int:
        f = lambda y: f_1(y) + B[0]
        return f(x)

    # make sure the call in the lambda is detected
    with storage:
        x = f_4(10)
    call = storage.get_ref_creator(x)
    version = storage.get_versioner().get_flat_versions()[call.content_version]
    assert version.support == {
        (MODULE_NAME, "f_4"),
        (MODULE_NAME, "f_1"),
        (MODULE_NAME, "A"),
        (MODULE_NAME, "B"),
    }

    # using comprehensions and generators
    @op
    def f_5(x) -> int:
        x = storage.unwrap(x)
        a = [f_1.f(y) for y in range(x)]
        b = {f_2.f(y) for y in range(x)}
        c = {y: f_3.f(y) for y in range(x)}
        return sum(storage.unwrap(f_4.f(y)) for y in range(x))

    with storage:
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
    @op
    def f_6(x) -> int:
        x = storage.unwrap(x)
        # nested list comprehension
        a = sum([sum([f_1.f(y) for y in range(x)]) for z in range(x)])
        # nested comprehension with generator
        b = sum(sum(f_2.f(y) for y in range(x)) for z in range(storage.unwrap(f_3.f(x))))
        return a + b

    with storage:
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
    storage.get_code(version_id=version.content_version)

