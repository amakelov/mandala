from mandala.all import *
from mandala.tests.utils import *


def test_rel_storage():
    rel_storage = DuckDBRelStorage()
    assert set(rel_storage.get_tables()) == set()
    rel_storage.create_relation(
        name="test", columns=[("a", None), ("b", None)], primary_key="a", defaults={}
    )
    assert set(rel_storage.get_tables()) == {"test"}
    assert rel_storage.get_data(table="test").empty
    df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
    ta = pa.Table.from_pandas(df)
    rel_storage.insert(relation="test", ta=ta)
    assert (rel_storage.get_data(table="test") == df).all().all()
    rel_storage.upsert(relation="test", ta=ta)
    assert (rel_storage.get_data(table="test") == df).all().all()
    rel_storage.create_column(relation="test", name="c", default_value="a")
    df["c"] = ["a", "a"]
    assert (rel_storage.get_data(table="test") == df).all().all()
    rel_storage.delete(relation="test", index=["x", "y"])
    assert rel_storage.get_data(table="test").empty


def test_main_storage():
    storage = Storage()
    # check that things work on an empty storage
    storage.sig_adapter.load_state()
    storage.rel_adapter.get_all_call_data()


def test_wrapping():
    assert unwrap(23) == 23
    assert unwrap(23.0) == 23.0
    assert unwrap("23") == "23"
    assert unwrap([1, 2, 3]) == [1, 2, 3]

    vref = wrap(23)
    assert wrap(vref) is vref
    try:
        wrap(vref, uid="aaaaaa")
        assert False
    except:
        assert True


def test_unwrapping():
    # tuples
    assert unwrap((1, 2, 3)) == (1, 2, 3)
    vrefs = (wrap(23), wrap(24), wrap(25))
    assert unwrap(vrefs, through_collections=True) == (23, 24, 25)
    assert unwrap(vrefs, through_collections=False) == vrefs
    # sets
    assert unwrap({1, 2, 3}) == {1, 2, 3}
    vrefs = {wrap(23), wrap(24), wrap(25)}
    assert unwrap(vrefs, through_collections=True) == {23, 24, 25}
    assert unwrap(vrefs, through_collections=False) == vrefs
    # lists
    assert unwrap([1, 2, 3]) == [1, 2, 3]
    vrefs = [wrap(23), wrap(24), wrap(25)]
    assert unwrap(vrefs, through_collections=True) == [23, 24, 25]
    assert unwrap(vrefs, through_collections=False) == vrefs
    # dicts
    assert unwrap({"a": 1, "b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}
    vrefs = {"a": wrap(23), "b": wrap(24), "c": wrap(25)}
    assert unwrap(vrefs, through_collections=True) == {"a": 23, "b": 24, "c": 25}
    assert unwrap(vrefs, through_collections=False) == vrefs


def test_reprs():
    x = wrap(23)
    repr(x), str(x)
