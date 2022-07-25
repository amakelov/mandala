from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_rel_storage():
    rel_storage = DuckDBRelStorage()
    assert set(rel_storage.get_tables()) == set()
    rel_storage.create_relation(name='test', columns=[('a', None), ('b', None)], 
                                primary_key='a')
    assert set(rel_storage.get_tables()) == {'test'}
    assert rel_storage.get_data(table='test').empty
    df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']})
    ta = pa.Table.from_pandas(df)
    rel_storage.insert(relation='test', ta=ta)
    assert (rel_storage.get_data(table='test') == df).all().all()
    rel_storage.upsert(relation='test', ta=ta)
    assert (rel_storage.get_data(table='test') == df).all().all()
    rel_storage.create_column(relation='test', name='c', default_value='a')
    df['c'] = ['a', 'a']
    assert (rel_storage.get_data(table='test') == df).all().all()
    rel_storage.delete(relation='test', index=['x', 'y'])
    assert rel_storage.get_data(table='test').empty
    
def test_signatures():
    sig = Signature(
        external_name="f",
        external_input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=0,
        is_super=False,
    )

    # if internal data has not been set, it should not be accessible
    try:
        sig.internal_name
        assert False
    except ValueError:
        assert True

    try:
        sig.ext_to_int_input_map
        assert False
    except ValueError:
        assert True

    with_internal = sig._generate_internal()
    assert not sig.has_internal_data
    assert with_internal.has_internal_data

    ### invalid signature changes
    # remove input
    new = Signature(
        external_name="f",
        external_input_names={"x", "z"},
        n_outputs=1,
        defaults={"y": 42},
        version=0,
        is_super=False,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True
    # change number of outputs
    new = Signature(
        external_name="f",
        external_input_names={"x", "y"},
        n_outputs=2,
        defaults={"y": 42},
        version=0,
        is_super=False,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True
    # remove default
    new = Signature(
        external_name="f",
        external_input_names={"x", "y"},
        n_outputs=1,
        defaults={},
        version=0,
        is_super=False,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True
    # change version
    new = Signature(
        external_name="f",
        external_input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=1,
        is_super=False,
    )
    try:
        sig.update(new=new)
        assert False
    except AssertionError:
        assert True

    # add input
    sig = sig._generate_internal()
    try:
        sig.create_input(name="y", default=23)
        assert False
    except ValueError:
        assert True
    new = sig.create_input(name="z", default=23)
    assert new.external_input_names == {"x", "y", "z"}
    assert set(new.ext_to_int_input_map.keys()) == {"x", "y", "z"}
