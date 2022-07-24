from mandala_lite.all import *
from mandala_lite.tests.utils import *


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
