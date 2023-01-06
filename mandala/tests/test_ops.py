from mandala.all import *
from mandala.tests.utils import *


def test_signatures():
    sig = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=0,
    )

    # if internal data has not been set, it should not be accessible
    try:
        sig.internal_name
        assert False
    except ValueError:
        assert True

    try:
        sig.ui_to_internal_input_map
        assert False
    except ValueError:
        assert True

    with_internal = sig._generate_internal()
    assert not sig.has_internal_data
    assert with_internal.has_internal_data

    ### invalid signature changes
    # remove input
    new = Signature(
        ui_name="f",
        input_names={"x", "z"},
        n_outputs=1,
        defaults={"y": 42},
        version=0,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True
    # # change number of outputs
    # new = Signature(
    #     ui_name="f",
    #     input_names={"x", "y"},
    #     n_outputs=2,
    #     defaults={"y": 42},
    #     version=0,
    # )
    # try:
    #     sig.update(new=new)
    #     assert False
    # except ValueError:
    #     assert True
    # remove default
    new = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={},
        version=0,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True
    # change version
    new = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=1,
    )
    try:
        sig.update(new=new)
        assert False
    except ValueError:
        assert True

    # add input
    sig = sig._generate_internal()
    try:
        sig.create_input(name="y", default=23)
        assert False
    except ValueError:
        assert True
    new = sig.create_input(name="z", default=23)
    assert new.input_names == {"x", "y", "z"}


def test_output_name_failure():

    try:

        @op
        def f(output_0: int) -> int:
            return output_0

        assert False
    except:
        assert True


def test_changing_num_outputs():

    storage = Storage()

    @op
    def f(x: int):
        return x

    try:
        with storage.run():
            f(1)
        assert False
    except AssertionError:
        assert True

    @op
    def f(x: int) -> int:
        return x

    with storage.run():
        f(1)

    @op
    def f(x: int) -> Tuple[int, int]:
        return x

    try:
        with storage.run():
            f(1)
        assert False
    except ValueError:
        assert True

    @op
    def f(x: int) -> int:
        return x

    with storage.run():
        f(1)


def test_nout():
    storage = Storage()

    @op(nout=2)
    def f(x: int):
        return x, x

    with storage.run():
        a, b = f(1)
        assert unwrap(a) == 1 and unwrap(b) == 1

    @op(nout=0)
    def g(x: int) -> Tuple[int, int]:
        pass

    with storage.run():
        c = g(1)
        assert c is None
