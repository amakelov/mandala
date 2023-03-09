from mandala.all import *
from mandala.tests.utils import *


def test_unit():
    storage = Storage()

    @op
    def f(x) -> int:
        return Transient(x + 1)

    with storage.run(attach_call_to_outputs=True):
        a = f(42)
        call: Call = a._call
    assert call.transient
    assert a.transient
    assert unwrap(a) == 43

    with storage.run(attach_call_to_outputs=True):
        a = f(42)
        call: Call = a._call
    assert not a.in_memory
    assert a.transient
    assert call.transient

    with storage.run(recompute_transient=True, attach_call_to_outputs=True):
        a = f(42)
        call: Call = a._call

    assert a.in_memory
    assert a.transient
    assert call.transient


def test_composition():
    storage = Storage()

    @op
    def f(x) -> int:
        return Transient(x + 1)

    @op
    def g(x) -> int:
        return Transient(x**2)

    with storage.run():
        a = f(42)
        b = g(a)

    with storage.run():
        a = f(23)

    try:
        with storage.run():
            a = f(23)
            b = g(a)
        assert False
    except ValueError as e:
        assert True

    with storage.run(recompute_transient=True):
        a = f(23)
        b = g(a)
    assert unwrap(b) == 576
