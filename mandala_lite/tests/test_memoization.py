from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_func_creation():
    storage = Storage()

    @op
    def add(x: int, y: int = 42) -> int:
        return x + y

    assert add.op.sig.n_outputs == 1
    assert add.op.sig.input_names == {"x", "y"}
    assert add.op.sig.defaults == {"y": 42}
    check_invariants(storage)


def test_computation():
    for evict_on_commit in [False, True]:
        Config.evict_on_commit = evict_on_commit

        storage = Storage()

        @op
        def inc(x: int) -> int:
            return x + 1

        @op
        def add(x: int, y: int) -> int:
            return x + y

        # chain some functions
        with storage.run():
            x = 23
            y = inc(x)
            z = add(x, y)
        check_invariants(storage)
        # run it again
        with storage.run():
            x = 23
            y = inc(x)
            z = add(x, y)
        check_invariants(storage)
        # do some more things
        with storage.run():
            x = 42
            y = inc(x)
            z = add(x, y)
            for i in range(10):
                z = add(z, i)
        check_invariants(storage)


def test_nosuperops():
    # disable auto-boxing things
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
    check_invariants(storage)

    # re-enable for isolation
    Config.autowrap_inputs = True
    Config.autounwrap_inputs = True
