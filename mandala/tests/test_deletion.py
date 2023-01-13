from mandala.all import *
from mandala.tests.utils import *
from mandala.core.builtins_ import Builtins


def test_unit():
    storage = Storage()

    @op
    def f(x: int) -> int:
        return x + 1

    with storage.run():
        f(1)

    df = storage.get_table(f)
    assert len(df) == 1

    with storage.run():
        with storage.delete():
            f(1)

    df = storage.get_table(f)
    assert len(df) == 0


def test_nesting():
    storage = Storage()

    @op
    def f(x: int) -> int:
        return x + 1

    @op
    def g(x: int) -> int:
        return x - 1

    with storage.run():
        for i in range(10):
            a = f(i)
            b = g(a)

    with storage.run():
        for i in range(10):
            a = f(i)
            with storage.delete():
                b = g(a)

    df_f = storage.get_table(f)
    df_g = storage.get_table(g)
    assert len(df_f) == 10
    assert len(df_g) == 0


def test_structs():
    storage = Storage()

    @op
    def get_prime_factors(n: int) -> Set[int]:
        factors = set()
        d = 2
        while d * d <= n:
            while (n % d) == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    with storage.run():
        get_prime_factors(15)

    with storage.delete():
        get_prime_factors(15)

    set_op = FuncInterface(Builtins.set_op)

    primes_df = storage.get_table(get_prime_factors)
    set_df = storage.get_table(set_op)
    assert len(primes_df) == 0
    assert len(set_df) == 0

    with storage.run():
        get_prime_factors(15)
        factors_105 = get_prime_factors(105)

    with storage.delete():
        get_prime_factors(15)

    set_df = storage.get_table(set_op)
    assert len(set_df["st"].unique()) == 1
    assert len(set_df["elt"]) == 3
    for elt in factors_105:
        storage.obj_get(obj_uid=elt.uid)
