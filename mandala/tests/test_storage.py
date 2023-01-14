from mandala.all import *
from mandala.tests.utils import *

OUTPUT_ROOT = Path(__file__).parent / "output"


def test_get():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        y = inc(23)

    y_full = storage.rel_adapter.obj_get(uid=y.uid)
    assert y_full.in_memory
    assert unwrap(y_full) == 24

    y_lazy = storage.rel_adapter.obj_get(uid=y.uid, _attach_atoms=False)
    assert not y_lazy.in_memory
    assert y_lazy.obj is None

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
        factors = get_prime_factors(42)

    factors_full = storage.rel_adapter.obj_get(uid=factors.uid)
    assert factors_full.in_memory
    assert all([x.in_memory for x in factors_full])

    factors_shallow = storage.rel_adapter.obj_get(uid=factors.uid, depth=1)
    assert factors_shallow.in_memory
    assert all([not x.in_memory for x in factors_shallow])

    storage.rel_adapter.mattach(vrefs=[factors_shallow])
    assert all([x.in_memory for x in factors_shallow])

    @superop
    def get_factorizations(n: int) -> List[List[int]]:
        # get all factorizations of a number into factors
        n = unwrap(n)
        divisors = [i for i in range(2, n + 1) if n % i == 0]
        result = [[n]]
        for divisor in divisors:
            sub_solutions = unwrap(get_factorizations(n // divisor))
            result.extend(
                [
                    [divisor] + sub_solution
                    for sub_solution in sub_solutions
                    if min(sub_solution) >= divisor
                ]
            )
        return result

    with storage.run():
        factorizations = get_factorizations(42)

    factorizations_full = storage.rel_adapter.obj_get(uid=factorizations.uid)
    assert unwrap(factorizations_full) == [[42], [2, 21], [2, 3, 7], [3, 14], [6, 7]]
    factorizations_shallow = storage.rel_adapter.obj_get(
        uid=factorizations.uid, depth=1
    )
    assert factorizations_shallow.in_memory
    storage.rel_adapter.mattach(vrefs=[factorizations_shallow[0]])
    assert unwrap(factorizations_shallow[0]) == [42]

    result, call = storage.call_run(
        func_op=get_factorizations.func_op,
        inputs={"n": 42},
    )


def test_persistent():
    db_path = OUTPUT_ROOT / "test_persistent.db"
    if db_path.exists():
        db_path.unlink()
    storage = Storage(db_path=db_path)

    try:

        @op
        def inc(x: int) -> int:
            return x + 1

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

        @superop
        def get_factorizations(n: int) -> List[List[int]]:
            # get all factorizations of a number into factors
            n = unwrap(n)
            divisors = [i for i in range(2, n + 1) if n % i == 0]
            result = [[n]]
            for divisor in divisors:
                sub_solutions = unwrap(get_factorizations(n // divisor))
                result.extend(
                    [
                        [divisor] + sub_solution
                        for sub_solution in sub_solutions
                        if min(sub_solution) >= divisor
                    ]
                )
            return result

        with storage.run():
            y = inc(23)
            factors = get_prime_factors(42)
            factorizations = get_factorizations(42)
            assert all([x.in_memory for x in (y, factors, factorizations)])

        with storage.run():
            y = inc(23)
            factors = get_prime_factors(42)
            factorizations = get_factorizations(42)
            assert all([not x.in_memory for x in (y, factors, factorizations)])

        with storage.run(lazy=False):
            y = inc(23)
            factors = get_prime_factors(42)
            factorizations = get_factorizations(42)
            assert all([x.in_memory for x in (y, factors, factorizations)])

        with storage.run():
            y = inc(23)
            assert not y.in_memory
            y._auto_attach()
            assert y.in_memory
            factors = get_prime_factors(42)
            assert not factors.in_memory
            7 in factors
            assert factors.in_memory

            factorizations = get_factorizations(42)
            assert not factorizations.in_memory
            n = len(factorizations)
            assert factorizations.in_memory
            assert not factorizations[0].in_memory
            factorizations[0][0]
            assert factorizations[0].in_memory

            for elt in factorizations[1]:
                assert not elt.in_memory

    except Exception as e:
        raise e
    finally:
        db_path.unlink()
