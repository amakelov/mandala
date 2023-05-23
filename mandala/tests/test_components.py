from mandala.all import *
from mandala.tests.utils import *


def test_rel_storage():
    rel_storage = SQLiteRelStorage()
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
    # rel_storage.delete(relation="test", index=["x", "y"])
    # assert rel_storage.get_data(table="test").empty


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

    vref = wrap_atom(23)
    assert wrap_atom(vref) is vref
    try:
        wrap_atom(vref, uid="aaaaaa")
    except:
        assert True


def test_unwrapping():
    # tuples
    assert unwrap((1, 2, 3)) == (1, 2, 3)
    vrefs = (wrap_atom(23), wrap_atom(24), wrap_atom(25))
    assert unwrap(vrefs, through_collections=True) == (23, 24, 25)
    assert unwrap(vrefs, through_collections=False) == vrefs
    # sets
    assert unwrap({1, 2, 3}) == {1, 2, 3}
    vrefs = {wrap_atom(23), wrap_atom(24), wrap_atom(25)}
    assert unwrap(vrefs, through_collections=True) == {23, 24, 25}
    assert unwrap(vrefs, through_collections=False) == vrefs
    # lists
    assert unwrap([1, 2, 3]) == [1, 2, 3]
    vrefs = [wrap_atom(23), wrap_atom(24), wrap_atom(25)]
    assert unwrap(vrefs, through_collections=True) == [23, 24, 25]
    assert unwrap(vrefs, through_collections=False) == vrefs
    # dicts
    assert unwrap({"a": 1, "b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}
    vrefs = {"a": wrap_atom(23), "b": wrap_atom(24), "c": wrap_atom(25)}
    assert unwrap(vrefs, through_collections=True) == {"a": 23, "b": 24, "c": 25}
    assert unwrap(vrefs, through_collections=False) == vrefs


def test_reprs():
    x = wrap_atom(23)
    repr(x), str(x)


################################################################################
### contexts
################################################################################
def test_nesting_new_api():
    storage = Storage()

    with storage.run() as c:
        assert c.mode == MODES.run

    with storage.query() as q:
        assert q.mode == MODES.query

    with storage.run() as c_1:
        with storage.run() as c_2:
            with storage.run() as c_3:
                assert c_1 is c_2 is c_3

    with storage.run() as c:
        assert c.mode == MODES.run
        assert c.storage is storage
        with storage.query() as q:
            assert q is c
            assert q.storage is storage
            assert q.mode == MODES.query
        assert c.mode == MODES.run


def test_noop():
    # check that ops are noops when not in a context

    @op
    def inc(x: int) -> int:
        return x + 1

    assert inc(23) == 24


def test_failures():
    storage = Storage()

    try:
        with storage.run(bla=23):
            pass
    except:
        assert True


################################################################################
### test ops
################################################################################
def test_signatures():
    sig = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=0,
        input_annotations={"x": int, "y": int},
        output_annotations=[Any],
    )

    # if internal data has not been set, it should not be accessible
    try:
        sig.internal_name
    except ValueError:
        assert True

    try:
        sig.ui_to_internal_input_map
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
        input_annotations={"x": int, "z": int},
        output_annotations=[Any],
    )
    try:
        sig.update(new=new)
    except ValueError:
        assert True
    new = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={},
        version=0,
        input_annotations={"x": int, "y": int},
        output_annotations=[Any],
    )
    try:
        sig.update(new=new)
    except ValueError:
        assert True
    # change version
    new = Signature(
        ui_name="f",
        input_names={"x", "y"},
        n_outputs=1,
        defaults={"y": 42},
        version=1,
        input_annotations={"x": int, "y": int},
        output_annotations=[Any],
    )
    try:
        sig.update(new=new)
    except ValueError:
        assert True

    # add input
    sig = sig._generate_internal()
    try:
        sig.create_input(name="y", default=23, annotation=Any)
    except ValueError:
        assert True
    new = sig.create_input(name="z", default=23, annotation=Any)
    assert new.input_names == {"x", "y", "z"}


def test_output_name_failure():

    try:

        @op
        def f(output_0: int) -> int:
            return output_0

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
    except Exception:
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


################################################################################
### test storage
################################################################################
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
    storage.rel_adapter.mattach(vrefs=[factorizations_shallow.obj[0]])
    assert unwrap(factorizations_shallow.obj[0]) == [42]

    # result, call, wrapped_inputs = storage.call_run(
    #     func_op=get_factorizations.func_op,
    #     inputs={"n": 42},
    #     _call_depth=0,
    # )


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
            #! this is now in memory b/c new caching
            # assert not factorizations[0].in_memory
            # factorizations[0][0]
            # assert factorizations[0].in_memory

            #! this is now in memory b/c new caching
            # for elt in factorizations[1]:
            #     assert not elt.in_memory

    except Exception as e:
        raise e
    finally:
        db_path.unlink()


def test_magics():
    db_path = OUTPUT_ROOT / "test_magics.db"
    if db_path.exists():
        db_path.unlink()
    storage = Storage(db_path=db_path)

    try:
        Config.enable_ref_magics = True

        @op
        def inc(x: int) -> int:
            return x + 1

        with storage.run():
            x = inc(23)

        with storage.run():
            x = inc(23)
            assert not x.in_memory
            if x > 0:
                y = inc(x)
            assert x.in_memory

        with storage.run():
            x = inc(23)
            y = inc(x)
            if x + y > 0:
                z = inc(x)

        with storage.run():
            x = inc(23)
            y = inc(x)
            if x:
                z = inc(x)

    except Exception as e:
        raise e
    finally:
        db_path.unlink()


def test_spillover():
    db_path = OUTPUT_ROOT / "test_spillover.db"
    if db_path.exists():
        db_path.unlink()
    spillover_dir = OUTPUT_ROOT / "test_spillover/"
    if spillover_dir.exists():
        shutil.rmtree(spillover_dir)
    storage = Storage(db_path=db_path, spillover_dir=spillover_dir)

    try:
        import numpy as np

        @op
        def create_large_array() -> np.ndarray:
            return np.random.rand(10_000_000)

        with storage.run():
            x = create_large_array()

        assert len(os.listdir(spillover_dir)) == 1
        path = spillover_dir / os.listdir(spillover_dir)[0]
        with open(path, "rb") as f:
            data = unwrap(joblib.load(f))
        assert np.allclose(data, unwrap(x))

        with storage.run():
            x = create_large_array()
            assert not x.in_memory
            x = unwrap(x)

    except Exception as e:
        raise e
    finally:
        db_path.unlink()
        shutil.rmtree(spillover_dir)


def test_batching_unit():

    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.batch():
        y = inc(23)

    assert unwrap(y) == 24
    assert y.uid is not None
    all_data = storage.rel_storage.get_all_data()
    assert all_data[Config.vref_table].shape[0] == 2
    assert all_data[inc.func_op.sig.versioned_ui_name].shape[0] == 1


def test_exclude_arg():
    storage = Storage()

    @op
    def inc(x: int, __excluded__=False) -> int:
        return x + 1

    with storage.run():
        y = inc(23)


def test_provenance():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        y = inc(23)
    storage.prov(y)
    storage.print_graph(y)

    ### struct inputs
    @op
    def avg_list(nums: List[int]) -> float:
        return sum(nums) / len(nums)

    @op
    def avg_dict(nums: Dict[str, int]) -> float:
        return sum(nums.values()) / len(nums)

    @op
    def avg_set(nums: Set[int]) -> float:
        return sum(nums) / len(nums)

    with storage.run():
        x = avg_list([1, 2, 3])
        y = avg_dict({"a": 1, "b": 2, "c": 3})
        z = avg_set({1, 2, 3})
    for v in [x, y, z]:
        storage.prov(v)
        storage.print_graph(v)

    ### struct outputs
    @op
    def get_list() -> List[int]:
        return [1, 2, 3]

    @op
    def get_dict() -> Dict[str, int]:
        return {"a": 1, "b": 2, "c": 3}

    @op
    def get_set() -> Set[int]:
        return {1, 2, 3}

    with storage.run():
        x = get_list()
        y = get_dict()
        z = get_set()
        a = x[0]
        b = y["a"]
    for v in [x, y, z, a, b]:
        storage.prov(v)
        storage.print_graph(v)

    ### a mess of stuff
    with storage.run():
        a = get_list()
        x = avg_list(a[:2])
        y = avg_dict(get_dict())
        z = avg_set(get_set())
    for v in [a, x, y, z]:
        storage.prov(v)
        storage.print_graph(v)
