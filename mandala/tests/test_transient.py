from .utils import *
from .funcs import *
from .conftest import _setup_tests, setup_tests

def test_unit(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    calls_state = {
        'add': 0,
        'decompose': 0
    }
    
    with define(storage):
        @op()
        def add_array_recomputable(x:Array, y:Array) -> Array:
            calls_state['add'] += 1
            return AsTransient(x + y)
        
        @op()
        def decompose_array_recomputable(x:Array) -> ArrayList:
            calls_state['decompose'] += 1
            return [AsTransient(x[i]) for i in range(x.shape[0])]
        
    ### test plain atoms
    with run(storage) as c:
        calls_state['add'] = 0
        x = np.zeros(shape=(3, 4))
        y = np.zeros(shape=(3, 4))
        res = add_array_recomputable(x=x, y=y)
        assert calls_state['add'] == 1
        res = add_array_recomputable(x=x, y=y)
        assert calls_state['add'] == 2
        c.commit()
    ### test a compound value with transient constituents
    with run(storage) as c:
        calls_state['decompose'] = 0
        x = np.zeros(shape=(3, 4))
        res = decompose_array_recomputable(x=x)
        assert calls_state['decompose'] == 1
        res = decompose_array_recomputable(x=x)
        assert calls_state['decompose'] == 2
        c.commit()
    storage.drop_instance_data(answer=True)

def test_forcing():
    storage = Storage()

    @op(storage)
    def inc(x:int) -> int:
        return AsTransient(x + 1)
    
    @op(storage)
    def add(x:int, y:int) -> int:
        return AsTransient(x + y)
    
    ### check that repeated calls with transient returns are not stored again 
    with run(storage) as c:
        current = 0
        for x in range(10):
            current = add(inc(current), current)
        c.commit()
    with run(storage) as c:
        current = 0
        for x in range(10):
            current = add(inc(current), current)
    temp_partition = CALLS.default_temp_partition    
    assert not storage.call_st.locs(partitions=[temp_partition])
    storage.drop_instance_data(answer=True)

    ############################################################################ 
    ### chains of transient calls with lazy retracing and forcing
    ############################################################################ 
    with run(storage, autocommit=True):
        current = 0
        for x in range(10):
            current = add(inc(current), current)
    # without forcing it should fail
    try:
        with run(storage, lazy=True):
            current = 0
            for x in range(11):
                current = add(inc(current), current)
        assert False
    except VRefNotInMemoryError:
        assert True
    except:
        assert False
    # with forcing it should work
    with run(storage, lazy=True, force=True):
        current = 0
        for x in range(11):
            current = add(inc(current), current)
    assert unwrap(current) == 2047

    ############################################################################ 
    ### force only the calls that need forcing
    ############################################################################ 
    @op(storage)
    def get_prime_factors(x:int) -> TList[int]:
        if x < 2:
            return []
        nums = list(range(2, x + 1))
        primes = [a for a in nums if x % a == 0 and
                  all([a % div != 0 for div in nums if 1 < div and div < a])]
        return primes
    
    ### run some computations
    with run(storage, lazy=True, autocommit=True):
        for x in range(10, 20):
            facts = get_prime_factors(x=x)
            for fact in facts:
                a = inc(fact)
                b = add(a, x)

    ### add stuff with forcing only the transient things
    with run(storage, lazy=True, autocommit=True) as r:
        for x in range(10, 20):
            facts = get_prime_factors(x=x)
            with r(force=True):
                for fact in facts:
                    a = inc(fact)
                    b = add(a, x)
                    c = add(a, b)
    storage.drop_instance_data(answer=True)

    ### autonesting 
    with run(storage, lazy=True, autocommit=True):
        for x in range(10, 20):
            facts = get_prime_factors(x=x)
            for fact in facts:
                a = inc(fact)
                b = add(a, x)
    with run(storage, lazy=True, autocommit=True) as r:
        for x in range(10, 20):
            facts = get_prime_factors(x=x)
            with run(force=True):
                for fact in facts:
                    a = inc(fact)
                    b = add(a, x)
                    c = add(a, b)
    storage.drop_instance_data(answer=True)

def test_delayed_storage(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    with define(storage):
        @op()
        def inc_array(x:Array) -> Array:
            return AsDelayedStorage(x+1)
    
    ### unit test
    with run(storage) as c:
        a = np.zeros(shape=(10, 10))
        b = inc_array(a)
        assert not c.get(c.where_is(vref=b)).in_memory 
        c.save(b)
        assert c.get(c.where_is(vref=b)).in_memory 
        c.commit()
    storage.drop_instance_data(answer=True)
    
    ### test in a loop
    with run(storage) as c:
        a = np.zeros(shape=(3, 4))
        vals = []
        for i in range(100):
            a = inc_array(x=a)
            if i % 5 == 0:
                c.save(a)
            vals.append(a)
        c.commit()
    for i, val in enumerate(vals):
        recovered = storage.get(storage.where_is(vref=val))
        if i % 5 == 0:
            assert recovered.in_memory
        else:
            assert not recovered.in_memory
        
    storage.drop_instance_data(answer=True)