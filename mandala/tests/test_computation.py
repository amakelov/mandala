from .utils import *
from .funcs import *
from .conftest import _setup_tests, setup_tests

def setup_debug():
    default_kv = JoblibStorage
    # default_kv = SQLiteStorage
    storage, cts = _setup_tests(default_kv=default_kv)
    
################################################################################
### tests
################################################################################
# @pytest.mark.parametrize('setup_tests', [JoblibStorage, SQLiteStorage], indirect=True)
def test_computations(setup_tests):
    def run_storage_actions(storage:Storage):
        storage.verify_static()
        storage.rel_adapter.describe_rels()
        storage.rel_adapter.describe_vrefs()
    storage, cts = setup_tests
    storage:Storage
    for parallel in (True, False):
        if parallel:
            storage.parallelize_all_kvs()
        with run(storage=storage, partition='test') as c:
            for ct in cts:
                ct.run_all()
                c.commit()
                run_storage_actions(storage=storage)
        # do it again, raising if any new call is made
        with run(storage=storage, allow_calls=False) as c:
            for ct in cts:
                ct.run_all()
                run_storage_actions(storage=storage)
        storage.drop_instance_data(answer=True)

def test_buffers(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    buffer = storage.make_buffer()

    with run(storage=storage, buffer=buffer, partition='test') as c:
        things = []
        for i, j in itertools.product(range(20), range(30)):
            x = add(i, j)
            things.append(x)
        means = mean([mean(things[i:i+100]) for i in range(3)])
        c.commit(buffer_first=True)
    storage.drop_instance_data(answer=True)
            
def test_lazy_patterns(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    ### first pass: compute
    with run(storage=storage, partition='test') as c:
        numbers = [4, 5, 6, 7, 8]
        all_factors = []
        for num in numbers:
            all_factors.append(get_prime_factors(x=num))
        result_1 = mean_2d(arr=all_factors)
        result_2 = mean(x=[mean(x=elt) for elt in all_factors])
        c.commit()

    ### resume lazily with explicit attaching of values
    with run(storage=storage, lazy=True) as c:
        numbers = [4, 5, 6, 7, 8]
        all_factors = []
        for num in numbers:
            all_factors.append(get_prime_factors(x=num))
        assert all([not vref.in_memory for vref in all_factors])
        for vref in all_factors: c.attach(vref=vref)
        result_1 = mean_2d(arr=all_factors)
        assert not result_1.in_memory
        means = [mean(x=elt) for elt in all_factors]
        for vref in means: c.attach(vref=vref)
        result_2 = mean(x=means)
        assert not result_2.in_memory
        c.attach(result_1)
        c.attach(result_2)
        assert result_1.in_memory
        assert result_2.in_memory
        assert result_1.obj() == result_2.obj()
    
    ### now do some other stuff
    with run(storage=storage, partition='test') as c:
        nums = range(100, 200)
        inc_nums = [inc(x=x) for x in nums]
        prime_things = [get_prime_factors(x=x) for x in inc_nums]
        c.commit()

    ### do more stuff with lazy loading, and operate on lazy values to check 
    ### that they are automatically put in memory when needed by a computation
    with run(storage=storage, lazy=True, partition='test') as c:
        nums = range(100, 200)
        inc_nums = [inc(x=x) for x in nums]
        prime_things = [get_prime_factors(x=x) for x in inc_nums]
        for lst in prime_things:
            assert not lst.in_memory
            primes_mean = mean(lst)
            assert lst.in_memory
            # at this point, the list should have been loaded
            primes_inc = [inc(x) for x in lst]
        c.commit()
        
    ### test attachment of compound values
    with run(storage=storage, lazy=True) as c:
        nums = range(100, 200)
        inc_nums = [inc(x=x) for x in nums]
        prime_things = [get_prime_factors(x=x) for x in inc_nums]
        for lst in prime_things:
            assert not lst.in_memory
            c.attach(vref=lst, shallow=True)
            assert lst.in_memory
            assert len(lst) == len(c.get(c.where_is(lst)))
    storage.drop_instance_data(answer=True)
    
    ############################################################################ 
    ### test automatic attachment of compound values by iterators, getitem, len
    ############################################################################ 
    # run computation for the first time, where some outputs come as a list
    first_counter = 0
    num_factors = []
    factors_list = []
    with run(storage, lazy=True, partition='temp') as c:
        for num in range(10, 20):
            factors = get_prime_factors(x=num)
            num_factors.append(len(factors))
            factors_list.append(unwrap(factors))
            for factor in factors:
                first_counter += 1
                inc(x=factor)
        c.commit()
    # retrace lazily with iterator
    second_counter = 0
    with run(storage, lazy=True) as c:
        for num in range(10, 20):
            factors = get_prime_factors(x=num)
            for factor in factors:
                second_counter += 1
                inc(x=factor)
    assert first_counter == second_counter
    # retrace lazily with explicit indexing
    with run(storage, lazy=True) as c:
        for i, num in enumerate(range(10, 20)):
            factors = get_prime_factors(x=num)
            first_factor = factors[0]
            assert unwrap(first_factor) == factors_list[i][0]
    # get lengths
    with run(storage, lazy=True) as c:
        for i, num in enumerate(range(10, 20)):
            factors = get_prime_factors(x=num)
            assert len(factors) == num_factors[i]
    assert first_counter == second_counter
    storage.drop_instance_data(answer=True)


def test_wrapped_input_ops():
    storage = Storage()
    
    @op(storage)
    def get_list(x:int, times:int) -> TList[int]:
        return [x for _ in range(times)]
    
    @op(storage, unwrap_inputs=False)
    def lazy_list_avg(lst:TList[int], __context__:Context) -> float:
        storage:Storage = __context__.storage
        storage.attach(vref=lst, shallow=True)
        running_sum = 0
        for elt in lst:
            loc = storage.where_is(elt)
            running_sum += unwrap(storage.get(loc))
        return running_sum / len(lst)
    
    with run(storage):
        get_list(x=23, times=42)
    
    with run(storage, lazy=True) as c:
        lst = get_list(x=23, times=42)
        final = lazy_list_avg(lst=lst, __context__=c)
    assert unwrap(final) == 23.0



def _test_mrun(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    ############################################################################ 
    ### unit test + mrecoverable test
    ############################################################################ 
    ### prepare query 
    def run_add_query():
        with query(storage=storage) as c:
            x = Any()
            y = Any()
            z = add(x=x, y=y)
            df = c.qeval(x, y, z, names=['x', 'y', 'z'])
        return df

    ### test on empty
    with run(storage=storage, partition='temp') as c:
        xs = []
        ys = []
        results = add.mcall(x=xs, y=ys)
        assert len(results) == 0
        c.commit()

    with run(storage=storage, partition='temp') as c:
        xs = list(range(10))
        ys = list(range(10))
        mrec = add.mrecoverable(x=xs, y=ys)
        assert not any(mrec)
        results = add.mcall(x=xs, y=ys)
        c.commit()
        mrec = add.mrecoverable(x=xs, y=ys)
        assert all(mrec)

    ### partially computed set of calls
    with run(storage=storage, partition='temp', lazy=True) as c:
        results = add.mcall(x=list(range(20)), y=list(range(20)))
        c.commit()
    
    df = run_add_query()
    assert df.shape[0] == 20
    assert set(df['x'].values) == set(range(20))
    assert set(df['y'].values) == set(range(20))
    assert np.all(df['x'] + df['y'] == df['z'])
    storage.drop_instance_data(answer=True)

    ### check automatic attachment
    with run(storage=storage, partition='temp') as c:
        xs = [inc(i) for i in range(20, 30)]
        ys = [inc(i) for i in range(20, 30)]
        c.commit()
    with run(storage=storage, lazy=True, partition='temp') as c:
        xs = [inc(i) for i in range(20, 30)]
        ys = [inc(i) for i in range(20, 30)]
        add.mcall(x=xs, y=ys)
        c.commit()
    
    df = run_add_query()
    assert df.shape[0] == 10
    assert set(df['x'].values) == set(range(21, 31))
    assert set(df['y'].values) == set(range(21, 31))
    assert np.all(df['x'] + df['y'] == df['z'])
    storage.drop_instance_data(answer=True)
    
    ############################################################################ 
    ### test multi-output func 
    ############################################################################ 
    def run_inc_and_dec_query():
        with query(storage=storage) as c:
            x = Int()
            x_inc, x_dec = inc_and_dec(x)
            df = c.qeval(x, x_inc, x_dec, names=['x', 'x_inc', 'x_dec'])
        return df

    with run(storage=storage, partition='temp') as c:
        incs, decs = inc_and_dec.mcall(x=list(range(10)))
        c.commit()
    
    df = run_inc_and_dec_query()
    assert df.shape[0] == 10
    assert set(df['x'].values) == set(range(10))
    assert np.all(df['x'] + 1 == df['x_inc'])
    assert np.all(df['x'] - 1 == df['x_dec'])
    storage.drop_instance_data(answer=True)

    ############################################################################ 
    ### test __returns__, context
    ############################################################################ 
    with run(storage=storage, partition='temp') as c:
        xs = list(range(10))
        ys = list(range(10))
        c_copy = c.spawn()
        # write 0 to ensure we are properly setting this
        inc_and_dec.mcall(x=xs, __returns__=([0 for _ in range(10)], [0 for _ in range(10)]), 
                          __context__=c_copy)
        add.mcall(x=xs, y=ys, __returns__=[0 for _ in range(10)], __context__=c_copy)
        c.commit()
    
    add_df = run_add_query()
    inc_and_dec_df = run_inc_and_dec_query()
    assert add_df.shape[0] == 10
    assert set(add_df['x']) == set(range(10))
    assert set(add_df['y']) == set(range(10))
    assert np.all(add_df['z'] == 0)
    assert inc_and_dec_df.shape[0] == 10
    assert set(inc_and_dec_df['x']) == set(range(10))
    assert np.all(inc_and_dec_df['x_inc'] == 0)
    assert np.all(inc_and_dec_df['x_dec'] == 0)
    storage.drop_instance_data(answer=True)
        