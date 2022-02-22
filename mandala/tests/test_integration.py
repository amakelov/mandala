from .test_computation import setup_tests
from .conftest import *
from .utils import *

def test_patterns(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    with context(storage=storage):
        @op()
        def add_1(x:Array) -> Array:
            res = x + 1 
            return AsTransient(res)
        
        @op()
        def sub_1(x:Array) -> Array:
            res = x - 1
            return res
        
        @op()
        def add_range(x:Array, nums:IntList) -> ArrayList:
            return [AsTransient(x + num) for num in nums]

        @op()
        def array_mean(arrays:ArrayList) -> Array:
            return AsTransient(sum(arrays) / len(arrays))
            
    ############################################################################ 
    ### lazy + transient + buffer stuff 
    ############################################################################ 
    
    ### some computing and checking invariants
    fixed_array = np.random.uniform(size=(10, 20))
    buffer = storage.make_buffer()
    with run(storage=storage, buffer=buffer, partition='test') as c:
        x = Array(fixed_array)
        y = add_1(x=x)
        z = sub_1(x=y)
        assert y.in_memory
        assert not y.is_persistable
        transient_loc = c.where_is(y)
        c.commit()
    y_residue = storage.val_adapter.get(loc=transient_loc)
    assert not y_residue.in_memory

    ### lazy recovery
    with run(storage=storage, lazy=True) as c:
        x = Array(fixed_array)
        y = add_1(x=x)
        assert not y.in_memory
        z = sub_1(x=y)
        assert not z.in_memory
        c.attach(vref=z)
        assert z.in_memory
        c.attach(vref=y)
        assert not y.in_memory
    storage.drop_instance_data(answer=True)
    
    ### more computations 
    buffer = storage.make_buffer()
    fixed_arrays = [np.random.uniform(size=(10, 10)) for _ in range(10)]
    with run(storage=storage, buffer=buffer, partition='test') as c:
        arrays = [Array(x) for x in fixed_arrays]
        arrays_plus_1 = [add_1(x=x) for x in arrays]
        arrays_recovered = [sub_1(x=x) for x in arrays_plus_1]
        c.commit()
    
    ### query results 
    with query(storage=storage) as c:
        array = Array()
        array_plus_1 = add_1(x=array)
        recovered = sub_1(x=array_plus_1)
        df = c.qeval(array, recovered, names=['array', 'recovered'])
        for x, y in df.itertuples(index=False):
            assert eq_objs(x=x, y=y)
    storage.drop_instance_data(answer=True)

    ### chaining transient values
    # change operation to have transient output
    with context(storage=storage):
        @op()
        def sub_1(x:Array) -> Array:
            res = x - 1
            return AsTransient(res)

    with run(storage=storage, partition='temp') as c:
        a = Array(np.random.uniform(size=(100, 100)))
        x = add_1(x=a)
        y = sub_1(x=x)
        c.commit() 
    storage.drop_instance_data(answer=True)
    
    ### function returning multiple transient values
    with run(storage=storage, partition='temp') as c:
        x = Array(np.random.uniform(size=(100, 100)))
        things = add_range(x=x, nums=list(range(10)))
        inc_things = [add_1(x=thing) for thing in things]
        final_mean = array_mean(inc_things)
        c.commit()
    storage.drop_instance_data(answer=True)