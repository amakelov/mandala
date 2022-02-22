from .utils import *

from .funcs import *
from .conftest import setup_tests

def test_simple(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    ############################################################################ 
    ### unit
    ############################################################################ 
    ### do some work
    with run(storage=storage) as c:
        x = add(23, 42)
        c.commit()
    ### delete the work
    with delete(storage=storage) as c:
        x = add(23, 42)
        c.commit_deletions()
    ### check if things got deleted
    df = storage.rel_adapter.get_op_vrefs(op=add.op, rename=True)
    assert df.empty
    assert not storage.call_st.locs()
    ### do computation again
    with run(storage=storage) as c:
        x = add(23, 42)
        c.commit()
    with query(storage=storage) as c:
        x, y = Any(), Any()
        z = add(x, y)
        df = c.qeval(x, y, z, names=['x', 'y', 'z'])
        assert {tuple(elt) for elt
                in df.itertuples(index=False)} == {(23, 42, 65)}
    storage.drop_instance_data(answer=True)
    
    ############################################################################ 
    ### deleting multiple calls at once
    ############################################################################ 
    with run(storage=storage) as c:
        things = []
        means = []
        for i in range(10):
            thing = inc(i)
            things.append(thing)
            for j in range(10):
                things.append(add(thing, j))
            cur_mean = mean(things)
            means.append(cur_mean)
        final = mean(means)
        c.commit()
    with delete(storage=storage) as c:
        things = []
        means = []
        for i in range(10):
            thing = inc(i)
            things.append(thing)
            for j in range(10):
                things.append(add(thing, j))
            cur_mean = mean(things)
            means.append(cur_mean)
        final = mean(means)
        c.commit_deletions()
    for func in (inc, add, mean):
        df = storage.rel_adapter.get_op_vrefs(op=func.op, rename=True)
        assert df.empty
    assert not storage.call_st.locs()
    storage.drop_instance_data(answer=True)
        
    ############################################################################ 
    ### deleting some things only
    ############################################################################ 
    # run a workflow of several parts
    with run(storage=storage) as c:
        nums = range(10)
        incs = [inc(x) for x in nums]
        final = mean(x=incs)
        c.commit()
    # delete only latter part
    with run(storage=storage) as c:
        nums = range(10)
        incs = [inc(x) for x in nums]
        with c(mode=MODES.delete) as d:
            final = mean(x=incs)
            d.commit_deletions()
    # check it got deleted but earlier things didn't
    df = storage.rel_adapter.get_op_vrefs(op=mean.op, rename=True)
    assert df.empty
    df =  storage.rel_adapter.get_op_vrefs(op=inc.op, rename=True)
    assert df.shape[0] == 10
    storage.call_st.locs()
    storage.drop_instance_data(answer=True)

    ############################################################################ 
    ### deleting calls only, verifying vrefs remain orphaned
    ############################################################################ 
    with run(storage) as c:
        nums = range(10)
        incs = [inc(x) for x in nums]
        final = mean(x=incs)
        c.commit()
    inc_locs = [storage.where_is(vref=x) for x in incs]
    assert not any(storage.rel_adapter.mis_orphan(locs=inc_locs))
    final_loc = storage.where_is(vref=final)
    assert not storage.rel_adapter.mis_orphan(locs=[final_loc])[0]
    with delete(storage, autodelete=False) as c:
        nums = range(10)
        incs = [inc(x) for x in nums]
        final = mean(x=incs)
        c.commit_deletions()
    assert all(storage.rel_adapter.mis_orphan(locs=inc_locs))
    assert storage.rel_adapter.mis_orphan(locs=[final_loc])[0]
    storage.drop_instance_data(answer=True)
        
    ############################################################################ 
    ### deleting with a superop
    ############################################################################ 
    with run(storage, autocommit=True):
        add_three(x=23, y=42, z=5)
    
    with delete(storage, autodelete=True):
        add_three(x=23, y=42, z=5)
    assert not storage.call_st.locs()
    storage.drop_instance_data(answer=True) 

def test_superops():
    storage = Storage()
    
    @op(storage)
    def get_divisors(num:int) -> TList[int]:
        return [x for x in range(1, num) if num % x == 0]
    
    @superop(storage)
    def concat_divisors(nums:TList[int]) -> TList[int]:
        divisors_list = [get_divisors(num) for num in nums]
        return [elt for divs in divisors_list for elt in divs]
    
    @op(storage)
    def inc(x:int) -> int:
        return x + 1 
    
    @superop(storage)
    def inc_by_chunk(chunk:TList[int]) -> TList[int]:
        return [inc(x) for x in chunk]
    
    with run(storage, autocommit=True):
        nums = list(range(20))
        concat_divisors(nums=nums)
    
    with delete(storage, autodelete=True):
        nums = list(range(20))
        concat_divisors(nums=nums)
    assert len(storage.call_st.locs()) == 0
    
def test_bug():
    storage = Storage()
    
    @op(storage)
    def get_divisors(num:int) -> TList[int]:
        return [x for x in range(1, num) if num % x == 0]
    
    @superop(storage)
    def f(lst:TList[int]) -> int:
        return lst[0]
    
    with run(storage, autocommit=True):
        lst = get_divisors(100)
        f(lst)
    with run(storage):
        lst = get_divisors(100)
        with delete(autodelete=True):
            f(lst)
    assert f.get_table().empty
    storage.drop_instance_data(answer=True)

def test_drop_op():
    """
    Tests for deleting operations are isolated to prevent schema changes across tests
    """
    storage = Storage()
    
    @op(storage)
    def inc(x:int) -> int:
        return x + 1 
    
    @op(storage)
    def add(x:int, y:int) -> int:
        return x + y 

    ### drop empty op
    storage.drop_func(f=add)
    assert not storage.op_adapter.has_op(ui_name='add', version='0')
    with run(storage, autocommit=True):
        for i in range(10):
            inc(i)
    ### drop op with results
    storage.drop_func(f=inc)
    assert not storage.op_adapter.has_op(ui_name='inc', version='0')
    # cleanup
    storage.drop_instance_data(answer=True)

def test_drop_uncommitted(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    ### unit 
    with run(storage, autocommit=False):
        for i in range(10):
            inc(i)
    assert len(storage.call_st.locs()) == 10
    storage.drop_uncommitted_calls()
    assert len(storage.call_st.locs()) == 0
    storage.drop_instance_data(answer=True)

    ### after committed work
    with run(storage, autocommit=False) as c:
        for i in range(10):
            inc(i)
        c.commit()
    with run(storage, autocommit=False) as c:
        for i in range(10, 20):
            inc(i)
    assert len(storage.call_st.locs()) == 20
    storage.drop_uncommitted_calls()
    assert len(storage.call_st.locs()) == 10
    storage.drop_instance_data(answer=True)

    ### test isolation of commits between partitions
    with run(storage, autocommit=False, partition='first') as c:
        for i in range(10):
            inc(i)
        c.commit()
    with run(storage, autocommit=False, partition='second') as c:
        for i in range(10, 20):
            inc(i)
    with run(storage, autocommit=False, partition='third') as c:
        for i in range(20, 30):
            inc(i)
        c.commit()
    assert len(storage.call_st.locs()) == 30
    storage.drop_uncommitted_calls()
    assert len(storage.call_st.locs()) == 20
    storage.drop_instance_data(answer=True)