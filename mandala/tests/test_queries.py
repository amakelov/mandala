from .funcs import *
from .conftest import _setup_tests, setup_tests
from .utils import *

REPRESENTATION = 'asymmetric' if CoreConfig.decompose_struct_as_many else 'symmetric'
INDEXING_STYLE = 'val_query' if (REPRESENTATION == 'asymmetric') else 'explicit'

def test_types(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    with run(storage) as c:
        for num in [1, 2, 3]:
            factors = get_prime_factors(x=num)
        c.commit()
        
    ### check that type constraints are parsed
    with query(storage) as q:
        num = Query(Int).named('num')
        df = q.get_table(num)
        # check atom type
        assert all(isinstance(value, int) for value in df['num'].values.tolist())
    with query(storage) as q:
        # check compound type: we can only guarantee that 
        lst = Query(IntList).named('lst')
        df = q.get_table(lst)
        assert  all(isinstance(value, list) for value in df['lst'].values.tolist())
    with query(storage) as q:
        # check union
        something = Query(TUnion[Int, IntList]).named('something').where(lambda x: isinstance(x, list))
        df = q.get_table(something)
    storage.drop_instance_data(answer=True)

def test_compound_destructive(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    ############################################################################ 
    ### test lists
    ############################################################################ 
    with run(storage, partition='test') as c:
        nums = [23, 42, 65]
        for num in nums:
            facts = get_prime_factors(x=num)
        c.commit()
    
    def parametric_query(idx_query:Int, c:Context):
        x = Int()
        facts = get_prime_factors(x=x)
        z = facts[idx_query]
        df = c.qeval(x, facts, z, 
                     names=['x', 'facts', 'z']).sort_values(by=['x', 'z'])
        return df

    ### generic index
    with query(storage) as c:
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery()
        else:
            idx_query = None
        df = parametric_query(idx_query=idx_query, c=c)
        truth = [(23, [23], 23),
                 (42, [2, 3, 7], 2),
                 (42, [2, 3, 7], 3),
                 (42, [2, 3, 7], 7),
                 (65, [5, 13], 5),
                 (65, [5, 13], 13)]
        assert [tuple(elt) for elt in df.itertuples(index=False)] == truth
        
    ### specific index present in all
    with query(storage) as c:
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery().equals(0)
        else:
            idx_query = 0
        df = parametric_query(idx_query=idx_query, c=c)
        assert ([tuple(elt) for elt in df.itertuples(index=False)] ==
                [(23, [23], 23), (42, [2, 3, 7], 2), (65, [5, 13], 5)])

    ### specific index present in some 
    with query(storage) as c:
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery().equals(1)
        else:
            idx_query = 1
        df = parametric_query(idx_query=idx_query, c=c)
        assert ([tuple(elt) for elt in df.itertuples(index=False)] ==
                [(42, [2, 3, 7], 3), (65, [5, 13], 13)])
        
    ### specific index present in none 
    with query(storage) as c:
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery().equals(23)
        else:
            idx_query = 23
        df = parametric_query(idx_query=idx_query, c=c)
        assert [tuple(elt) for elt in df.itertuples(index=False)] == []
        
    ### index range
    with query(storage) as c:
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery().isin([0, 1])
        else:
            idx_query = [0, 1]
        df = parametric_query(idx_query=idx_query, c=c)
        truth = [(23, [23], 23),
                 (42, [2, 3, 7], 2),
                 (42, [2, 3, 7], 3),
                 (65, [5, 13], 5),
                 (65, [5, 13], 13)]
        assert [tuple(elt) for elt in df.itertuples(index=False)] == truth
    storage.drop_instance_data(answer=True)
        
    ############################################################################ 
    ### test dicts
    ############################################################################ 
    with run(storage, partition='test') as c:
        nums = [23, 42, 65]
        for i, num in enumerate(nums):
            dct = get_some_metrics(x=num, y=nums[:i])
        c.commit()
    
    def parametric_query(key_query:Str, c:Context):
        num = Int()
        dct = get_some_metrics(x=num)
        a = dct[key_query]
        df = c.qeval(num, a, names=['num', 'val']).sort_values(by=['num', 'val'])
        return df

    with query(storage) as c:
        key_query = KeyQuery().equals('b')
        df = parametric_query(key_query=key_query, c=c)
        assert ([tuple(elt) for elt in df.itertuples(index=False)] ==
                [(23, 0.0), (42, 0.1), (65, 0.2)])

    with query(storage) as c:
        key_query = KeyQuery().equals('c')
        df = parametric_query(key_query=key_query, c=c)
        assert [tuple(elt) for elt in df.itertuples(index=False)] == []
    storage.drop_instance_data(answer=True)

def test_compound_constructive(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    ############################################################################ 
    ### test lists
    ############################################################################ 
    with run(storage, autocommit=True):
        result = int_mean([1, 2, 3])
    
    ### any index
    with query(storage) as q:
        x = Query().named('x')
        y = MakeList(containing=x).named('y')
        z = int_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 3
        assert set(df['x']) == {1, 2, 3}
    ### specific index
    with query(storage) as q:
        x = Query().named('x')
        y = MakeList(containing=x, at_index=1).named('y')
        z = int_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 1
        assert set(df['x']) == {2}
    ### set of indices
    with query(storage) as q:
        x = Query().named('x')
        y = MakeList(containing=x, at_indices=[0, 2]).named('y')
        z = int_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 2
        assert set(df['x']) == {1, 3}
    
    storage.drop_instance_data(answer=True)
    ############################################################################ 
    ### test dicts
    ############################################################################ 
    dict_value = {'a': 1, 'b': 2, 'c': 3}
    with run(storage, autocommit=True):
        result = dict_mean(x=dict_value)

    ### any key
    with query(storage) as q:
        x = Query().named('x')
        y = MakeDict(containing=x).named('y')
        z = dict_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 3
        assert all(df['y'] == dict_value)
        assert set(df['x']) == {1, 2, 3}
    ### specific key
    with query(storage) as q:
        x = Query().named('x')
        y = MakeDict(containing=x, at_key='b').named('y')
        z = dict_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 1
        assert all(df['y'] == dict_value)
        assert set(df['x']) == {2}
    ### set of keys
    with query(storage) as q:
        x = Query().named('x')
        y = MakeDict(containing=x, at_keys=['a', 'c']).named('y')
        z = dict_mean(y).named('z')
        df = q.get_table(x, y, z)
        assert df.shape[0] == 2
        assert all(df['y'] == dict_value)
        assert set(df['x']) == {1, 3}
    storage.drop_instance_data(answer=True)

def test_deletion(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    ############################################################################ 
    ### unit test 
    ############################################################################ 
    with run(storage) as c:
        add(23, 42)
        c.commit()
    # make sure the thing is there
    with query(storage) as q:
        x = Query(Any)
        y = Query(Any)
        z = add(x, y)
        df = q.get_table(x, y, z)
    assert df.shape[0] == 1
    # delete the thing
    with qdelete(storage, autodelete=True) as c:
        x = Query(Any)
        y = Query(Any)
        z = add(x, y)
    # check again
    with query(storage) as q:
        x = Query(Any)
        y = Query(Any)
        z = add(x, y)
        df = q.get_table(x, y, z)
    assert df.shape[0] == 0
    storage.drop_instance_data(answer=True)

    ############################################################################ 
    ### composition and constraints
    ############################################################################ 
    with run(storage) as c:
        for i in range(10):
            for j in range(10):
                z = add_int(i, j)
                w = inc(z)
                factors = get_prime_factors(w)
        c.commit()
    
    ### see what we have
    with query(storage) as q:
        x = Query(Int)
        y = Query(Int)
        z = add_int(x, y)
        w = inc(z)
        factors = get_prime_factors(w)
        df = q.get_table(x, y, z, w, factors)
    assert df.shape[0] == 100
    ### delete things
    with qdelete(storage, autodelete=True):
        x = Query(Int).isin([0, 1, 2, 3, 4, 5])
        y = Query(Int).where(lambda x: x % 2 == 0)
        z = add_int(x, y)
        w = inc(z)
        factors = get_prime_factors(w)
    ### see what's left
    with query(storage) as q:
        x = Query(Int)
        y = Query(Int)
        z = add_int(x, y)
        w = inc(z)
        factors = get_prime_factors(w)
        df = q.get_table(x, y, z, w, factors)
    for x, y, z, w, factors in df.itertuples(index=False):
        assert not ((x in [0, 1, 2, 3, 4, 5]) and (y % 2 == 0))
    storage.drop_instance_data(answer=True)
    
def test_branching(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    ### run some workflow that branches into different "analyses"
    with run(storage, autocommit=True):
        for x in range(3):
            for y in range(5):
                z = add_int(x, y)
                w = inc(z)
                if unwrap(y) < 3:
                    t = inc_and_dec(z)
    
    ### query all things together first: downstream constraints are imposed!
    with query(storage) as q:
        x, y = Query(int).named('x'), Query(int).named('y')
        z = add_int(x, y).named('z')
        w = inc(z).named('w')
        a, b = inc_and_dec(z)
        a.named('a'), b.named('b')
        df = q.get_table(x, y, z, w)
        tree = q.cur_tree
        repr(tree), str(tree)
    assert set(df['y'].values.tolist()) == set(range(3))
    ### now query with branching 
    with query(storage) as q:
        x, y = Query(int).named('x'), Query(int).named('y')
        z = add_int(x, y).named('z')
        tree = q.cur_tree
        tree.show()
        repr(tree), str(tree)
        with q.branch():
            w = inc(z).named('w')
            df_1 = q.get_table(x, y, z, w)
            tree = q.cur_tree
            tree.show()
            repr(tree), str(tree)
        assert set(df_1['y'].values.tolist()) == set(range(5))
        with q.branch():
            a, b = inc_and_dec(z)
            a.named('a'), b.named('b')
            df_2 = q.get_table(x, y, z, a, b)
        assert set(df_2['y'].values.tolist()) == set(range(3))
        t = q.cur_tree
        tree = q.cur_tree
        tree.show()
        repr(tree), str(tree)
    tree = q.cur_tree
    repr(tree), str(tree)
    tree.show()
    storage.drop_instance_data(answer=True)

    ### some things that often failed in the past
    with run(storage, autocommit=True):
        a = inc(23)
        b = inc(a)
    
    with query(storage) as q:
        a = inc()
        with q.branch():
            b = inc(a).named('b')
            df = q.get_table(b)
    assert df['b'].item() == 25
    storage.drop_instance_data(answer=True)

def test_magics(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    with run(storage, autocommit=True):
        for i in range(10):
            a = inc(i)
            b = add_int(x=a, y=i)
    
    with query(storage) as q:
        i = (Int() > 5).named('i')
        a = inc(i).named('a') < 10
        b = add_int(a, i).named('b')
        df = q.get_table(i, a, b)
        assert all(df['i'] > 5)
        assert df.shape[0] == 3
        assert all(df['b'] == df['a'] + df['i'])

def test_qfuncs(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    with run(storage, autocommit=True):
        x = inc(23)
        y, z = inc_and_dec(x)
        t = add(x, x)
        final_1 = int_mean([y, z])
        final_2 = int_mean([t])

    @qfunc()
    def f(use_add:bool=False):
        with query(storage) as q:
            a = Int().named('a')
            x = inc(a).named('x')
            if use_add:
                y, z = inc_and_dec(x)
                lst = MakeList(containing=y, at_index=0)
            else:
                t = add(x, x)
                lst = MakeList(containing=t, at_index=0)
            final = int_mean(lst).named('final')
            return q.get_table(a, x, final)
            
    with f:
        f(True)
        f(False)
        df = f.query()
    assert np.all(df.values == np.array([[23, 24, 24.0, True], 
                                        [23, 24, 48.0, False]], dtype=object))

    storage.drop_instance_data(answer=True)

def test_superops_and_structs():
    """
    Queries, superops and structs interact in some non-trivial ways
    """
    storage = Storage()

    @op(storage)
    def get_divisors(num:int) -> TList[int]:
        return [x for x in range(1, num) if num % x == 0]

    @op(storage)
    def inc(x:int) -> int:
        return x + 1 

    ############################################################################ 
    ### a standard aggregation pattern
    ############################################################################ 
    @op(storage)
    def sum_nums(nums:TList[int]) -> int:
        return sum(nums)
    
    with run(storage, autocommit=True):
        div_sums = []
        for num in range(10, 20):
            divs = get_divisors(num=num)
            div_sums.append(sum_nums(divs))
        final = sum_nums(div_sums)
    
    # trace only one number's path
    with query(storage) as q:
        num = Query(int).named('num')
        divs = get_divisors(num)
        div_sum = sum_nums(divs)
        div_sums = MakeList(containing=div_sum,
                            at_index=0) # note the index constraint
        final = sum_nums(div_sums).named('final')
        df = q.get_table(num, final)
    assert df.shape[0] == 1
    assert df['num'].item() == 10
    assert df['final'].item() == 83
    
    # trace all numbers' paths
    with query(storage) as q:
        num = Query(int).named('num')
        divs = get_divisors(num)
        div_sum = sum_nums(divs)
        div_sums = MakeList(containing=div_sum) # note the lack of index constraint
        final = sum_nums(div_sums).named('final')
        df = q.get_table(num, final)
    assert df.shape[0] == 10
    assert sorted(df['num'].values.tolist()) == list(range(10, 20))
    assert all(df['final'] == 83)
    storage.drop_instance_data(answer=True)

    ### an unusual superop
    @superop(storage)
    def get_max_len_divs(nums:TList[int]) -> TList[int]:
        all_divs = [get_divisors(num) for num in nums]
        lengths = [len(x) for x in all_divs]
        i = np.argmax(lengths)
        return all_divs[i]

    with run(storage, autocommit=True):
        max_len_divs = get_max_len_divs(list(range(10, 20)))
        for elt in max_len_divs:
            final = inc(elt)
    
    with query(storage) as q:
        nums = Query(annotation=TList[int])
        max_len_divs = get_max_len_divs(nums=nums)
        final = inc(max_len_divs[None])
        df = q.get_table(nums, max_len_divs, final)
    storage.drop_instance_data(answer=True)

    ### a superop that internally gets a list from an op and returns a prefix
    @superop(storage)
    def divisor_prefix(num:int, how_many:int) -> TList[int]:
        divisors = get_divisors(num)
        return divisors[:unwrap(how_many)]

    with run(storage, autocommit=True):
        for x in range(10, 20):
            result = divisor_prefix(num=x, how_many=3)

    # query the superop end-to-end
    with query(storage) as q:
        num, how_many = Query(int).named('num'), Query(int)
        divs_prefix = divisor_prefix(num, how_many)
        first_div = divs_prefix[0].named('first_div')
        df = q.get_table(num, first_div).sort_values(by='num')
    assert df['num'].values.tolist() == list(range(10, 20))
    assert all(df['first_div'] == 1)

    # query through the superop internals
    with query(storage) as q:
        num = Query(int).named('num')
        divisors = get_divisors(num)
        elt = divisors[0]
        divs_prefix = MakeList(containing=elt, at_index=0)
        first_div = divs_prefix[0].named('first_div')
        df = q.get_table(num, first_div).sort_values(by='num')
    if REPRESENTATION == 'asymmetric':
        assert df['num'].values.tolist() == list(range(10, 20))
        assert all(df['first_div'] == 1)
    else:
        assert set(df['num'].values.tolist()) == set(range(10, 20))
        assert all(df['first_div'] == 1)
    storage.drop_instance_data(answer=True)

    ### a superop that takes a list input, internally creates many lists and
    ### concatenates them
    @superop(storage)
    def concat_divisors(nums:TList[int]) -> TList[int]:
        divisors_list = [get_divisors(num) for num in nums]
        return [elt for divs in divisors_list for elt in divs]
    
    with run(storage, autocommit=True):
        all_divs = concat_divisors(nums=list(range(10, 13)))
    all_divs = [1, 2, 5, 1, 1, 2, 3, 4, 6]

    # query end to end
    with query(storage) as q:
        nums = Query(TList[int]).named('nums')
        all_divs_query = concat_divisors(nums).named('all_the_divs')
        df = q.get_table(nums, all_divs_query)
    assert df.shape[0] == 1
    assert df['nums'].item() == list(range(10, 13))
    assert df['all_the_divs'].item() == all_divs
    
    # query through the superop internals
    with query(storage) as q:
        nums = MakeList(containing=Query(int), at_index=0)
        if INDEXING_STYLE == 'val_query':
            idx_query = IndexQuery()
        else:
            idx_query = None
        num = nums[idx_query].named('num')
        divisors = get_divisors(num)
        all_divs = MakeList(containing=divisors[0])
        df = q.get_table(num, divisors.named('divisors'), 
                         all_divs.named('all_divs')).sort_values(by='num')
    if REPRESENTATION == 'asymmetric':
        # tests under asymmetric representation of structs
        assert df.shape[0] == 3
        assert df['num'].values.tolist() == list(range(10, 13))
        assert df['divisors'].values.tolist() == [[1, 2, 5], [1], [1, 2, 3, 4, 6]]
        all_divs_col = df['all_divs'].values.tolist() 
        assert all(all_divs_col[i] == all_divs for i in range(len(all_divs_col)))
    else:
        assert df.shape[0] == 6
    storage.drop_instance_data(answer=True)
    
    ### a superop processing data in chunks that can overlap
    @superop(storage)
    def inc_by_chunk(chunk:TList[int]) -> TList[int]:
        return [inc(x) for x in chunk]
    
    with run(storage, autocommit=True):
        nums = list(range(100))
        for prefix_size in (10, 20, 50, 100):
            results = inc_by_chunk(chunk=nums[:prefix_size])

    with query(storage) as q:
        chunk = Query(TList[int])
        chunk_result = inc_by_chunk(chunk)
        df = q.get_table(chunk, chunk_result)
    
    with query(storage) as q:
        chunk = Query(TList[int])
        elt = chunk[0]
        inc_elt = inc(elt)
        final = MakeList(containing=inc_elt, at_index=0)
        df = q.get_table(elt, final)
    storage.drop_instance_data(answer=True)

    ### a different way to express this
    @superop(storage)
    def inc_with_prefix_arg(all_nums:TList[int], prefix_size:int) -> TList[int]:
        return [inc(x) for x in all_nums[:unwrap(prefix_size)]]
    with run(storage, autocommit=True):
        nums = list(range(100))
        for prefix_size in (10, 20, 50, 100):
            results = inc_with_prefix_arg(all_nums=nums, prefix_size=prefix_size)
    with query(storage) as q:
        all_nums = Query(TList[int])
        elt = all_nums[0]
        inc_elt = inc(elt)
        final = MakeList(containing=inc_elt, at_index=0)
        df = q.get_table(elt, final)
    storage.drop_instance_data(answer=True)