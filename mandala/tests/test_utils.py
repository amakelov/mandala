from .utils import *
from ..util.common_ut import *
import random

def test_recursive_collec():
    vg = ValueGenerator()
    vg.populate(iterations=30)
    for collec in vg.pool:
        assert collec_same_shape(collec, collec)
        flat, index = flatten_collection_atoms(collection=collec, start_idx=0)
        assert collec_same_shape(collec_1=collec, collec_2=index)
        assert unflatten_atoms_like(atoms=flat, index_reference=index) == collec

def test_vref_funcs():
    # test unwrap, detached, ...

    ### generate some values 
    storage = Storage()
    Int = AtomType(int)

    def make_int() -> AtomRef:
        return wrap_detached(obj=random.randint(0, 100), reference=Int)
    
    def make_list() -> ListRef:
        return ListRef(obj=[make_int() for _ in range(10)], tp=ListType(Int))
    
    def make_dict() -> DictRef:
        return DictRef(obj={get_uid(): make_int()}, tp=DictType(Int))
        
    ############################################################################ 
    ### test unwrap
    ############################################################################ 

    # on vrefs
    x = make_int()
    assert unwrap(x) == x.obj()
    x = make_list()
    assert unwrap(x) == [elt.obj() for elt in x]
    x = make_dict()
    assert unwrap(x) == {k: v.obj() for k, v in x.items()}

    # on structs
    x = [make_int() for i in range(10)]
    assert unwrap(x, recursive=True) == [elt.obj() for elt in x]
    x = tuple([make_int() for i in range(10)])
    assert unwrap(x, recursive=True) == tuple([elt.obj() for elt in x])
    x = {get_uid(): make_int() for i in range(10)}
    assert unwrap(x, recursive=True) == {k: v.obj() for k, v in x.items()}
    x = np.array([make_int() for i in range(10)])
    assert eq_objs(unwrap(x), np.array([elt.obj() for elt in x]))
    x = pd.Series(data=[make_int() for i in range(10)], 
                  index=pd.Index([make_int() for i in range(10)]))
    unwrapped_x = unwrap(x)
    assert eq_objs(unwrapped_x.index.values.tolist(), [elt.obj() for elt in x.index])
    assert eq_objs(unwrapped_x.values.tolist(), [elt.obj() for elt in x.values])
    x = pd.DataFrame({'a': [make_int() for i in range(10)],
                      'b': [make_int() for i in range(10)]},
                     index=[make_int() for i in range(10)])
    unwrapped_x = unwrap(x)
    assert eq_objs(unwrapped_x.index.values.tolist(), [elt.obj() for elt in x.index])
    assert eq_objs(unwrapped_x['a'].values.tolist(), [elt.obj() for elt in x['a'].values])
    assert eq_objs(unwrapped_x['b'].values.tolist(), [elt.obj() for elt in x['b'].values])

    # on custom objects
    class C(object):
        pass
    x = C()
    assert unwrap(x) is x

    ############################################################################ 
    ### test detached
    ############################################################################ 
    # on vrefs
    x = make_int()
    assert detached(x).in_memory == False
    assert detached(x).uid == x.uid
    assert x.in_memory

def test_collection_helpers():
    ### extract_uniques
    def check_extract_uniques(objs, keys):
        representatives, index, projection = extract_uniques(objs=objs, 
                                                             keys=keys)
        for i, idx in enumerate(index):
            assert objs[idx] == representatives[i]
        assert len(set(keys)) == len(representatives)
        fibers = {j: [x for x in projection.keys() if projection[x] == j]
                  for j in projection.values()}
        for j, fiber in fibers.items():
            assert len(set([keys[i] for i in fiber])) == 1

    check_extract_uniques(objs=[], keys=[])
    try:
        check_extract_uniques(objs=[1, 2], keys=[1, 2, 3])
        assert False
    except:
        assert True
    check_extract_uniques(objs=[1, 2, 3, 4, 5], keys=[1, 1, 2, 2, 3])

    class C(object):
        def __init__(self, x:int):
            self.x = x
    objs = [C(1), C(3), C(2), C(1), C(2)]
    keys = [obj.x for obj in objs]
    check_extract_uniques(objs=objs, keys=keys)

    # randomized testing 
    sampling_range = range(5)
    for trial in range(10):
        sample = np.random.choice(sampling_range, replace=True, size=10)
        objs = [C(int(x)) for x in sample]
        keys = [obj.x for obj in objs]
        check_extract_uniques(objs=objs, keys=keys)

    ### chunk list
    def verify_chunk_by_num(lst:TList[TAny], num_chunks:int):
        parts = chunk_list(lst=lst, num_chunks=num_chunks)
        assert len(parts) == num_chunks
        assert concat_lists(parts) == lst
    
    def verify_chunk_by_size(lst:TList[TAny], chunk_size:int):
        parts = chunk_list(lst=lst, chunk_size=chunk_size)
        assert concat_lists(parts) == lst
        assert len(set([len(x) for x in parts])) <= 2

    for num_chunks in [1, 2, 5, 10, 20, 100]:
        verify_chunk_by_num(lst=list(range(10)), num_chunks=num_chunks)
    for chunk_size in [1, 2, 5, 10, 20, 100]:
        verify_chunk_by_size(lst=list(range(10)), chunk_size=chunk_size)
    
def test_content_hashing():
    storage = Storage()

    with define(storage):
        Int = Var(annotation=int, hash_method='content')
        
        @op()
        def f(x:Int) -> Int:
            return x
        
        @op()
        def duplicate(x:Int, num_times:Int) -> TList[Int]:
            return [x for _ in range(num_times)]
    
    with run(storage) as c:
        ### unit test
        x = f(23)
        y = f(x)
        assert x.uid == y.uid
        ### test if collections observe this behavior
        copies = duplicate(x=23, num_times=42)
        assert all([elt.uid == y.uid for elt in copies])
    storage.drop_instance_data(answer=True)