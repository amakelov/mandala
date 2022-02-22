from .utils import *
from ..util.common_ut import concat_lists
from .test_context import setup_storage
from ..session import get_scratch_dir

def generate_structs(lists_only:bool=False):
    scalars = [23, 'what']
    lists = [
            [],
            [1, 2, 3],
            [1, 2, [3, 4]], 
    ]
    type_dict = {
        int: AtomType(annotation=int),
        str: AtomType(annotation=str),
        list: ListType(),
    }
    objs = lists if lists_only else scalars + lists
    return objs, type_dict

def check_type_match(obj:TAny, vref:ValueRef) -> bool:
    if isinstance(obj, list):
        return isinstance(vref, ListRef)
    if isinstance(obj, (int, str)):
        return isinstance(vref, AtomRef)
    raise NotImplementedError()

def test_detach():
    a = AtomRef(obj=23, in_memory=True, uid=get_uid(), 
                persistable=True, tp=AtomType(annotation=int), 
                delayed_storage=True)
    b = a.detached()
    assert all(b.__dict__[k] == v for k, v in a.__dict__.items() 
               if k not in ('_obj', '_in_memory'))
    a = AtomRef(obj=None, in_memory=False, uid=get_uid(), 
                persistable=True, tp=AtomType(annotation=int), 
                delayed_storage=True)
    b = a.attached(obj=23)
    assert all(b.__dict__[k] == v for k, v in a.__dict__.items() 
               if k not in ('_obj', '_in_memory'))

def test_simple():
    objs, type_dict = generate_structs()
    for obj in objs:
        vref = wrap_structure(obj=obj, type_dict=type_dict)
        assert vref.unwrap() == obj
        assert check_type_match(obj=obj, vref=vref)
            
def test_indexing():
    objs, type_dict = generate_structs(lists_only=True)
    for obj in objs:
        vref:ListRef = wrap_structure(obj=obj, type_dict=type_dict)
        for i, elt in enumerate(obj):
            assert vref[i].unwrap() == elt

def test_generative():
    va = ValAdapter(obj_storage=ObjStorage(root=get_scratch_dir()))
    a = ValueGenerator()
    a.populate(iterations=20)
    type_dict = {
        int: AtomType(annotation=int),
        str: AtomType(annotation=str),
        np.ndarray: AtomType(annotation=np.ndarray),
        list: ListType(),
        dict: DictType()
    }
    for elt in a.pool:
        vref:ValueRef = wrap_detached(obj=elt, reference=None, type_dict=type_dict,
                    through_collections=True)
        loc = va.set(vref=vref)
        recovered = va.get(loc=loc).unwrap()
        assert eq_objs(x=elt, y=recovered)

def test_kv(setup_storage):
    """
    Test different KVStore implementations
    """
    storage:Storage = setup_storage
    
    with context(storage=storage):
        Int = Var(annotation=int, name='Int', kv=SQLiteStorage())
        Dict = Var(annotation=dict, name='Dict', kv=JoblibStorage())

    va = storage.val_adapter
    objs = [i for i in range(100)]
    vals = [wrap_detached(obj=i, reference=Int.tp) for i in objs]
    locs = va.mset(vals)
    recovered = [x.unwrap() for x in va.mget(locs=locs)]
    assert recovered == objs
    storage.drop_instance_data(answer=True)

def test_wrap(setup_storage):
    storage:Storage = setup_storage
    
    with context(storage=storage):
        Int = Var(annotation=int, name='Int')
        IntList = Var(annotation=TList[Int], name='IntList')
        IntListList = Var(annotation=TList[IntList], name='IntListList')
        FakeInt = Var(annotation=int, name='FakeInt')
        FakeIntList = Var(annotation=TList[FakeInt], name='FakeIntList')
        Dict = Var(annotation=dict, name='Dict')
        IntDict = Var(annotation=TDict[str, Int])
        IntDictList = Var(annotation=TList[IntDict])
        Any = Var(annotation=TAny, name='Any')

    ### check that the things that must work work
    matched_sets = [
        (Int, [
            23
            ]),
        (IntList, [
            [], 
            [1, 2, 3],
            ]),
        (IntListList, [
            [],
            [[]],
            [[], [1, 2, 3], [23]],
        ]),
        (Dict, [
            {},
            {'a': 23, 'b': [1, 2, 3]},
        ]),
        (IntDict, [
            {},
            {'a': 23, 'b': 42}, 
        ]),
        (IntDictList, [
            [],
            [{}],
            [{'a': 23, 'b': 42}, {}]
            ]),
    ]
    with run(storage=storage):
        for wrapper, values in matched_sets:
            for value in values:
                wrapped = wrapper(value)
    storage.drop_instance_data(answer=True)

    ### check that the resulting type of fully unwrapped values matches the wrapper's type
    with run(storage=storage):
        values = concat_lists([values for _, values in matched_sets])
        wrappers = [Int, IntList, IntListList, Dict, IntDict, IntDictList, Any]
        # see what works
        res = [] 
        for v in values:
            for wrapper in wrappers:
                try: 
                    w = wrapper(v)
                    res.append((v, wrapper, w))
                except:
                    pass
        for _, wrapper, w in res:
            assert w.get_type() == wrapper.tp
    storage.drop_instance_data(answer=True)
    
    ### check that things that shouldn't happen don't
    mismatched_pairs = [(Int, [1, 2, 3]), 
                        (IntList, 23), 
                        (Dict, 42),
                        (Dict, []),
                        (IntDict, {'a': 'x', 'b': 'y'}),
    ]
    with run(storage=storage):
        for wrapper, v in mismatched_pairs:
            try:
                _ = wrapper(v)
                assert False
            except:
                assert True
    storage.drop_instance_data(answer=True)
    
    ### check storage recovery 
    with run(storage=storage, partition='test') as c:
        val_and_wrapped_pairs = []
        for wrapper, values in matched_sets:
            for value in values: 
                wrapped = wrapper(value)
                val_and_wrapped_pairs.append((value, wrapped))
        c.commit()
        storage.verify_static()
    with run(storage=storage):
        for val, wrapped in val_and_wrapped_pairs:
            loc = c.where_is(vref=wrapped)
            recovered = storage.val_adapter.get(loc=loc)
            assert recovered.unwrap() == val
    storage.drop_instance_data(answer=True)

    ### check lazy invariants
    matched_locs = []
    with run(storage=storage, partition='test') as c:
        for wrapper, values in matched_sets:
            locs = []
            for value in values: 
                wrapped = wrapper(value)
                loc = c.where_is(wrapped)
                locs.append(loc)
            matched_locs.append(locs)
        c.commit()
    
    with run(storage=storage) as c:
        for locs in matched_locs:
            for loc in locs:
                vref = storage.val_adapter.get(loc=loc, lazy=True)
                assert not vref.in_memory
                c.attach(vref=vref)
                assert vref.in_memory
                vref = storage.val_adapter.get(loc=loc, lazy=False)
                assert is_deeply_in_memory(vref=vref)
    storage.drop_instance_data(answer=True)

def test_uids(setup_storage:Storage):
    
    storage = setup_storage
    with context(storage=storage):
        Int = Var(annotation=int, name='Int', kv=SQLiteStorage())
        AnotherInt = Var(annotation=int, name='AnotherInt')
        Dict = Var(annotation=dict, name='Dict', kv=JoblibStorage())
        IntDict = Var(annotation=TDict[str, int])
        StrList = Var(annotation=TList[str])
    
    with transient(storage=storage):
        x = Int(23)
        y = AnotherInt(23)
        assert x.uid != y.uid
    storage.drop_instance_data(answer=True)

def test_no_overwriting(setup_storage):
    storage:Storage = setup_storage 
    with context(storage) as c:
        Int = Var(annotation=int)
        IntList = Var(annotation=TList[int])
    ### check .set()
    a = wrap_detached(obj=23, reference=Int.tp)
    va = storage.val_adapter
    loc = va.set(vref=a)
    # nobody should do this in reality, but need to make sure it doesn't break
    # anything
    a._obj = 42 
    loc = va.set(vref=a)
    b = va.get(loc=loc)
    assert b.obj() == 23
    storage.drop_instance_data(answer=True)

    ### check .mset()
    vrefs = [wrap_detached(obj=i, reference=Int.tp) for i in (1, 2, 3)]
    locs = va.mset(vrefs=vrefs)
    vrefs[0]._obj = 23
    locs = va.mset(vrefs=vrefs)
    recovered = va.mget(locs)
    assert recovered[0].obj() == 1