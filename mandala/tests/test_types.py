from .utils import *
from mandala.core.tps import is_subtype
from .test_context import setup_storage

def test_creation():
    # creating types manually. It is important later that these types are 
    # pairwise distinct
    types = [
        AnyType(),
        AtomType(int),
        AtomType(str),
        ListType(),
        ListType(elt_type=AtomType(float)),
        ListType(ListType(AtomType(int))),
        DictType(),
        DictType(value_type=AtomType(int)),
        DictType(value_type=ListType(elt_type=DictType(value_type=AtomType(annotation=int)))),
        UnionType(operands=[ListType(), AnyType()]),
        UnionType(operands=[AtomType(int), AtomType(float), AtomType(str)]),
    ]

    ### from annotations
    assert Type.from_annotation(annotation=typing.Any) == AnyType()
    assert Type.from_annotation(annotation=int) == AtomType(int)
    assert Type.from_annotation(list) == ListType()
    assert Type.from_annotation(annotation=TList[float]) == ListType(elt_type=AtomType(float))
    assert Type.from_annotation(dict) == DictType()
    assert Type.from_annotation(annotation=TDict[str, int]) == DictType(value_type=AtomType(annotation=int))
    assert Type.from_annotation(TUnion[list, typing.Any]) == UnionType([ListType(), AnyType()])
    
    ### specificity of type equality
    n_types = len(types)
    for i, j in itertools.product(range(n_types), range(n_types)):
        if i != j:
            assert types[i] != types[j]
    
    ### things that must fail
    try:
        _ = UnionType(operands=[])
        assert False
    except Exception as e:
        assert True

    try:
        _ = UnionType(operands=[AnyType()])
        assert False
    except Exception as e:
        assert True

    try:
        _ = UnionType(operands=[AtomType(int), AtomType(int)])
        assert False
    except Exception as e:
        assert True
        
def test_subtype():
    # creating types manually. It is important later that these types are 
    # pairwise distinct
    types = [
        AnyType(),
        AtomType(int),
        AtomType(str),
        ListType(),
        ListType(elt_type=AtomType(float)),
        ListType(ListType(AtomType(int))),
        DictType(),
        DictType(value_type=AtomType(int)),
        DictType(value_type=ListType(elt_type=DictType(value_type=AtomType(annotation=int)))),
        UnionType(operands=[ListType(), AnyType()]),
        UnionType(operands=[AtomType(int), AtomType(float), AtomType(str)]),
    ]

    # check that it works for all pairs
    for tp_1, tp_2 in itertools.product(types, types):
        is_subtype(s=tp_1, t=tp_2)
    
    # check tautological
    for tp in types:
        assert is_subtype(s=tp, t=tp)
    
    # check specific cases
    assert not is_subtype(AtomType(int), AtomType(str))

def test_wrapping(setup_storage):
    storage:Storage = setup_storage

    with context(storage=storage) as c:
        Int = Var(annotation=int, name='Int')
        IntList = Var(annotation=TList[Int], name='IntList')

    with context(storage=storage) as c:
        c_copy = c.spawn()
    
    x = Int(23, __context__=c_copy)
    y = IntList([1, 2, 3], __context__=c_copy)