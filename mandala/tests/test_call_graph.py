from .utils import *
from .funcs import *

def test_unit():
    storage = Storage()
    
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    @superop(storage)
    def f_twice(x:int) -> int:
        return f(f(x))
    
    with run(storage, autocommit=True):
        f_twice(42)

    cg = storage.call_graph_st
    nodes = cg.get_nodes()
    assert nodes == [f_twice.op.qualified_name, f.op.qualified_name]
    assert cg.get_neighbors(node=nodes[0]) == [f.op.qualified_name]
    assert cg.get_callers(node=f.op.qualified_name) == [f_twice.op.qualified_name]

    ### now, check that we detect invalidation of previous version of calling superop
    @op(storage, version='1')
    def f(x:int) -> int:
        return x - 1 

    # this should not work
    try:
        @superop(storage)
        def f_twice(x:int) -> int:
            return f(f(x))
        assert False
    except SynchronizationError:
        assert True
    except:
        assert False
    
    # this should work
    try:
        @superop(storage, version='1')
        def f_twice(x:int) -> int:
            return f(f(x))
        assert True
    except SynchronizationError:
        assert False
    except:
        assert False