from .utils import *
from .funcs import *
from .conftest import _setup_tests, setup_tests

class AssertNoOpChange(object):
    """
    Context manager to verify that the stored representation of signatures and
    the connected operations are not changed by invalid actions 
    """
    def __init__(self, storage:Storage, compare_connected:bool=True):
        self.storage = storage
        self.compare_connected = compare_connected
        self.op_adapter_before = None
        self.op_adapter_after = None
        self.connected_before = None
        self.connected_after = None
        
    def __enter__(self):
        self.op_adapter_before = self.storage.op_adapter._dump_state()
        self.connected_before = self.storage.synchronizer._dump_state()
    
    def __exit__(self, tp, value, traceback):
        self.op_adapter_after = self.storage.op_adapter._dump_state()
        self.connected_after = self.storage.synchronizer._dump_state()
        assert self.op_adapter_before == self.op_adapter_after
        if self.compare_connected:
            assert self.connected_before == self.connected_after


assert_no_op_change = AssertNoOpChange

def test_invalid_changes():
    storage = Storage()

    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    repr(f), str(f)
    
    ### rename input without declaring this
    with assert_no_op_change(storage):
        try:
            @op(storage)
            def f(y:int) -> int:
                return y + 1
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False
    
    ### incompatible input type change
    with assert_no_op_change(storage):
        try:
            @op(storage)
            def f(x:str) -> int:
                return int(x)
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False
    
    ### incompatible output type change
    with assert_no_op_change(storage):
        try:
            @op(storage)
            def f(x:int) -> str:
                return int(x)
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False
    
    ### change number of outputs
    with assert_no_op_change(storage):
        try:
            @op(storage)
            def f(x:int) -> TTuple[str, int]:
                return int(x)
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False
    
    ### reorder potentially positional arguments
    @op(storage)
    def g(x:int, y:int) -> int:
        return x + y 

    with assert_no_op_change(storage):
        try:
            @op(storage)
            def g(y:int, x:int) -> int:
                return x + y 
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False

def test_synchronizer():
    """
    Check that various updates go through / don't go through as they should
    """
    storage = Storage()

    ############################################################################  
    ### test invariant at creation/update/drop
    ############################################################################  
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    assert storage.is_connected(f=f.op)
    assert storage.synchronizer.connected_names == ['f']

    ### test re-creating op with same value
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    assert storage.is_connected(f=f.op)
    assert storage.synchronizer.connected_names == ['f']

    ### drop and verify it's all gone
    storage.drop_func(f)
    assert not storage.is_connected(f)
    assert len(storage.synchronizer.connected_names) == 0
    assert len(storage.stored_signatures) == 0
    try:
        with run(storage):
            f(23)
        assert False
    except SynchronizationError:
        assert True
    except Exception as e:
        assert False

    ### create it again 
    @op(storage)
    def f(x:int) -> int:
        return x + 1 

    ############################################################################  
    ### test rename op
    ############################################################################  
    # test f is properly invalidated, but its signature remains
    orig_sig = f.op.orig_sig
    storage.rename_func(f, new_name='g')
    assert f.is_invalidated
    repr(f), str(f) # print invalidation message
    assert not storage.is_connected(f)
    try:
        with run(storage):
            f(23)
        assert False
    except SynchronizationError:
        assert True
    except Exception as e:
        assert False
    assert not storage.synchronizer.connected_names
    assert storage.stored_signatures['g', '0'] == orig_sig

    # test re-connecting renamed op
    @op(storage)
    def g(x:int) -> int:
        return x + 1 
    assert storage.is_connected(g)
    with run(storage, autocommit=True):
        g(23)
    
    # test renaming to already existing name fails
    @op(storage)
    def f(x:int) -> int:
        return x - 1
    
    assert storage.is_connected(f)
    assert storage.is_connected(g)
    with assert_no_op_change(storage):
        try:
            storage.rename_func(g, new_name='f')
            assert False
        except SynchronizationError:
            assert True
        except Exception as e:
            assert False
    assert storage.is_connected(f)
    assert storage.is_connected(g)

    # reset storage state
    storage.drop_func(f)
    storage.drop_func(g)

    ############################################################################ 
    ### test rename args
    ############################################################################ 
    @op(storage)
    def f(x:int, y:int) -> int:
        return x + y
    
    orig_sig = f.op.orig_sig
    storage.rename_args(func_ui=f, mapping={'x': 'z', 'y': 'w'})
    # check invalidation
    assert f.is_invalidated
    repr(f), str(f) # print invalidation message
    assert not storage.is_connected(f)
    try:
        with run(storage):
            f(23, 42)
        assert False
    except SynchronizationError:
        assert True
    except Exception as e:
        assert False
    assert not storage.synchronizer.connected_names
    new_sig:BaseSignature = storage.stored_signatures['f', '0']
    assert new_sig != orig_sig
    assert new_sig.poskw_names == ['z', 'w']

    # connect updated version
    @op(storage)
    def f(z:int, w:int) -> int:
        return z + w
    
    with run(storage, autocommit=True):
        f(z=23, w=42)
    storage.drop_instance_data(answer=True)