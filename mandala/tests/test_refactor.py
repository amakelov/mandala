from .test_context import setup_storage
from .funcs import *
from .utils import *

def test_refactorings(setup_storage):
    storage:Storage = setup_storage
    
    with run(storage):
        Int = Var(annotation=int)
        Str = Var(annotation=str)

        ########################### 
        ### valid input type change
        ###########################
        @op()
        def f_1(x:Int) -> Int:
            return 23 
        @op()
        def f_1(x:TUnion[Int, Str]) -> Int:
            return 23 
        x = f_1(x='a')
        assert unwrap(x) == 23

        ########################### 
        ### valid output type change
        ###########################
        @op()
        def f_1_1(x:Int) -> Int:
            return 23 
        @op()
        def f_1_1(x:Int) -> TUnion[Int, Str]:
            return 23 
        x = f_1_1(x=23)
        assert unwrap(x) == 23
    
        ############################## 
        ### invalid output type change
        ############################## 
        @op()
        def f_1_2(x:int) -> TAny:
            return 23
        try:
            @op()
            def f_1_2(x:int) -> int:
                return 23
            assert False
        except SynchronizationError:
            assert True

    #############################
    ### creating an input
    #############################
    with define(storage):
        @op()
        def f_4(x:Int) -> Int:
            return 23 
    with define(storage):
        @op()
        def f_4(x:Int, y:Int=CompatArg(default=1)) -> Int:
            return 23
    with run(storage):
        x = f_4(x=23, y=42)
        assert unwrap(x) == 23
    storage.drop_instance_data(answer=True)
        
    ########################################
    ### creating a backward-compatible input
    ########################################
    with define(storage) as c:
        @op()
        def f_5(x:Int) -> Int:
            return x 
    with run(storage) as c:
        res = [f_5(x) for x in range(10)]
        c.commit()

    ### check that we preserve old results and compute with the new function
    with define(storage) as c:
        @op()
        def f_5(x:Int, y:Int=CompatArg(default=0)) -> Int:
            return x + y
        
    # check that not providing the argument causes no recomputation
    with run(storage, allow_calls=False) as c:
        old_res = [f_5(x) for x in range(10)]
    
    # check that passing the default value explicitly causes no recomputation
    # either 
    with run(storage, allow_calls=False, autocommit=True) as c:
        old_res = [f_5(x, 0) for x in range(10)]
    
    # run with new functionality
    with run(storage, autocommit=True) as c:
        new_res = [f_5(x, x) for x in range(10)]
    
    ### check that we preserve old values in queries but can restrict to old/new/all
    with query(storage) as c:
        x = Int().named('x')
        res = f_5(x=x).named('res')
        df = c.qeval(x, res)
        assert df.shape[0] == 19
    with query(storage) as c:
        x = Int().named('x')
        y = Int().named('y')
        res = f_5(x=x, y=y).named('res')
        df = c.qeval(x, y, res)
        assert df.shape[0] == 19
    with query(storage) as c:
        x = Int().named('x')
        y = Int().named('y').equals(0)
        res = f_5(x=x, y=y).named('res')
        df = c.qeval(x, y, res)
        assert df.shape[0] == 10
    with query(storage) as c:
        x = Int().named('x')
        y = Int().named('y').where(lambda x: x != 0)
        res = f_5(x=x, y=y).named('res')
        df = c.qeval(x, y, res)
        assert df.shape[0] == 9
    storage.drop_instance_data(answer=True)


def test_update_bug():
    storage = Storage()

    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    with run(storage):
        f(23)
        
    @op(storage)
    def f(x:int, y:int=CompatArg(default=1)) -> int:
        return x + y
    
    f.get_table()
    
    

def test_changes_before_commit():
    storage = Storage()

    ############################################################################ 
    ### rename op while there are uncommitted calls
    ############################################################################ 
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    with run(storage):
        [f(i) for i in range(5)]
    
    storage.rename_func(func_ui=f, new_name='g')

    @op(storage)
    def g(x:int) -> int:
        return x + 1 
    
    with run(storage, autocommit=True):
        pass
    assert len(storage.rel_adapter.get_op_rel(op=g.op)) == 5
    storage.drop_instance_data(answer=True)
    storage.drop_func(g)

    ############################################################################ 
    ### rename argument while there are uncommitted calls
    ############################################################################ 
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    with run(storage):
        [f(i) for i in range(5)]
    storage.rename_args(func_ui=f, mapping={'x': 'y'})

    @op(storage)
    def f(y:int) -> int:
        return y + 1 
    
    with run(storage, autocommit=True):
        pass
    assert len(storage.rel_adapter.get_op_rel(op=f.op)) == 5
    storage.drop_instance_data(answer=True)
    storage.drop_func(f)

    ############################################################################ 
    ### add argument while there are uncommitted calls
    ############################################################################ 
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    with run(storage):
        [f(i) for i in range(5)]
    
    @op(storage)
    def f(x:int, y:int=CompatArg(1)) -> int:
        return x + y
    
    with run(storage, autocommit=True):
        pass
    assert len(storage.rel_adapter.get_op_rel(f.op)) == 5