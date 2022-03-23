from .test_context import setup_storage
from .funcs import *
from .utils import *
from mandala.core.exceptions import SynchronizationError
from mandala.all import Skip, Mark

def test_definition_styles():
    storage = Storage()

    ### define multiple ops with a context
    with define(storage):
        @op()
        def f(x:int, y, z:str='stuff') -> TTuple[float, list]:
            return 0.0, []
        
        @op()
        def g() -> TAny:
            return 23
    
    with run(storage):
        f(23, 42)
        g()
    
    ### define and synchronize
    @op(storage)
    def h(a:int, b:int) -> TList[int]:
        return [a, b]

    ### redefine existing
    @op(storage=storage)
    def f(x:int, y, z:str='stuff') -> TTuple[float, list]:
        return 0.0, []
    with run(storage):
        f(23, 42)
        h(23, 42)
    storage.drop_instance_data(answer=True)

def test_defaults(setup_storage):
    storage:Storage = setup_storage

    with run(storage):
        @op()
        def f(x:int, y:int=42) -> int:
            return x + y 
        a = f(23)
        assert unwrap(a) == 65

def test_skip():
    storage = Storage()
    
    @op(storage)
    def f(x:int, log:Mark[bool, Skip]) -> TTuple[int, Mark[float, Skip], int]:
        if log:
            print('hi')
        return x, time.time(), x
    
    with run(storage):
        x, y, z = f(x=23, log=True)
        assert isinstance(y, AnnotatedObj) and y.is_final
        assert isinstance(x, ValueRef) and isinstance(z, ValueRef)
    
    with query(storage) as q:
        inp = Query()
        x, y, z = f(x=inp, log=True)
        df = q.get_table(x, z, inp)
    assert df.shape[0] == 1 
    assert df.values.tolist() == [[23, 23, 23]]

