from .test_context import setup_storage
from .funcs import *
from .utils import *
from mandala.core.exceptions import SynchronizationError

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