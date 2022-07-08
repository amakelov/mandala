from .common_imports import *
from .core.model import ValueRef, Call, FuncOp, unwrap, wrap
from .core.config import Config
from .storages.main import Storage
from .ui.execution import op
from .ui.context import Context, run, query



def test():
    storage = Storage()
    
    @op(storage)
    def inc(x) -> int:
        print('Hi from inc')
        return x + 1 

    @op(storage)
    def add(x:int, y:int) -> int:
        print('Hi from add!')
        return x + y
    
    with run(storage=storage):
        a = 23
        b = inc(23)
        c = add(a, b)

        
    