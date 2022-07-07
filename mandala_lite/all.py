from .core.model import ValueRef, Call, FuncOp
from .storages.main import Storage
from .ui.execution import op
from .ui.context import Context, run, query



def test():
    storage = Storage()
    
    @op(storage)
    def f(x) -> int:
        print('Hello world!')
        return x + 1 
    
    with run(storage=storage):
        a = f(23)
        
    