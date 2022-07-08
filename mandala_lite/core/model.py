from ..common_imports import *
from .utils import Hashing
from .sig import Signature

class ValueRef:
    """
    Wraps objects with storage metadata (uid for now). 
    
    This is the object passed between memoized functions (ops).
    """
    def __init__(self, uid:str, obj:Any, in_memory:bool):
        self.uid = uid
        self.obj = obj
        self.in_memory = in_memory
    
    def __repr__(self) -> str:
        if self.in_memory:
            return f'ValueRef({self.obj}, uid={self.uid})' 
        else:
            return f'ValueRef(in_memory=False, uid={self.uid})'
    

def wrap(obj:Any, uid:Optional[str]=None) -> ValueRef:
    """
    Wraps a value as a `ValueRef` (with uid = content hash) if it isn't one
    already.
    """
    if isinstance(obj, ValueRef):
        return obj
    else:
        uid = Hashing.get_content_hash(obj) if uid is None else uid
        return ValueRef(uid=uid, obj=obj, in_memory=True)


T = TypeVar('T')
def unwrap(obj:Union[T, ValueRef]) -> T:
    if not isinstance(obj, ValueRef):
        return obj
    else:
        return obj.obj


class Call:
    """
    Represents the inputs, outputs and uid of a call to an operation. 
    
    The inputs to an operation are represented as a dictionary, and the outputs
    are a (possibly empty) list, mirroring how Python has named inputs but
    nameless outputs for functions. This convention is followed throughout.
    """
    def __init__(self, uid:str, inputs:Dict[str, ValueRef], 
                 outputs:List[ValueRef], op:'FuncOp'):
        self.uid = uid
        self.inputs = inputs
        self.outputs = outputs
        self.op = op
    

class FuncOp:
    """
    Operation that models function execution.
    """
    def __init__(self, func:Callable, version:int=0, is_super:bool=False):
        self.func = func 
        self.py_sig = inspect.signature(self.func)
        self.sig = Signature.from_py(sig=inspect.signature(func), 
                                     name=func.__name__, version=version, 
                                     is_super=is_super)
        self.is_synchronized = False
    
    def compute(self, inputs:Dict[str, Any]) -> List[Any]:
        inv_mapping = {v: k for k, v in self.sig.input_mapping.items()}
        inputs = {inv_mapping[k]: v for k, v in inputs.items()}
        result = self.func(**inputs)
        if self.sig.n_outputs == 0:
            assert result is None, "Function returned non-None value"
            return []
        elif self.sig.n_outputs == 1:
            return [result]
        else:
            return list(result)