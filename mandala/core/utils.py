from .tps import Type, TypeWrapper

from ..common_imports import *

################################################################################
### 
################################################################################
class BackwardCompatible(object):
    def __init__(self, default:TAny=None):
        self.default = default
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackwardCompatible):
            return False
        return self.default == other.default
    
    def __repr__(self) -> str:
        return f'CompatArg(default={self.default})'
        

def CompatArg(default:TAny=None) -> TAny:
    return BackwardCompatible(default=default)

################################################################################
### 
################################################################################
class AnnotatedObj(object):
    
    def __init__(self, obj:TAny, transient:bool=False, target_type:Type=None, 
                 delayed_storage:bool=False):
        """
        Used to annotate returns of operations with intended interpretations. 
        """
        self._obj = obj
        self._transient = transient
        self._delayed_storage = delayed_storage
        self._target_type = target_type
    
    @property
    def obj(self) -> TAny:
        return self._obj
    
    @property
    def transient(self) -> bool:
        return self._transient
    
    @property
    def delayed_storage(self) -> bool:
        return self._delayed_storage
    
    @property
    def target_type(self) -> TOption[Type]:
        return self._target_type
    
    @property
    def has_type(self) -> bool:
        return self._target_type is not None


T = typing.TypeVar('T')

def AsType(obj:T, tp:TAny) -> T:
    #! check if the type matches the object here, if possible
    if not isinstance(obj, AnnotatedObj):
        obj = AnnotatedObj(obj=obj)
    assert obj.target_type is None
    if isinstance(tp, TypeWrapper):
        tp = tp.tp
    obj._target_type = tp
    return obj

def AsTransient(obj:T) -> T:
    if not isinstance(obj, AnnotatedObj):
        obj = AnnotatedObj(obj=obj)
    assert not obj.delayed_storage
    obj._transient = True
    return obj

def AsDelayedStorage(obj:T) -> T:
    if not isinstance(obj, AnnotatedObj):
        obj = AnnotatedObj(obj=obj)
    assert not obj.transient
    obj._delayed_storage = True
    return obj