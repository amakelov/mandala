from .context import GlobalContext, Context
from .storage import Storage
from .execution import wrap

from ..common_imports import *
from ..storages.kv import KVStore
from ..storages.objects import BaseObjLocation
from ..core.tps import Type, TypeWrapper, BuiltinTypes
from ..core.bases import ValueRef
from ..core.impl import AtomRef
from ..core.config import CoreConfig, MODES
from ..queries.rel_weaver import ValQuery
from ..session import *

class _Var(TypeWrapper):
    _VALUE_SENTINEL = object()
    
    def __init__(self, annotation:TAny, name:str=None, kv:KVStore=None,
                 storage:Storage=None, hash_method:str=None):
        if hash_method is None:
            hash_method = CoreConfig.default_hash_method
        self.annotation = annotation
        self.name = name
        self.kv = kv
        tp = Type.from_annotation(annotation=annotation)
        tp.set_hash_method(method=hash_method)
        self._tp = tp
        self._synchronized = False
        if self.kv is not None:
            assert name is not None, 'Name must be provided for custom KVStores'
        if GlobalContext.exists():
            storage = GlobalContext.get().storage
            if storage is not None:
                self.synchronize(storage=storage)
            
    def synchronize(self, storage:Storage):
        if self.name is not None:
            storage.val_adapter.synchronize_type(ui_name=self.name, tp=self.tp, 
                                                 kv=self.kv)
        self._synchronized = True
    
    @property
    def synchronized(self) -> bool:
        return self._synchronized
    
    def set_synchronized(self):
        self._synchronized = True
        
    @property
    def tp(self) -> Type:
        return self._tp
    
    def get_locs(self, storage:Storage=None) -> TList[BaseObjLocation]:
        storage = GlobalContext.get().storage if storage is None else storage
        assert storage is not None
        return storage.val_adapter.get_tp_locs(tp=self.tp) 
    
    def eval(self, storage:Storage=None) -> TDict[BaseObjLocation, ValueRef]:
        storage = GlobalContext.get().storage if storage is None else storage
        assert storage is not None
        return storage.val_adapter.eval_tp(tp=self.tp)
    
    def wrap_query(self, value:TAny=_VALUE_SENTINEL) -> ValQuery:
        res =  ValQuery(tp=self.tp)
        if value is not self._VALUE_SENTINEL:
            res = res.equals(value=value)
        return res
    
    def __call__(self, value:TAny=_VALUE_SENTINEL, __context__:Context=None):
        assert self.synchronized
        context = GlobalContext.get(fallback=__context__)
        mode = context.mode
        if mode == MODES.noop:
            return value
        elif mode in (MODES.run, MODES.transient):
            assert value is not self._VALUE_SENTINEL
            return wrap(obj=value, reference=self.tp, type_dict=None, c=context)
        elif mode == MODES.query:
            return self.wrap_query(value=value)
        else:
            raise NotImplementedError()
    
    def __repr__(self) -> str:
        return f'Variable(annotation={self.tp.annotation}, type={self.tp}, kv={self.kv})'
        

T = typing.TypeVar('T')
def Var(annotation:T, name:str=None, kv:KVStore=None, hash_method:str=None) -> T:
    if hash_method is None:
        hash_method = CoreConfig.default_hash_method
    return _Var(annotation=annotation, name=name, kv=kv, hash_method=hash_method)

    
class BuiltinVars(object):
    #! bogus
    x = Var(annotation=BuiltinTypes.get(py_type=int))
    x.set_synchronized()
    # mostly used to match indices in a list in queries
    IndexQuery = x
    x = Var(annotation=BuiltinTypes.get(py_type=str))
    x.set_synchronized()
    # mostly used to match keys in a dict 
    KeyQuery = x
    
def Query(annotation:TAny=TAny, name:str=None) -> TAny:
    return ValQuery.from_annotation(annotation=annotation, display_name=name)