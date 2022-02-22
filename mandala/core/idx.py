from ..common_imports import *

class ImplIndex(object):

    def __init__(self):
        self._data:TDict[str, type] = {}
    
    def implementations(self) -> TDict[str, type]:
        return self._data
    
    def get_impl_id(self, cls:type) -> str:
        return cls.__name__
    
    def get(self, impl_id:str) -> type:
        return self._data[impl_id]
    
    def register(self, impl:type, impl_id:str=None, safe:bool=False):
        if impl_id is None:
            impl_id = impl.__name__
        if safe:
            assert impl_id not in self._data
        self._data[impl_id] = impl
    

OpIndex = ImplIndex()
ValueIndex = ImplIndex()
KVIndex = ImplIndex()
BuiltinOpClasses = []