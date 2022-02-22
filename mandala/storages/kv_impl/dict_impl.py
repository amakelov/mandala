import warnings

from ..kv import KVStore, KVIndex

from ...common_imports import *


class DictStorage(KVStore):

    def __init__(self, copy_outputs:bool=False):
        self._root = None
        self._data = {}
        self._copy_outputs = copy_outputs

    @staticmethod
    def from_meta(root: Path, desc: TAny) -> 'KVStore':
        return DictStorage()

    def get_meta(self) -> TAny:
        return None
    
    @property
    def root(self) -> Path:
        return self._root
    
    @root.setter
    def root(self, value):
        self._root = value
    
    def reflect_root_change(self, new_root: Path):
        self._root = new_root

    def exists(self, k: str) -> bool:
        return k in self._data
    
    def get(self, k: str) -> TAny:
        if self._copy_outputs:
            return copy.deepcopy(self._data[k])
        else:
            return self._data[k]
    
    def set(self, k: str, v: TAny):
        self._data[k] = v
    
    def delete(self, k: str, must_exist: bool = True):
        if must_exist:
            assert k in self._data
        if k in self._data:
            del self._data[k]
        
    def keys(self, limit: int = None) -> TList[str]:
        sorted_keys = sorted(self._data)
        if limit is not None:
            sorted_keys = sorted_keys[:limit]
        return sorted_keys
    
    @property
    def empty(self) -> bool:
        return not self._data
    
    def isin(self, rng: TList[TAny], keys: TList[str] = None) -> TList[str]:
        keys = self.keys() if keys is None else list(set(keys))
        result = []
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            for k in keys:
                try:
                    if self._data[k] in rng:
                        result.append(k)
                except:
                    continue
        return result
    
    def where(self, pred:TCallable, keys:TList[str]=None) -> TList[str]:
        keys = self.keys() if keys is None else list(set(keys))
        result = []
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            for k in keys:
                try:
                    if pred(self._data[k]):
                        result.append(k)
                except:
                    continue
        return result

KVIndex.register(impl=DictStorage, impl_id=DictStorage.get_impl_id())