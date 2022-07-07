from ..common_imports import *

class KVStore:
    """
    Interface for key-value stores for Python objects (keyed by UID).
    """
    def exists(self, k:str) -> bool:
        raise NotImplementedError()
    
    def set(self, k:str, v:Any) -> None:
        raise NotImplementedError()
    
    def get(self, k:str) -> Any:
        raise NotImplementedError()
    
    def delete(self, k:str) -> None:
        raise NotImplementedError()
    

class JoblibStorage(KVStore):
    """
    Simple file-based implementation for local testing
    """
    def __init__(self, root:Path):
        self.root = root

    def get_obj_path(self, k:str) -> Path:
        return self.root / f'{k}.joblib'

    def exists(self, k:str) -> bool:
        return self.get_obj_path(k).exists() 

    def set(self, k:str, v:Any):
        joblib.dump(v, self.get_obj_path(k))
    
    def get(self, k:str) -> Any:
        return joblib.load(self.get_obj_path(k))
    
    def delete(self, k:str):
        os.remove(path=self.get_obj_path(k=k))


class InMemoryStorage(KVStore):
    """
    In-memory implementation
    """
    def __init__(self):
        self.data = {}

    def exists(self, k:str) -> bool:
        return k in self.data

    def set(self, k:str, v:Any):
        self.data[k] = v

    def get(self, k:str) -> Any:
        return self.data[k]

    def delete(self, k:str):
        del self.data[k]
