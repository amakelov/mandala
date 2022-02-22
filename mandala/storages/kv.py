from abc import ABC, abstractmethod

from ..common_imports import *
from ..core.idx import KVIndex
from ..util.common_ut import group_like, ungroup_like, chunk_list

class KVStore(ABC):
    """
    fs-based key-value store
    """
    def __init__(self, root:Path=None):
        self._root = root 

    ############################################################################  
    ### logistics
    ############################################################################  
    def attach(self, root:Path) -> 'KVStore':
        """
        set the location for this storage's data. 
        """
        assert not self.is_attached()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dump_meta()
        return self
    
    def is_attached(self) -> bool:
        return self.root is not None

    @property
    @abstractmethod
    def root(self) -> Path:
        raise NotImplementedError()
    
    @root.setter
    @abstractmethod
    def root(self, value:Path):
        raise NotImplementedError()

    @property
    def objs_root(self) -> Path:
        assert self.is_attached()
        return self.root / 'objs/'
    
    @staticmethod
    def meta_path(root) -> Path:
        return root / '__meta__.joblib'
    
    @staticmethod
    def impl_path(root) -> Path:
        return root / '__impl__.joblib'
    
    @classmethod
    def get_impl_id(cls) -> str:
        return cls.__name__

    @abstractmethod
    def get_meta(self) -> TAny:
        raise NotImplementedError()

    def dump_meta(self):
        desc = self.get_meta()
        joblib.dump(value=desc, filename=self.meta_path(root=self.root))
        joblib.dump(value=self.get_impl_id(), 
                    filename=self.impl_path(root=self.root))
    
    @staticmethod
    @abstractmethod
    def from_meta(root:Path, desc:TAny) -> 'KVStore':
        raise NotImplementedError()
    
    @staticmethod
    def load_implementation(root:Path) -> 'KVStore':
        meta_path = KVStore.meta_path(root=root)
        meta = joblib.load(filename=meta_path)
        impl_id_path = KVStore.impl_path(root=root)
        impl_id = joblib.load(filename=impl_id_path)
        ImplClass:TType[KVStore] = KVIndex.get(impl_id=impl_id)
        return ImplClass.from_meta(root=root, desc=meta)

    ### moving around
    @abstractmethod
    def reflect_root_change(self, new_root:Path):
        """
        update dynamic attributes when the data on the fs has been moved.
        """
        raise NotImplementedError()
    
    def move(self, new_root:Path):
        """
        Move all fs data of this KVStore to a new root.
        """
        if self.root is None:
            raise ValueError('Cannot move detached KVStore')
        if not new_root.is_absolute():
            raise ValueError('Can only move to absolute path')
        if new_root.exists():
            raise ValueError()
        shutil.move(src=str(self.root.absolute()), dst=new_root)
        self.reflect_root_change(new_root=new_root)
    
    ############################################################################  
    @property
    def tqdm_delay(self) -> float:
        return 5.0
    
    def getiter(self, objs:TIter) -> TIter:
        return tqdm.tqdm(objs, delay=self.tqdm_delay)
    
    @abstractmethod
    def exists(self, k:str) -> bool:
        raise NotImplementedError()
    
    def mexists(self, ks:TList[str]) -> TList[bool]:
        return [self.exists(k=k) for k in self.getiter(ks)]
    
    @abstractmethod
    def set(self, k:str, v:TAny):
        raise NotImplementedError()
    
    def set_if_not_exists(self, k:str, v:TAny):
        if not self.exists(k=k):
            self.set(k=k, v=v)
    
    def create(self, k:str, v:TAny):
        if self.exists(k=k):
            raise ValueError()
        self.set(k=k, v=v)
    
    def mset(self, mapping:TDict[str, TAny]):
        for k, v in self.getiter(mapping.items()):
            self.set(k=k, v=v)
        
    def mset_if_not_exists(self, mapping:TDict[str, TAny]):
        """
        A default implementation that tries to be reasonable if other routines
        have been optimized
        """
        keys = list(mapping.keys())
        values = [mapping[k] for k in keys]
        exist_mask = self.mexists(ks=keys)
        new_mapping = {k: v for i, (k, v) in enumerate(zip(keys, values))
                       if not exist_mask[i]}
        self.mset(mapping=new_mapping)
    
    def mcreate(self, mapping:TDict[str, TAny]):
        exist_mask = self.mexists(ks=list(mapping))
        if any(exist_mask):
            raise ValueError()
        self.mset(mapping=mapping)
            
    def _mset_new_only(self, mapping:TMutMap):
        keys = np.array(sorted(mapping.keys())).tolist()
        exist_mask = np.array(self.mexists(ks=keys)).astype(bool) # to guard against empty values
        new_keys = keys[~exist_mask]
        new_mapping = {k: mapping[k] for k in new_keys}
        self.mset(mapping=new_mapping)
    
    @abstractmethod
    def get(self, k:str) -> TAny:
        raise NotImplementedError()
    
    def get_default(self, k:str, default:TAny) -> TAny:
        if self.exists(k=k):
            return self.get(k=k)
        else:
            return default
    
    def mget(self, ks:TList[str]) -> TList[TAny]:
        return [self.get(k) for k in self.getiter(ks)]
    
    def mget_default(self, ks:TList[str], default:TAny) -> TList[TAny]:
        exist_mask = self.mexists(ks=ks)
        groups = group_like(objs=ks, labels=exist_mask)
        results = {
            True: self.mget(ks=groups[True]),
            False: [default for _ in range(len(groups[False]))]
        }
        return ungroup_like(groups=results, labels=exist_mask)

    def mget_repeats(self, ks:TList[str]=None) -> TList[TAny]:
        uniques, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        vs = self.mget(ks=uniques)
        return [vs[i] for i in key_to_unique_idx]
    
    @abstractmethod
    def delete(self, k:str, must_exist:bool=True):
        raise NotImplementedError()
        
    def mdelete(self, ks:TList[str], must_exist:bool=True):
        for k in self.getiter(ks):
            self.delete(k=k, must_exist=must_exist)
    
    def delete_all(self):
        self.mdelete(ks=self.keys())
    
    @abstractmethod
    def keys(self, limit:int=None) -> TList[str]:
        raise NotImplementedError()
    
    @property
    def size(self) -> int:
        return len(self.keys())
    
    @property
    @abstractmethod
    def empty(self) -> bool:
        raise NotImplementedError()

    def space_usage(self) -> str:
        assert self.root is not None
        return get_folder_size(path=self.root)
    
    ############################################################################ 
    ### query interface
    ############################################################################ 
    @abstractmethod
    def where(self, pred:TCallable, keys:TList[str]=None) -> TList[str]:
        """
        Return the keys of the values satisfying the given predicate, optionally
        restricting search to `keys`
        
        NOTE: 
            - `keys` could contain duplicates
        """
        raise NotImplementedError()
    
    @abstractmethod
    def isin(self, rng:TList[TAny], keys:TList[str]=None) -> TList[str]:
        """
        Return keys of the values in the given list, optionally
        restricting search to `keys`
        
        NOTE: 
            - `keys` could contain duplicates
        """
        raise NotImplementedError()
    

class KVGroup(object):

    def __init__(self, root:Path, default_kv:KVStore):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.default_kv = default_kv
        self.kvs:TDict[str, KVStore] = {}
    
    def keys(self) -> TList[str]:
        return os.listdir(self.root)
    
    def get_kv_root(self, k:str) -> Path:
        return self.root / k
    
    def exists(self, k:str) -> bool:
        return k in self.keys()
    
    def reflect_kvs(self):
        """
        Populate the dictionary of KVStores by examining the files under this
        group's root directory.
        """
        for k in self.keys():
            kv_root = self.get_kv_root(k=k)
            kv = KVStore.load_implementation(root=kv_root)
            kv.attach(root=kv_root)
            self.kvs[k] = kv
    
    def get(self, k:str) -> KVStore:
        if k not in self.kvs:
            kv = copy.deepcopy(self.default_kv)
            self.set(k=k, kv=kv)
        return self.kvs[k]        
   
    def set(self, k:str, kv:KVStore):
        kv.attach(self.get_kv_root(k=k))
        self.kvs[k] = kv
        
    def get_full_data(self) -> TDict[str, TDict[str, TAny]]:
        res = {}
        for k in self.keys():
            kv = self.get(k)
            ks = kv.keys()
            vs = kv.mget(ks)
            res[k] = {k: v for k, v in zip(ks, vs)}
        return res

    def __getitem__(self, k:str) -> KVStore:
        return self.get(k=k)

    def __setitem__(self, k:str, kv:KVStore):
        self.set(k=k, kv=kv)
    
    def space_usage(self) -> TDict[str, str]:
        return {k: self.kvs[k].space_usage() for k in self.keys()}


def get_folder_size(path:Path) -> str:
    assert path.is_absolute()
    result = subprocess.check_output(['du', '-sh', str(path)]).split()[0]
    return result.decode('utf-8')