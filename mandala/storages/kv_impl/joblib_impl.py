import math

from joblib import Parallel, delayed

from ..kv import KVStore, KVIndex

from ...common_imports import *
from ...util.common_ut import chunk_list

class JoblibStorage(KVStore):
    DEFAULT_QUERY_BATCH_SIZE = 1_000
    
    def __init__(self, forgetful:bool=False, parallel:bool=False, 
                 progress:bool=False, query_batch_size:int=None, 
                 tqdm_delay:float=5, parallel_threshold:int=10):
        self.forgetful = forgetful
        self.parallel = parallel
        self.parallel_verbosity = 5
        self.progress = progress
        self._tqdm_delay = tqdm_delay
        self.query_batch_size = query_batch_size
        self.parallel_threshold = parallel_threshold
        self._root = None
    
    @property
    def tqdm_delay(self) -> float:
        return self._tqdm_delay

    def attach(self, root: Path):
        self._root = root 
        root.mkdir(parents=True, exist_ok=True)
        self.objs_root.mkdir(parents=True, exist_ok=True)
        self.dump_meta()
    
    @property
    def root(self) -> Path:
        return self._root

    def get_meta(self) -> TAny:
        return {}

    @staticmethod
    def from_meta(root:Path, desc:TAny) -> 'KVStore':
        res = JoblibStorage()
        res.attach(root=root)
        return res
    
    def reflect_root_change(self, new_root: Path):
        self._root = new_root
    
    ############################################################################ 
    ### internals
    ############################################################################ 
    def obj_path(self, k:str) -> Path:
        return self.objs_root / f'{k}.joblib'

    ############################################################################ 
    ### 
    ############################################################################ 
    def exists(self, k:str) -> bool:
        return self.obj_path(k=k).is_file()
    
    def mexists(self, ks:TList[str], batched:bool=False) -> TList[bool]:
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        unique_exists = self.mexists_unique(ks=unique_ks, batched=batched)
        return [unique_exists[i] for i in key_to_unique_idx]

    def mexists_unique(self, ks: TList[str], batched:bool=False) -> TList[bool]:
        """
        When batched is True, we load all keys and check against them;
        otherwise, we check each key.
        """
        if batched:
            all_keys = self.keys()
            intersection = set(ks) & set(all_keys)
            return [k in intersection for k in ks]
        else:
            if not self.parallel or len(ks) < self.parallel_threshold:
                return [self.exists(k) 
                        for k in tqdm.tqdm(ks, delay=self.tqdm_delay)]
            else:
                checker = lambda k: self.exists(k=k)
                verbose = self.parallel_verbosity if self.progress else 0
                return Parallel(n_jobs=-1,
                                verbose=verbose)(delayed(checker)(k) 
                                                 for k in ks)

    def get(self, k:str) -> TAny:
        return joblib.load(filename=self.obj_path(k=k))

    def mget(self, ks: TList[str]) -> TList[TAny]:
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        unique_vs = self.mget_unique(ks=unique_ks)
        return [unique_vs[i] for i in key_to_unique_idx]
    
    def mget_unique(self, ks: TList[str]) -> TList[TAny]:
        if not self.parallel or len(ks) < 10:
            return super().mget(ks=ks)
        else:
            getter = lambda x: self.get(x)
            verbose = self.parallel_verbosity if self.progress else 0
            return Parallel(n_jobs=-1, verbose=verbose)(delayed(getter)(k)
                                                        for k in ks)

    def set(self, k:str, v:TAny):
        v = None if self.forgetful else v
        joblib.dump(value=v, filename=self.obj_path(k=k))
    
    def mset(self, mapping: TDict[str, TAny]):
        if not self.parallel or len(mapping) < 10:
            return super().mset(mapping)
        else:
            setter = lambda x: self.set(k=x[0], v=x[1])
            verbose = self.parallel_verbosity if self.progress else 0
            Parallel(n_jobs=-1, verbose=verbose)(delayed(setter)((k, v))
                                                 for k, v in mapping.items())
    
    def delete(self, k:str, must_exist:bool=True):
        if os.path.exists(self.obj_path(k=k)):
            os.remove(path=self.obj_path(k=k))
            return
        if must_exist:
            raise ValueError()
        
    def mdelete(self, ks: TList[str], must_exist: bool = True):
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        self.mdelete_unique(ks=unique_ks, must_exist=must_exist)
    
    def mdelete_unique(self, ks: TList[str], must_exist: bool = True):
        if not self.parallel or len(ks) < 10:
            return super().mdelete(ks, must_exist=must_exist)
        else:
            deleter = lambda x: self.delete(k=x, must_exist=must_exist)
            verbose = self.parallel_verbosity if self.progress else 0
            Parallel(n_jobs=-1, verbose=verbose)(delayed(deleter)(k) 
                                                 for k in ks)
    
    ############################################################################ 
    ### 
    ############################################################################ 
    def keys(self, limit:int=None) -> TList[str]:
        res = []
        if limit is None:
            limit = math.inf
        iterations = 0
        with os.scandir(self.objs_root) as it:
            iterator = tqdm.tqdm(it, delay=self.tqdm_delay)
            for elt in iterator:
                res.append(str(elt.name).split('.')[0])
                iterations += 1
                if iterations >= limit:
                    break
        return res
    
    @property
    def empty(self) -> bool:
        with os.scandir(self.objs_root) as it:
            found_any = False
            for elt in it:
                found_any = True
                break
        return (not found_any)
        
    def where(self, pred: TCallable, keys:TList[str]=None) -> TList[str]:
        if keys is None:
            keys = self.keys()
        else:
            keys = list(set(keys))
        if self.query_batch_size is not None:
            batch_size = self.query_batch_size
        else:
            batch_size = self.DEFAULT_QUERY_BATCH_SIZE
        key_chunks = chunk_list(lst=keys, chunk_size=batch_size)
        res = []
        for chunk in key_chunks:
            chunk_vs = self.mget(ks=chunk)
            for k, v in zip(chunk, chunk_vs):
                try:
                    if pred(v):
                        res.append(k)
                except:
                    pass
        return res
    
    def isin(self, rng: TList[TAny], keys:TList[str]=None) -> TList[str]:
        if keys is None:
            keys = self.keys()
        else:
            keys = list(set(keys))
        if self.query_batch_size is not None:
            batch_size = self.query_batch_size
        else:
            batch_size = self.DEFAULT_QUERY_BATCH_SIZE
        key_chunks = chunk_list(lst=keys, chunk_size=batch_size)
        res = []
        for chunk in key_chunks:
            chunk_vs = self.mget(ks=chunk)
            for k, v in zip(chunk, chunk_vs):
                try:
                    if v in rng:
                        res.append(k)
                except:
                    pass
        return [k for k in keys if self.get(k) in rng]

    def __repr__(self) -> str:
        return f'JoblibStorage(root={self._root}, parallel={self.parallel})'
    
    
KVIndex.register(impl=JoblibStorage, impl_id=JoblibStorage.get_impl_id())