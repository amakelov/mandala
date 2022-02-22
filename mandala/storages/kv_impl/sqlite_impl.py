from ..kv import KVStore, KVIndex

from ...common_imports import *


class SQLiteStorage(KVStore):
    UID_COL = '__index__'
    
    def __init__(self, 
                 root:Path=None,
                 dtype:str='BLOB', 
                 serializer:TCallable=None,
                 deserializer:TCallable=None, 
                 journal_mode:str='WAL',
                 page_size:int=32768, 
                 mmap_size_MB:int=256,
                 cache_size_pages:int=1000,
                 synchronous:str='normal'):
        self._root = root 
        self.dtype = dtype
        if serializer is None:
            serializer = pickle.dumps
        self.serializer = serializer
        if deserializer is None:
            deserializer = pickle.loads
        self.deserializer = deserializer

        self.journal_mode = journal_mode
        self.page_size = page_size
        self.mmap_size_MB = mmap_size_MB
        self.cache_size_pages = cache_size_pages
        self.synchronous = synchronous
        self.path = None
        self.address = None
        self.uri = True

        if self.root is not None:
            self._init(root=self.root)
            self.setup()
    
    @property
    def root(self) -> TOption[Path]:
        return self._root
    
    @root.setter
    def root(self, value:Path=None):
        print(value)
        self._init(root=value)
        self.setup()
    
    def get_meta(self) -> TAny:
        return {
            'dtype': self.dtype,
            'journal_mode': self.journal_mode,
            'page_size': self.page_size,
            'mmap_size_MB': self.mmap_size_MB,
            'cache_size_pages': self.cache_size_pages,
            'synchronous': self.synchronous
        }
    
    def reflect_root_change(self, new_root: Path):
        self._init(root=new_root)
    
    @staticmethod
    def from_meta(root:Path, desc:TAny) -> 'KVStore':
        raise NotImplementedError()

    def _init(self, root:Path):
        """
        Sets dynamic attributes associated with this root
        """
        # sets root, path, address and uri
        assert root.is_absolute()
        if not root.exists():
            root.mkdir(parents=True)
        self._root = root
        self.path = self._root / 'data.db'
        self.address = self.path

    ############################################################################ 
    ### SQLite internals
    ############################################################################ 
    def get_optimizations(self) -> TList[str]:
        """
        NOTE:
            - you cannot change `page_size` after setting `journal_mode = WAL`
            - `journal_mode = WAL` is persistent across database connections
            - `cache_size` is in pages when positive, in kB when negative
        """
        if self.mmap_size_MB is None:
            mmap_size = 0
        else:
            mmap_size = self.mmap_size_MB * 1024**2
        pragma_dict = OrderedDict([
            # 'temp_store': 'memory',
            ('synchronous', self.synchronous),
            ('page_size', self.page_size),
            ('cache_size', self.cache_size_pages),
            ('journal_mode', self.journal_mode),
            ('mmap_size', mmap_size),
        ])
        lines = [f'PRAGMA {k} = {v};' for k, v in pragma_dict.items()]
        return lines
    
    def apply_optimizations(self, conn:sqlite3.Connection):
        opts = self.get_optimizations()
        for line in opts:
            conn.execute(line)
    
    def conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.address),
                               isolation_level=None, uri=self.uri)
    
    def setup(self):
        query = f"""CREATE TABLE IF NOT EXISTS var({self.UID_COL} CHAR(32) PRIMARY KEY NOT NULL, value {self.dtype} NOT NULL); """
        with self.conn() as conn:
            # certain pragmas, such as page_size, must be given before the database
            # is even written to disk
            # self.run_optimization(conn=conn)
            self.apply_optimizations(conn=conn)
            conn.execute(query)

    def execute(self, query:str, parameters:TIter[TAny]=None):
        with self.conn() as conn:
            self.apply_optimizations(conn=conn)
            if parameters is not None:
                res = conn.execute(query, parameters)
            else:
                res = conn.execute(query)
        return res
    
    def read(self, query:str) -> TList[TTuple[TAny,...]]:
        return self.execute(query=query).fetchall()
    
    def where_clause(self, keys:TUnion[TList[str], pd.Index]) -> str:
        return ', '.join([f"'{k}'" for k in keys])
    
    def disk_usage(self) -> float:
        """
        Return disk usage in megabytes
        """
        assert self.path is not None
        return os.stat(self.path).st_size / 1024**2
            
    ############################################################################ 
    ### core interface
    ############################################################################ 
    def exists(self, k:str) -> bool:
        query = f"""SELECT EXISTS(SELECT 1 FROM var WHERE var.{self.UID_COL} = '{k}')"""
        results = self.execute(query).fetchall()
        return results[0][0] == 1

    def mexists(self, ks: TList[str]) -> TList[bool]:
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        unique_exists = self.mexists_unique(ks=unique_ks)
        return [unique_exists[i] for i in key_to_unique_idx]
    
    def mexists_unique(self, ks:TList[str]) -> TList[bool]:
        query = f"""SELECT {self.UID_COL} FROM var WHERE {self.UID_COL} IN ({self.where_clause(keys=ks)})"""
        result = set([elt[0] for elt in self.read(query=query)])
        return [k in result for k in ks]

    def set(self, k:str, v:TAny):
        query = f"""INSERT OR REPLACE INTO var({self.UID_COL}, value) VALUES('{k}', ?)"""
        self.execute(query=query, parameters=[self.serializer(v)])
    
    def mset(self, mapping:TMutMap, chunk_size:int=65_536):
        values_list = []
        parameters = []
        for k, v in mapping.items():
            values_list.append(f"('{k}', ?)")
            parameters.append(self.serializer(v))
        # you need to break this up into chunks b/c sqlite has limits on how
        # many parameters you can have
        num_items = len(mapping)
        num_chunks = math.ceil(num_items / chunk_size)
        value_chunks = [values_list[chunk_size*i:chunk_size*(i+1)] 
                        for i in range(num_chunks)]
        parameter_chunks = [parameters[chunk_size*i:chunk_size*(i+1)] 
                            for i in range(num_chunks)]
        with self.conn() as conn:
            for value_chunk, parameter_chunk in zip(value_chunks, parameter_chunks):
                values = ',\n'.join(value_chunk)
                query = f"""INSERT OR REPLACE INTO var({self.UID_COL}, value) VALUES {values}; """
                conn.execute(query, parameter_chunk)
    
    def get(self, k:str) -> TAny:
        query = f"""SELECT * FROM var WHERE {self.UID_COL} = '{k}'"""
        res = self.read(query=query)
        if not res:
            raise KeyError(k)
        return self.deserializer(res[0][1])
    
    def mget(self, ks: TList[str]) -> TList[TAny]:
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        unique_vs = self.mget_unique(ks=unique_ks)
        return [unique_vs[i] for i in key_to_unique_idx]
    
    def mget_unique(self, ks:TList[str]=None) -> TList[TAny]:
        if ks is None:
            ks = self.keys()
        query = f"""SELECT {self.UID_COL}, value FROM var WHERE {self.UID_COL} IN ({self.where_clause(keys=ks)})"""
        res = {elt[0]: self.deserializer(elt[1])
               for elt in self.read(query=query)}
        return [res[k] for k in ks]
    
    def delete(self, k:str, must_exist:bool=False):
        if must_exist:
            assert self.exists(k=k)
        query = f"""DELETE FROM var WHERE {self.UID_COL} = '{k}'"""
        self.execute(query)
    
    def mdelete(self, ks: TList[str], must_exist:bool=False):
        unique_ks, key_to_unique_idx = np.unique(ar=ks, return_inverse=True)
        self.mdelete_unique(ks=unique_ks, must_exist=must_exist)
    
    def mdelete_unique(self, ks:TList[str], must_exist:bool=False):
        if must_exist:
            exist_mask = self.mexists_unique(ks=ks)
            ks = [k for i, k in enumerate(ks) if exist_mask[i]]
        values = self.where_clause(keys=ks)
        query = f"""DELETE FROM var WHERE {self.UID_COL} IN ({values})"""
        self.execute(query)
    
    ############################################################################ 
    ### 
    ############################################################################ 
    def __len__(self) -> int:
        query = """
        SELECT count(*) FROM var;
        """
        return self.execute(query).fetchall()[0][0]
    
    def keys(self, limit:int=None) -> TList[str]:
        if limit is None:
            query = f"""
            SELECT {self.UID_COL} from var
            """
            return [elt[0] for elt in self.read(query=query)]
        else:
            query = f"""
            SELECT {self.UID_COL} from var
            LIMIT {limit}"""
            return [elt[0] for elt in self.read(query=query)]
    
    @property
    def empty(self) -> bool:
        keys = self.keys(limit=1)
        return len(keys) == 0

    def where(self, pred:TCallable, keys:TList[str]=None) -> TList[str]:
        keys = self.keys() if keys is None else list(set(keys))
        vals = self.mget_unique(ks=keys)
        result = []
        for k, val in zip(keys, vals):
            try:
                if pred(val):
                    result.append(k)
            except:
                pass
        return result
    
    def isin(self, rng:TList[TAny], keys:TList[str] = None) -> TList[str]:
        keys = self.keys() if keys is None else list(set(keys))
        vals = self.mget_unique(ks=keys)
        result = []
        for k, val in zip(keys, vals):
            try:
                if val in rng:
                    result.append(k)
            except:
                pass
        return result

    def __repr__(self) -> str:
        return f'SQLiteStorage(root={self._root})'

KVIndex.register(impl=SQLiteStorage, impl_id=SQLiteStorage.get_impl_id())