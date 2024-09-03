from .common_imports import *
from tqdm import tqdm
import uuid
from .utils import serialize, deserialize
from .model import Call
import joblib
import sqlite3
from abc import ABC, abstractmethod


class DBAdapter:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if self.in_memory:
            # maintain a single connection throughout the lifetime of the object
            # avoid clashes with other in-memory databases
            self._id = str(uuid.uuid4())
            self._connection_address = f"file:{self._id}?mode=memory&cache=shared"
            self._conn = sqlite3.connect(
                str(self._connection_address), isolation_level=None, uri=True
            )
        if not self.in_memory:
            if not os.path.exists(db_path):
                # create a database with incremental vacuuming
                conn = self.conn()
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                conn.execute("PRAGMA incremental_vacuum_threshold = 1024;")
                conn.close()
    
    @property
    def in_memory(self) -> bool:
        return self.db_path == ":memory:"
    
    def conn(self) -> sqlite3.Connection:
        if self.in_memory:
            return self._conn
        else:
            return sqlite3.connect(self.db_path)

def is_in_memory_db(conn):
    cursor = conn.execute("PRAGMA database_list")
    db_list = cursor.fetchall()
    if len(db_list) != 1:
        raise ValueError("Expected exactly one database")
    return db_list[0][2] == ''

def transaction(method):  # transaction decorator for classes with a `conn` method
    """

    """
    def wrapper(self, *args, **kwargs):
        if kwargs.get("conn") is not None:  # already in a transaction
            logging.debug("Folding into existing transaction")
            return method(self, *args, **kwargs)
        else:  # open a connection
            logging.debug(
                f"Opening new transaction from {self.__class__.__name__}.{method.__name__}"
            )
            conn = self.conn()
            try:
                res = method(self, *args, conn=conn, **kwargs)
                conn.commit()
                return res
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                if not is_in_memory_db(conn):
                    conn.close()
                else:
                    # in-memory databases are kept open
                    pass
    return wrapper


class DictStorage(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def drop(self, key: str) -> None:
        pass

    @abstractmethod
    def load_all(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.exists(key)
    
    def __len__(self) -> int:
        return len(self.keys())


class JoblibDictStorage(DictStorage):
    """
    A dictionary storage that uses joblib to store the data on disk.
    """
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
    
    def get_path_for_key(self, key: str) -> str:
        return os.path.join(self.root, f'{key}.joblib')
    
    def get(self, key: str) -> Any:
        return joblib.load(self.get_path_for_key(key))
    
    def exists(self, key: str) -> bool:
        return os.path.exists(self.get_path_for_key(key))
    
    def set(self, key: str, value: Any) -> None:
        if self.exists(key):
            return # this is a write-once storage
        joblib.dump(value, self.get_path_for_key(key))
    
    def drop(self, key: str) -> None:
        os.remove(self.get_path_for_key(key))

    def load_all(self) -> Dict[str, Any]:
        return {key: self.get(key) for key in self.keys()}
    
    def keys(self) -> List[str]:
        return [os.path.splitext(f)[0] for f in os.listdir(self.root) if f.endswith('.joblib')]
    
    def values(self) -> List[Any]:
        return [self.get(key) for key in self.keys()]
    
        


class SQLiteDictStorage(DictStorage):
    def __init__(self, db: DBAdapter,
                 table: str, 
                 overflow_storage: Optional[JoblibDictStorage] = None,
                 overflow_threshold_MB: Optional[Union[int, float]] = 50,
                 ):
        """

        """
        self.db = db
        self.table = table
        with self.conn() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value BLOB)"
            )
        self.overflow_storage = overflow_storage
        self.overflow_threshold_MB = overflow_threshold_MB
    
    def conn(self) -> sqlite3.Connection:
        return self.db.conn()
    
    def load_all(self) -> Dict[str, Any]:
        with self.conn() as conn:
            cursor = conn.execute(f"SELECT key, value FROM {self.table}")
            res = {row[0]: deserialize(row[1]) for row in cursor.fetchall()}
        if self.overflow_storage is not None:
            res.update(self.overflow_storage.load_all())
        return res

    @transaction
    def get(self, key: str, conn: Optional[sqlite3.Connection] = None) -> Any:
        cursor = conn.execute(f"SELECT value FROM {self.table} WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            if self.overflow_storage is not None:
                return self.overflow_storage.get(key)
            else:
                raise KeyError(f"Key {key} not found")
        return deserialize(result[0])

    @transaction
    def set(
        self, key: str, value: Any, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        serialized_value = serialize(value)
        # compute the space this string would take up in bytes
        size_MB = len(serialized_value) / 1024 / 1024
        if size_MB > self.overflow_threshold_MB:
            if self.overflow_storage is not None:
                self.overflow_storage.set(key, value)
            else:
                raise ValueError(
                    f"Value for key {key} is too large ({size_MB:.2f} MB) and no overflow storage is provided"
                )
        else:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.table} (key, value) VALUES (?, ?)",
                (key, serialized_value),
            )

    @transaction
    def drop(self, key: str, conn: Optional[sqlite3.Connection] = None) -> None:
        conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
        if self.overflow_storage is not None and self.overflow_storage.exists(key):
            self.overflow_storage.drop(key)

    @transaction
    def exists(self, key: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {self.table} WHERE key = ?", (key,)
        )
        count = cursor.fetchone()[0]
        if self.overflow_storage is None:
            return count > 0
        else:
            return count > 0 or self.overflow_storage.exists(key)

    @transaction
    def keys(self, conn: Optional[sqlite3.Connection] = None) -> List[str]:
        cursor = conn.execute(f"SELECT key FROM {self.table}")
        res = [row[0] for row in cursor.fetchall()]
        if self.overflow_storage is not None:
            res.extend(self.overflow_storage.keys())
        return res

    @transaction
    def values(self, conn: Optional[sqlite3.Connection] = None) -> List[Any]:
        cursor = conn.execute(f"SELECT value FROM {self.table}")
        res = [deserialize(row[0]) for row in cursor.fetchall()]
        if self.overflow_storage is not None:
            res.extend(self.overflow_storage.values())
        return res


class CachedDictStorage(DictStorage):
    def __init__(self, persistent: DictStorage):
        self.persistent = persistent
        # keep a cache of the values for faster lookups
        self.cache: Dict[str, Any] = {}
        # keep track of keys that have been added but not yet persisted
        self.dirty_keys: Set[str] = set()
    
    def load_all(self) -> Dict[str, Any]:
        return self.persistent.load_all()
    
    def __len__(self) -> int:
        return len(self.cache)

    def get(self, key: str) -> Any:
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.persistent.get(key)
            self.cache[key] = value
            return value

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.dirty_keys.add(key)

    def commit(self, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Persist the values of all dirty keys and clear them
        """
        for key in self.dirty_keys:
            self.persistent.set(key, self.cache[key], conn=conn)
        self.dirty_keys.clear()
    
    def clear(self, allow_uncommited: bool = False) -> None:
        if len(self.dirty_keys) > 0 and not allow_uncommited:
            # we add this as a precaution to avoid data loss. Otherwise, it's
            # easy to shoot yourself in the foot by calling `clear()` before
            # `commit()`
            msg = "Cannot clear cache with uncommitted changes; call `commit()` first, or use `allow_uncommited=True`"
            raise ValueError(msg)
        self.cache.clear()
        self.dirty_keys.clear()

    def drop(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
        if key in self.dirty_keys:
            self.dirty_keys.remove(key) # when we `drop`, we forget this key ever existed
        self.persistent.drop(key)

    def exists(self, key: str) -> bool:
        if key in self.cache:
            return True
        else:
            res = self.persistent.exists(key)
            return res


class InMemCallStorage:
    COLUMNS = [
        "call_history_id",
        "name",
        "direction",
        "call_content_id",
        "ref_content_id",
        "ref_history_id",
        "op",
        "semantic_version",
        "content_version",
    ]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is not None:
            self.df = df
            # for faster lookups
            self.call_hids = set(df.index.levels[0].unique())
        else:
            self.df = pd.DataFrame(columns=InMemCallStorage.COLUMNS).set_index(
                ["call_history_id", "name"]
            )
            self.call_hids = set()
        
    def __len__(self) -> int:
        return self.df.index.get_level_values(0).nunique()

    def save(self, call: Call):
        # if call.hid in self.df.index.levels[0]:
        if call.hid in self.call_hids:
            return
        if not hasattr(sess, '_times'):
            sess._times = []
        start = time.time()
        for k, v in call.inputs.items():
            self.df.loc[(call.hid, k), :] = ("in", call.cid, v.cid, v.hid, call.op.name, call.semantic_version, call.content_version)
        for k, v in call.outputs.items():
            self.df.loc[(call.hid, k), :] = (
                "out",
                call.cid,
                v.cid,
                v.hid,
                call.op.name,
                call.semantic_version,
                call.content_version,
            )
        end = time.time()
        sess._times.append(end - start)
        self.call_hids.add(call.hid)

    def drop(self, hid: str):
        """
        Remove all rows referencing the call with the given history_id.
        """
        if hid not in self.df.index.levels[0]:
            raise ValueError(f"Call with history_id {hid} does not exist")
        # self.df.drop(index=hid, level=0, inplace=True)
        self.df = self.df.drop(index=hid, level=0)
        #! this step is crucial, because otherwise the old `hid` remains in the index
        self.df.index = self.df.index.remove_unused_levels()
        self.call_hids.remove(hid)

    def exists(self, hid: str) -> bool:
        # return call_history_id in self.df.index.levels[0]
        return hid in self.call_hids
    
    def exists_content(self, cid: str) -> bool:
        return cid in self.df.call_content_id.unique()
    
    def mget_data(self, call_hids: List[str]) -> List[Dict[str, Any]]:
        idx = pd.IndexSlice
        filtered_df = self.df.loc[idx[call_hids, :], :]
        grouped = filtered_df.groupby(level=0)
        groups = {key: value for key, value in grouped}
        res_dict = {}
        for hid, group_df in tqdm(groups.items(), delay=5):
            rows = group_df.reset_index().to_dict(orient="records")
            input_hids, output_hids = {}, {}
            input_cids, output_cids = {}, {}
            for row in rows:
                if row["direction"] == "in":
                    input_hids[row["name"]] = row["ref_history_id"]
                    input_cids[row["name"]] = row["ref_content_id"]
                else:
                    output_hids[row["name"]] = row["ref_history_id"]
                    output_cids[row["name"]] = row["ref_content_id"]
            op_name = rows[0]["op"]
            res_dict[hid] = {
                "op_name": op_name,
                "cid": rows[0]["call_content_id"],
                "hid": hid,
                "input_hids": input_hids,
                "output_hids": output_hids,
                "input_cids": input_cids,
                "output_cids": output_cids,
                "semantic_version": rows[0]["semantic_version"],
                "content_version": rows[0]["content_version"],
            }
        return [res_dict[hid] for hid in call_hids]

    def get_data(self, call_history_id: str) -> Dict[str, Any]:
        """
        Get all the stuff associated with a call apart from the op.
        """
        if not self.exists(call_history_id):
            raise ValueError(f"Call with history_id {call_history_id} does not exist")
        return self.mget_data([call_history_id])[0]
    
    def get_data_content(self, cid: str) -> Dict[str, Any]:
        # find one hid associated with this cid
        hid = self.df.query('call_content_id == @cid').index.get_level_values(0).unique()[0]
        return self.get_data(hid)

    def get_creator_hids(self, ref_hids: Iterable[str]) -> Set[str]:
        #! slow
        call_history_ids = (
            self.df.query('ref_history_id in @ref_hids and direction == "out"')
            .index.get_level_values(0)
            .unique()
        )
        return set(call_history_ids)

    def get_consumer_hids(self, ref_hids: Iterable[str]) -> Set[str]:
        #! slow
        call_history_ids = (
            self.df.query('ref_history_id in @ref_hids and direction == "in"')
            .index.get_level_values(0)
            .unique()
        )
        return set(call_history_ids)

    def get_input_hids(self, call_hids: Iterable[str]) -> Set[str]:
        ref_hids = self.df.query(
            'call_history_id in @call_hids and direction == "in"'
        ).ref_history_id.unique()
        return set(ref_hids)

    def get_output_hids(self, call_hids: Iterable[str]) -> Set[str]:
        ref_hids = self.df.query(
            'call_history_id in @call_hids and direction == "out"'
        ).ref_history_id.unique()
        return set(ref_hids)

    def get_dependencies(
        self, ref_hids: Iterable[str], call_hids: Iterable[str]
    ) -> Tuple[Set[str], Set[str]]:
        refs_result = set(ref_hids).copy()
        calls_result = set(call_hids).copy()
        cur_refs = refs_result.copy()
        cur_calls = calls_result.copy()

        while True:
            calls_upd = self.get_creator_hids(cur_refs) - calls_result
            refs_upd = self.get_input_hids(cur_calls) - refs_result
            if (not calls_upd) and (not refs_upd):
                break
            calls_result |= calls_upd
            refs_result |= refs_upd
            cur_refs = refs_upd
            cur_calls = calls_upd
        return (refs_result, calls_result)

    def get_dependents(
        self, ref_hids: Iterable[str], call_hids: Iterable[str]
    ) -> Tuple[Set[str], Set[str]]:
        refs_result = set(ref_hids).copy()
        calls_result = set(call_hids).copy()
        cur_refs = refs_result.copy()
        cur_calls = calls_result.copy()

        while True:
            calls_upd = self.get_consumer_hids(cur_refs) - calls_result
            refs_upd = self.get_output_hids(cur_calls) - refs_result
            if (not calls_upd) and (not refs_upd):
                break
            calls_result |= calls_upd
            refs_result |= refs_upd
            cur_refs = refs_upd
            cur_calls = calls_upd
        return (refs_result, calls_result)


class SQLiteCallStorage:
    def __init__(self, db: DBAdapter, table_name: str):
        self.db = db
        self.table_name = table_name
        # if it doesn't exist, create a table with a two-column primary key
        # on call_history_id and name
        with self.db.conn() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} (call_history_id TEXT, name TEXT, direction TEXT, "
                "call_content_id TEXT, ref_content_id TEXT, ref_history_id TEXT, op TEXT, semantic_version TEXT, "
                "content_version TEXT, PRIMARY KEY (call_history_id, name))"
            )
    
    def conn(self) -> sqlite3.Connection:
        return self.db.conn()

    ###
    @transaction
    def get_df(self, conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
        """
        Load the full table into a DataFrame.
        """
        return pd.read_sql(f"SELECT * FROM {self.table_name}", conn).set_index(
            ["call_history_id", "name"]
        )

    @transaction
    def execute_df(
        self, query: str, conn: Optional[sqlite3.Connection] = None
    ) -> pd.DataFrame:
        return pd.read_sql(query, conn)

    @transaction
    def save(
        self, call_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None
    ):
        semantic_version = call_data["semantic_version"]
        content_version = call_data["content_version"]
        op_name = call_data["op_name"]
        for k in call_data["input_hids"]:
            hid = call_data["input_hids"][k]
            cid = call_data["input_cids"][k]
            conn.execute(
                f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (call_data["hid"], k, "in", call_data["cid"], cid, hid, op_name, semantic_version, content_version),
            )
        for k in call_data["output_hids"]:
            hid = call_data["output_hids"][k]
            cid = call_data["output_cids"][k]
            conn.execute(
                f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (call_data["hid"], k, "out", call_data["cid"], cid, hid, op_name, semantic_version, content_version),
            )

    @transaction
    def drop(self, hid: str, conn: Optional[sqlite3.Connection] = None):
        conn.execute(f"DELETE FROM {self.table_name} WHERE call_history_id = ?", (hid,))

    @transaction
    def exists(
        self, call_history_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE call_history_id = ?",
            (call_history_id,),
        )
        count = cursor.fetchone()[0]
        return count > 0
    
    @transaction
    def exists_content(
        self, cid: str, conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE call_content_id = ?", (cid,)
        )
        count = cursor.fetchone()[0]
        return count > 0

    @transaction
    def exists_ref_hid(
        self, hid: str, conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE ref_history_id = ?", (hid,)
        )
        count = cursor.fetchone()[0]
        return count > 0
    
    @transaction
    def mget_data(
        self, call_hids: List[str], conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the data of multiple `Call` objects given their history_ids,
        preserving order.
        """
        cursor = conn.execute(
            f"SELECT * FROM {self.table_name} WHERE call_history_id IN ({','.join('?' for _ in call_hids)})",
            call_hids,
        )
        rows = cursor.fetchall()
        call_data = {}
        for row in rows:
            hid = row[0]
            if hid not in call_data:
                call_data[hid] = {
                    "op_name": row[6], "cid": row[3], "hid": hid, "input_hids": {}, "output_hids": {}, "input_cids": {}, "output_cids": {}, 
                    "semantic_version": row[7], "content_version": row[8]
                    }
            if row[2] == "in":
                call_data[hid]["input_hids"][row[1]] = row[5]
                call_data[hid]["input_cids"][row[1]] = row[4]
            else:
                call_data[hid]["output_hids"][row[1]] = row[5]
                call_data[hid]["output_cids"][row[1]] = row[4]
        return [call_data[hid] for hid in call_hids]

    @transaction
    def get_data(
        self, call_history_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Any]:
        """
        Get the data of a `Call` object given its history_id.
        """
        return self.mget_data([call_history_id], conn=conn)[0]
    
    @transaction
    def get_data_content(
        self, cid: str, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Any]:
        cursor = conn.execute(
            f"SELECT * FROM {self.table_name} WHERE call_content_id = ?", (cid,)
        )
        rows = cursor.fetchall()
        hid = rows[0][0]
        return self.get_data(hid, conn)

    ### provenance queries
    @transaction
    def get_creator_hids(
        self, ref_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None
    ) -> Set[str]:
        # cursor = conn.execute(f"SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({','.join('?' for _ in hids)})", list(hids))
        cursor = conn.execute(
            f'SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({",".join("?" for _ in ref_hids)}) AND direction = "out"',
            list(ref_hids),
        )
        return set(row[0] for row in cursor.fetchall())

    @transaction
    def get_consumer_hids(
        self, ref_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None
    ) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({','.join('?' for _ in ref_hids)}) AND direction = 'in'",
            list(ref_hids),
        )
        return set(row[0] for row in cursor.fetchall())

    @transaction
    def get_input_hids(
        self, call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None
    ) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT ref_history_id FROM {self.table_name} WHERE call_history_id IN ({','.join('?' for _ in call_hids)}) AND direction = 'in'",
            list(call_hids),
        )
        return set(row[0] for row in cursor.fetchall())

    @transaction
    def get_output_hids(
        self, call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None
    ) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT ref_history_id FROM {self.table_name} WHERE call_history_id IN ({','.join('?' for _ in call_hids)}) AND direction = 'out'",
            list(call_hids),
        )
        return set(row[0] for row in cursor.fetchall())

    @transaction
    def get_dependencies(
        self,
        ref_hids: Iterable[str],
        call_hids: Iterable[str],
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[Set[str], Set[str]]:
        df = self.get_df(conn=conn)
        x = InMemCallStorage(df)
        return x.get_dependencies(ref_hids=ref_hids, call_hids=call_hids)

    @transaction
    def get_dependents(
        self,
        ref_hids: Iterable[str],
        call_hids: Iterable[str],
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[Set[str], Set[str]]:
        df = self.get_df(conn=conn)
        x = InMemCallStorage(df)
        return x.get_dependents(ref_hids=ref_hids, call_hids=call_hids)


class CachedCallStorage:
    """
    A cached version of the call storage that uses an in-memory storage as a
    cache, and can commit new data to a persistent storage.
    """

    def __init__(self, persistent: SQLiteCallStorage):
        self.persistent = persistent
        self.cache = InMemCallStorage()
        self.dirty_hids: Set[str] = set()

    def save(self, call: Call):
        self.cache.save(call)
        self.dirty_hids.add(call.hid)

    def drop(self, hid: str):
        self.cache.drop(hid)
        if hid in self.dirty_hids:
            self.dirty_hids.remove(hid) # when we `drop`, we forget this key ever existed

    def exists(self, call_history_id: str) -> bool:
        if self.cache.exists(call_history_id):
            return True
        else:
            res = self.persistent.exists(call_history_id)
            return res
    
    def exists_content(self, cid: str) -> bool:
        if self.cache.exists_content(cid):
            return True
        else:
            res = self.persistent.exists_ref_hid(cid)
            return res

    def get_data(
        self, call_history_id: str,
    ) -> Dict[str, Any]:
        if self.cache.exists(call_history_id):
            return self.cache.get_data(call_history_id)
        else:
            return self.persistent.get_data(call_history_id)
    
    def get_data_content(
        self, cid: str, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Any]:
        if self.cache.exists_content(cid):
            return self.cache.get_data_content(cid)
        else:
            return self.persistent.get_data_content(cid, conn)

    def get_creator_hids(self, hids: Iterable[str]) -> Set[str]:
        raise NotImplementedError()

    def get_consumer_hids(self, hids: Iterable[str]) -> Set[str]:
        raise NotImplementedError()

    def commit(self, conn: Optional[sqlite3.Connection] = None):
        if conn is None:
            conn = self.persistent.conn()
        # with conn:
        for hid in self.dirty_hids:
            self.persistent.save(self.cache.get_data(hid), conn=conn)
        self.dirty_hids.clear()
    
    def clear(self, allow_uncommited: bool = False):
        if len(self.dirty_hids) > 0 and not allow_uncommited:
            # see `CachedDictStorage.clear` for an explanation
            msg = "Cannot clear cache with uncommitted changes; call `commit()` first, or use `allow_uncommited=True`"
            raise ValueError(msg)
        self.cache = InMemCallStorage()
        self.dirty_hids.clear()
