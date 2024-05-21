from common_imports import *
from utils import serialize, deserialize
from model import Call
import joblib
import sqlite3
from abc import ABC, abstractmethod


def transaction(method): # transaction decorator for classes with a `get_conn` method
    def wrapper(self, *args, **kwargs):
        if kwargs.get("conn") is not None: # already in a transaction
            logging.debug("Folding into existing transaction")
            return method(self, *args, **kwargs)
        else: # open a connection
            logging.debug(f"Opening new transaction from {self.__class__.__name__}.{method.__name__}")
            conn = self.get_conn()
            try:
                res = method(self, *args, conn=conn, **kwargs)
                conn.commit()
                return res
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
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
    def exists(self, key: str) -> bool:
        pass

    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.exists(key)


class SQLiteDictStorage(DictStorage):
    def __init__(self, db_path: str, table: str):
        self.db_path = db_path
        self.table = table
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (key TEXT PRIMARY KEY, value BLOB)")
        # close the connection, we will open it when we need it
        self.conn.close()
    
    def get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    @transaction
    def get(self, key: str, conn: Optional[sqlite3.Connection] = None) -> Any:
        cursor = conn.execute(f"SELECT value FROM {self.table} WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError(f"Key {key} not found")
        return deserialize(result[0])

    @transaction
    def set(self, key: str, value: Any, conn: Optional[sqlite3.Connection] = None) -> None:
        conn.execute(f"INSERT OR REPLACE INTO {self.table} (key, value) VALUES (?, ?)", (key, serialize(value)))

    @transaction
    def drop(self, key: str, conn: Optional[sqlite3.Connection] = None) -> None:
        conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
    
    @transaction
    def exists(self, key: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE key = ?", (key,))
        count = cursor.fetchone()[0]
        return count > 0
    
    @transaction
    def keys(self, conn: Optional[sqlite3.Connection] = None) -> List[str]:
        cursor = conn.execute(f"SELECT key FROM {self.table}")
        return [row[0] for row in cursor.fetchall()]
    
    @transaction
    def values(self, conn: Optional[sqlite3.Connection] = None) -> List[Any]:
        cursor = conn.execute(f"SELECT value FROM {self.table}")
        return [deserialize(row[0]) for row in cursor.fetchall()]


class CachedDictStorage(DictStorage):
    def __init__(self, persistent: DictStorage):
        self.persistent = persistent
        self.cache: Dict[str, Any] = {}
        self.dirty_keys: Set[str] = set()
    
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
        for key in self.dirty_keys:
            self.persistent.set(key, self.cache[key], conn=conn)
        self.dirty_keys.clear()
    
    def drop(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
        self.dirty_keys.add(key)
        self.persistent.drop(key)
    
    def exists(self, key: str) -> bool:
        if key in self.cache:
            return True
        else:
            res = self.persistent.exists(key)
            return res



class InMemCallStorage:
    COLUMNS = ["call_history_id", "name", "direction", "call_content_id", "ref_content_id", "ref_history_id", "op"]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(columns=InMemCallStorage.COLUMNS).set_index(["call_history_id", "name"])
        
    def save(self, call: Call):
        if call.hid in self.df.index.levels[0]:
            return
        for k, v in call.inputs.items():
            self.df.loc[(call.hid, k), :] = ("in", call.cid, v.cid, v.hid, call.op.name)
        for k, v in call.outputs.items():
            self.df.loc[(call.hid, k), :] = ("out", call.cid, v.cid, v.hid, call.op.name)
    
    def drop(self, hid: str):
        if hid not in self.df.index.levels[0]:
            raise ValueError(f"Call with history_id {hid} does not exist")
        self.df.drop(index=hid, level=0, inplace=True)

    def exists(self, call_history_id: str) -> bool:
        return call_history_id in self.df.index.levels[0]
    
    def get_data(self, call_history_id: str) -> Dict[str, Any]:
        """
        Get all the stuff associated with a call apart from the op.
        """
        if not self.exists(call_history_id):
            raise ValueError(f"Call with history_id {call_history_id} does not exist")
        rows = self.df.loc[call_history_id].reset_index().to_dict(orient="records")
        input_hids, output_hids = {}, {}
        input_cids, output_cids = {}, {}
        for row in rows:
            if row["direction"] == "in":
                input_hids[row["name"]] = row["ref_history_id"]
                input_cids[row["name"]] = row["ref_content_id"]
            else:
                output_hids[row["name"]] = row["ref_history_id"]
                output_cids[row["name"]] = row["ref_content_id"]
        # return Call(op=op, cid=rows[0]["call_content_id"], hid=call_history_id, inputs=inputs, outputs=outputs)
        op_name = rows[0]["op"]
        return {
            "op_name": op_name,
            "cid": rows[0]["call_content_id"],
            "hid": call_history_id,
            "input_hids": input_hids,
            "output_hids": output_hids,
            "input_cids": input_cids,
            "output_cids": output_cids
        }

    def get_creator_hids(self, ref_hids: Iterable[str]) -> Set[str]:
        #! slow
        call_history_ids = self.df.query('ref_history_id in @ref_hids and direction == "out"').index.get_level_values(0).unique()
        return set(call_history_ids)
    
    def get_consumer_hids(self, ref_hids: Iterable[str]) -> Set[str]:
        #! slow
        call_history_ids = self.df.query('ref_history_id in @ref_hids and direction == "in"').index.get_level_values(0).unique()
        return set(call_history_ids)
    
    def get_input_hids(self, call_hids: Iterable[str]) -> Set[str]:
        ref_hids = self.df.query('call_history_id in @call_hids and direction == "in"').ref_history_id.unique()
        return set(ref_hids)
    
    def get_output_hids(self, call_hids: Iterable[str]) -> Set[str]:
        ref_hids = self.df.query('call_history_id in @call_hids and direction == "out"').ref_history_id.unique()
        return set(ref_hids)
    
    def get_dependencies(self, ref_hids: Iterable[str], call_hids: Iterable[str]) -> Tuple[Set[str], Set[str]]:
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
    
    def get_dependents(self, ref_hids: Iterable[str], call_hids: Iterable[str]) -> Tuple[Set[str], Set[str]]:
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
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        # if it doesn't exist, create a table with a two-column primary key
        # on call_history_id and name
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (call_history_id TEXT, name TEXT, direction TEXT, "
                          "call_content_id TEXT, ref_content_id TEXT, ref_history_id TEXT, op TEXT, PRIMARY KEY (call_history_id, name))")
        # close the connection, we will open it when we need it
        self.conn.close()
    
    @transaction
    def get_df(self, conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {self.table_name}", conn).set_index(["call_history_id", "name"])
    
    @transaction
    def execute_df(self, query: str, conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
        return pd.read_sql(query, conn)
    
    def get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    @transaction
    def save(self, call_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None):
        op_name = call_data["op_name"]
        for k in call_data["input_hids"]:
            hid = call_data["input_hids"][k]
            cid = call_data["input_cids"][k]
            conn.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?, ?)", (call_data["hid"], k, "in", call_data["cid"], cid, hid, op_name))
        for k in call_data["output_hids"]:
            hid = call_data["output_hids"][k]
            cid = call_data["output_cids"][k]
            conn.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?, ?, ?, ?)", (call_data["hid"], k, "out", call_data["cid"], cid, hid, op_name))

    @transaction
    def drop(self, hid: str, conn: Optional[sqlite3.Connection] = None):
        conn.execute(f"DELETE FROM {self.table_name} WHERE call_history_id = ?", (hid,))

    @transaction
    def exists(self, call_history_id: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE call_history_id = ?", (call_history_id,))
        count = cursor.fetchone()[0]
        return count > 0
    
    @transaction
    def exists_ref_hid(self, hid: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE ref_history_id = ?", (hid,))
        count = cursor.fetchone()[0]
        return count > 0
    
    @transaction
    def get_data(self, call_history_id: str, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        cursor = conn.execute(f"SELECT * FROM {self.table_name} WHERE call_history_id = ?", (call_history_id,))
        rows = cursor.fetchall()
        input_hids, output_hids = {}, {}
        input_cids, output_cids = {}, {}
        op_name = None
        for row in rows:
            if op_name is None:
                op_name = row[6]
            if row[2] == "in":
                input_hids[row[1]] = row[5]
                input_cids[row[1]] = row[4]
            else:
                output_hids[row[1]] = row[5]
                output_cids[row[1]] = row[4]
        return {
            "op_name": op_name,
            "cid": rows[0][3],
            "hid": call_history_id,
            "input_hids": input_hids,
            "output_hids": output_hids,
            "input_cids": input_cids,
            "output_cids": output_cids
        }
    
    ### provenance queries
    @transaction
    def get_creator_hids(self, ref_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Set[str]:
        # cursor = conn.execute(f"SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({','.join('?' for _ in hids)})", list(hids))
        cursor = conn.execute(
            f'SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({",".join("?" for _ in ref_hids)}) AND direction = "out"',
            list(ref_hids)
        )
        return set(row[0] for row in cursor.fetchall())
    
    @transaction
    def get_consumer_hids(self, ref_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT call_history_id FROM {self.table_name} WHERE ref_history_id IN ({','.join('?' for _ in ref_hids)}) AND direction = 'in'",
              list(ref_hids))
        return set(row[0] for row in cursor.fetchall())
    
    @transaction
    def get_input_hids(self, call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT ref_history_id FROM {self.table_name} WHERE call_history_id IN ({','.join('?' for _ in call_hids)}) AND direction = 'in'",
              list(call_hids))
        return set(row[0] for row in cursor.fetchall())
    
    @transaction
    def get_output_hids(self, call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Set[str]:
        cursor = conn.execute(
            f"SELECT DISTINCT ref_history_id FROM {self.table_name} WHERE call_history_id IN ({','.join('?' for _ in call_hids)}) AND direction = 'out'",
              list(call_hids))
        return set(row[0] for row in cursor.fetchall())
    
    @transaction
    def get_dependencies(self, ref_hids: Iterable[str], call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Tuple[Set[str], Set[str]]:
        df = self.get_df(conn=conn)
        x = InMemCallStorage(df)
        return x.get_dependencies(ref_hids=ref_hids, call_hids=call_hids)
    
    @transaction
    def get_dependents(self, ref_hids: Iterable[str], call_hids: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> Tuple[Set[str], Set[str]]:
        df = self.get_df(conn=conn)
        x = InMemCallStorage(df)
        return x.get_dependents(ref_hids=ref_hids, call_hids=call_hids)
    

    


class CachedCallStorage:
    def __init__(self, persistent: SQLiteCallStorage):
        self.persistent = persistent
        self.cache = InMemCallStorage()
        self.dirty_hids: Set[str] = set()
    
    def save(self, call: Call):
        self.cache.save(call)
        self.dirty_hids.add(call.hid)

    def drop(self, hid: str):
        self.cache.drop(hid)
        self.dirty_hids.add(hid)

    def exists(self, call_history_id: str) -> bool:
        if self.cache.exists(call_history_id):
            return True
        else:
            res = self.persistent.exists(call_history_id)
            return res
    
    def get_data(self, call_history_id: str, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        if self.cache.exists(call_history_id):
            return self.cache.get_data(call_history_id)
        else:
            if conn is None: conn = self.persistent.get_conn()
            with conn:
                return self.persistent.get_data(call_history_id, conn)
        
    def get_creator_hids(self, hids: Iterable[str]) -> Set[str]:
        raise NotImplementedError()
    
    def get_consumer_hids(self, hids: Iterable[str]) -> Set[str]:
        raise NotImplementedError()
    
    def commit(self, conn: Optional[sqlite3.Connection] = None):
        if conn is None:
            conn = self.persistent.get_conn()
        with conn:
            for hid in self.dirty_hids:
                self.persistent.save(self.cache.get_data(hid), conn=conn)
        self.dirty_hids.clear()