from abc import abstractmethod, ABC

from sqlalchemy import MetaData, Table, Column, Index
from sqlalchemy.engine.result import Result
from sqlalchemy.engine.base import Connection, Engine
from sqlalchemy.sql.selectable import Select, CompoundSelect

from ..common_imports import *
from ..core.config import CoreConsts
from ..util.common_ut import get_uid

UID_COL = CoreConsts.UID_COL

class RelSpec(object):

    def __init__(self, col_objs:TList[Column], indices:TList[TList[str]]=None,
                 extend_existing:bool=True):
        """
        Define mini-specification for a table in a database that works for our
        uses. 

        Args:
            col_objs (TList[Column]): list of sqlalchemy Column objects to use
            as columns
            indices (TList[TList[str]], optional): List of multi-column indices,
            if any
        """
        self.col_objs = col_objs
        self.indices = [] if indices is None else indices
        self.extend_existing = extend_existing
        
    def make_table(self, name:str, db_meta:MetaData, schema:str) -> Table:
        extra_args = []
        for index_cols in self.indices:
            # note that index names must be unique DB-wide
            extra_args.append(Index(f'multicol_{name}', *index_cols, unique=True))
        table = Table(name, db_meta, *self.col_objs, *extra_args, 
                      extend_existing=self.extend_existing, schema=schema)
        return table


class RelStorage(ABC):
    DUMMY_CONN = object()

    ############################################################################ 
    ### high-level helpers
    ############################################################################ 
    @abstractmethod
    def get_engine(self) -> Engine:
        raise NotImplementedError()

    @staticmethod
    def generate_db_name() -> str:
        return f'v{5}_{get_uid()}'

    @staticmethod
    @abstractmethod
    def exists_db(db_name:str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def create_db(self):
        raise NotImplementedError()
    
    @abstractmethod
    def describe(self, conn:Connection=None) -> TDict[str, TAny]:
        raise NotImplementedError()
    
    @abstractmethod
    def drop_db(self, must_exist:bool=False, 
                kill_connections:bool=True, answer:bool=None):
        raise NotImplementedError()
    
    @abstractmethod
    def drop_all_rows(self, conn:Connection=None, answer:bool=None):
        raise NotImplementedError()

    @abstractmethod
    def init_schema(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def sql_meta(self) -> MetaData:
        raise NotImplementedError()

    @abstractmethod
    def reload_sql_meta(self, conn:Connection=None):
        raise NotImplementedError()

    @abstractmethod
    def init(self):
        raise NotImplementedError()
    
    ############################################################################ 
    ### working with the metadata object
    ############################################################################ 
    @property
    def meta(self) -> TAny:
        raise NotImplementedError()

    @property
    @abstractmethod
    def metadata_table(self) -> Table:
        raise NotImplementedError()

    @abstractmethod
    def init_meta(self, value:TAny, conn:Connection=None):
        raise NotImplementedError()
        
    @abstractmethod
    def update_meta(self, value:TAny, conn:Connection=None):
        raise NotImplementedError()
        
    @abstractmethod
    def reload_meta(self, conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def dump_meta(self, conn:Connection=None):
        raise NotImplementedError()

    ############################################################################ 
    ### schema elements
    ############################################################################ 
    @abstractmethod
    def create_relation(self, name:str, rel_spec:RelSpec,
                        allow_exist:bool=False, conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def drop_relation(self, name:str, conn:Connection=None):
        raise NotImplementedError()
    
    ############################################################################ 
    ### instance elements
    ############################################################################ 
    @abstractmethod
    def fast_select(self, query:TUnion[str, Select, CompoundSelect], 
                    conn:Connection=None) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def fast_read(self, name:str, cols:TList[str]=None, 
                  index:pd.Index=None, conn:Connection=None) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def fast_insert(self, name:str, df:pd.DataFrame, conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def fast_upsert(self, name:str, df:pd.DataFrame, conn:Connection=None):
        raise NotImplementedError()

    @abstractmethod
    def read(self, qual_table:TUnion[str, Table], index:pd.Index=None,
             set_index:bool=False, conn:Connection=None) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def read_rp(self, rp:Result) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def delete_rows(self, name:str, index:TList[str], index_col:str,
                    conn:Connection=None):
        raise NotImplementedError()

    ### mmethods
    @abstractmethod
    def mfast_insert(self, relations_dict:TDict[str, pd.DataFrame], 
                     conn:Connection=None):
        raise NotImplementedError()

    @abstractmethod
    def mfast_upsert(self, relations_dict:TDict[str, pd.DataFrame], 
                     conn:Connection=None):
        raise NotImplementedError()

    ############################################################################ 
    """
    NOTE some conventions:
        - name stands for an unqualified name in the main schema
        - qtable stands for a qualified table name (schema.table)
    """
    ### schemas 
    @abstractmethod
    def create_schema(self, name:str, conn:Connection=None):
        raise NotImplementedError()
        
    ### tables
    @abstractmethod
    def get_qtable(self, name:str) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_tableobj(self, name:str) -> Table:
        raise NotImplementedError()
    
    @abstractmethod
    def has_table(self, name:str, conn:Connection=None) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def has_column(self, name:str, col:str):
        raise NotImplementedError()
    
    @abstractmethod
    def has_qtable(self, qtable:str, conn:Connection=None) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def drop_qtable(self, qtable:str, conn:Connection=None):
        raise NotImplementedError()

    @abstractmethod
    def get_count(self, table:str, conn:Connection=None) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def tables(self, return_qual_names:bool=False) -> TList[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def topsort_relnames(self, forward:bool=True, names:TIter[str]=None, 
                         conn:Connection=None) -> TList[str]:
        raise NotImplementedError()

    ### columns
    @abstractmethod
    def get_columns(self, qtable:str, conn:Connection=None) -> TList[str]:
        """
        Get the names of the columns of this table.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def create_column(self, qtable:str, name:str, dtype:str,
                      with_default:bool=False, default_value:TAny=None,
                      fk_qtable:str=None, fk_col:str=None,
                      conn:Connection=None):
        """
        NOTE: The interface for creating a column is bundled with the one for
        creating a foreign key on this column, since some dialects don't allow
        you to put a FK on an existing table.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def drop_column(self, qtable:str, name:str, conn:Connection=None):
        raise NotImplementedError()

    @abstractmethod
    def get_index_cols(self, qtable:str) -> TList[str]:
        raise NotImplementedError()
    
    ### foreign keys
    @abstractmethod
    def create_fk(self, s_qtable:str, s_col:str, t_qtable:str, 
                   t_col:str, fk_name:str, deferrable:bool=False,
                   initially_deferred:bool=False,
                   cascade_delete:bool=False, conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def get_out_fks(self, qtable:str, return_qual_names:bool=True) -> TSet[TTuple[str, str, str, str]]:
        """
        Return (source table, source column, target table, target column)
        tuples for foreign keys based on metadata
        """
        raise NotImplementedError()

    @abstractmethod
    def get_nx_graph(self) -> nx.MultiDiGraph:
        raise NotImplementedError()

    def __getitem__(self, table_id:TUnion[str, TTuple[str, str]]) -> Table:
        raise NotImplementedError()