from networkx.algorithms import topological_sort
from pandas.api.types import is_string_dtype
from sqlalchemy import MetaData, Table, Column, String, sql, LargeBinary
from sqlalchemy.engine.result import Result
from sqlalchemy.engine.base import Connection, Engine
from sqlalchemy.sql.selectable import Select
from sqlalchemy.sql.expression import select

from .psql_utils import get_db_root, get_connection_string
from .utils import transaction, get_metadata, get_engine, get_in_clause_rhs
from .psql_utils import fast_select, fast_insert, fast_upsert

from ..relations import RelSpec, RelStorage

from ...common_imports import *
from ...core.config import CoreConsts, CoreConfig, PSQLConfig
from ...util.shell_ut import ask

################################################################################
### Interfaces 
################################################################################
UID_COL = CoreConsts.UID_COL

class PSQLRelStorage(RelStorage):
    
    GRAPH_SCHEMA = 'graph'
    METADATA_SCHEMA = 'metadata'

    METADATA_TABLE = 'main'
    PROVENANCE_TABLE = 'provenance'
    METADATA_KEY = '0'
    
    def __init__(self, 
                 psql_config:PSQLConfig,
                 db_name:str=None,
                 autocommit:bool=False):
        self._db_name = self.generate_db_name() if db_name is None else db_name
        self.psql_config = psql_config
        self.autocommit = autocommit
        self.block_subtransactions = False
        self._sql_meta = None
        self._meta = None
        if not self.exists_db(db_name=self.db_name):
            logging.info('Creating storage...')
            self.create_db()
            self.init()
        self.reload_sql_meta()
        self.reload_meta()

    ############################################################################ 
    def get_engine(self) -> Engine:
        # get a DB sqlalchemy engine
        return get_engine(connection_string=self._connection_string)
    
    @property
    def db_name(self) -> str:
        return self._db_name
    
    @staticmethod
    def exists_db(db_name:str) -> bool:
        # whether the database for this storage exists 
        root_intf = get_db_root(CoreConfig.psql)
        return root_intf.exists_db(db_name=db_name)

    def create_db(self):
        # create the db for this storage
        root_intf = get_db_root(CoreConfig.psql)
        root_intf.create_db(db_name=self.db_name)
    
    @transaction()
    def describe(self, conn:Connection=None) -> TDict[str, TAny]:
        table_sizes = {t: self.get_count(table=t, conn=conn) for t in self.tables()}
        return {
            'table_sizes': table_sizes
        }
    
    @ask(question='Are you sure you want to drop this DB?', 
         desc_getter=lambda x: x.describe())
    def drop_db(self, must_exist:bool=False, kill_connections:bool=True, 
                answer:bool=None):
        self._sql_meta = None
        root_intf = get_db_root(CoreConfig.psql)
        if not root_intf.exists_db(db_name=self.db_name):
            if must_exist:
                raise ValueError()
            else:
                return
        try:
            root_intf.drop_db(db_name=self.db_name, kill_connections=False)
        except Exception as e:
            if kill_connections:
                logging.debug(f'Found active connections to DB {self.db_name}, killing...')
                root_intf.drop_db(db_name=self.db_name, kill_connections=True)
            else:
                raise e
    
    @ask(question='Are you sure you want to drop all rows from this DB?', 
         desc_getter=lambda x: x.describe())
    @transaction()
    def drop_all_rows(self, conn:Connection=None, answer:bool=None):
        tables = self.tables(return_qual_names=False)
        ord_tables = self.topsort_relnames(forward=False, names=tables, conn=conn)
        for t in ord_tables:
            logging.debug(f'Dropping all rows from table {t}...')
            tableobj = self.get_tableobj(name=t)
            query = sql.delete(tableobj)
            conn.execute(query)

    @transaction()
    def init_schema(self, conn:Connection=None):
        # create initial schemas and tables
        self.create_schema(name=self.GRAPH_SCHEMA, conn=conn)
        self.create_schema(name=self.METADATA_SCHEMA, conn=conn)
        # works as dict
        _ = Table(self.METADATA_TABLE, self.sql_meta,
                  Column('key', String(), primary_key=True),
                  Column('value', LargeBinary()),
                  schema=self.METADATA_SCHEMA)
        self.sql_meta.create_all(bind=conn)

    @property
    def sql_meta(self) -> MetaData:
        return self._sql_meta

    @transaction()
    def reload_sql_meta(self, conn:Connection=None):
        meta = get_metadata(conn=conn)
        insp = sqlalchemy.inspect(conn)
        schemas = insp.get_schema_names()
        for schema in schemas:
            meta.reflect(schema=schema)
        self._sql_meta = meta
    
    @transaction()
    def init(self, conn:Connection=None):
        self.reload_sql_meta(conn=conn)
        self.init_schema(conn=conn)
        self.init_meta(value=None, conn=conn)

    ############################################################################ 
    @property
    def meta(self) -> TAny:
        return self._meta

    @property
    def metadata_table(self) -> Table:
        return self[self.METADATA_SCHEMA, self.METADATA_TABLE]

    @transaction()
    def init_meta(self, value:TAny, conn:Connection=None):
        conn.execute(sql.insert(self.metadata_table).
                     values(key=self.METADATA_KEY, value=pickle.dumps(value)))
        
    @transaction()
    def update_meta(self, value:TAny, conn:Connection=None):
        t = self.metadata_table
        conn.execute(sql.update(t).
                     where(t.c['key'] == self.METADATA_KEY).
                     values(value=pickle.dumps(value)))
        self.reload_meta(conn=conn)
        
    @transaction()
    def reload_meta(self, conn:Connection=None):
        series = self.read(self.metadata_table, 
                           conn=conn).set_index('key')['value']
        self._meta = pickle.loads(series[self.METADATA_KEY])
    
    @transaction()
    def dump_meta(self, conn:Connection=None):
        self.update_meta(value=self.meta, conn=conn)
    
    ############################################################################ 
    ### schema elements
    ############################################################################ 
    @transaction()
    def create_relation(self, name:str, rel_spec:RelSpec, 
                        allow_exist:bool=False, conn:Connection=None):
        if self.has_qtable(qtable=self.get_qtable(name=name), conn=conn):
            if allow_exist:
                return
            else:
                raise ValueError()
        table_obj = rel_spec.make_table(name=name, db_meta=self.sql_meta,
                                        schema=self.GRAPH_SCHEMA)
        table_obj.create(bind=conn)
        self.reload_sql_meta(conn=conn)
    
    @transaction()
    def drop_relation(self, name:str, conn:Connection=None):
        qtable = self.get_qtable(name=name)
        # do we really need to drop the columns first?
        out_arrows = self.get_out_fks(qtable=qtable, return_qual_names=True)
        for sk, table, _, _ in out_arrows:
            self.drop_column(qtable=sk, name=table, conn=conn)
        self.drop_qtable(qtable=qtable, conn=conn)
        self.reload_sql_meta(conn=conn)
    
    ############################################################################ 
    ### instance elements
    ############################################################################ 
    def postprocess_df(self, df:pd.DataFrame) -> pd.DataFrame:
        for col, dtype in df.dtypes.items():
            if not is_string_dtype(dtype):
                df[col] = df[col].astype(str)
        return df

    @transaction()
    def fast_select(self, query:TUnion[str, Select],
                    conn:Connection=None) -> pd.DataFrame:
        return self.postprocess_df(fast_select(query=query, qual_table=None,
                                               index_col=None, cols=None,
                                               conn=conn))

    @transaction()
    def fast_read(self, name:str, cols:TList[str]=None, index:pd.Index=None,
                  conn:Connection=None) -> pd.DataFrame:
        tableobj = self.get_tableobj(name=name)
        if cols is None:
            query = tableobj.select()
        else:
            query = select(tableobj.c[col] for col in cols)
        if index is not None:
            query = query.where(tableobj.c[UID_COL].in_(get_in_clause_rhs(index)))
        return self.fast_select(query=query, conn=conn)
    
    @transaction()
    def fast_insert(self, name:str, df:pd.DataFrame, conn:Connection=None):
        if df.empty: # engine complains
            return
        fast_insert(df=df, qual_table=self.get_qtable(name=name), conn=conn,
                    include_index=False)
    
    @transaction()
    def mfast_insert(self, relations_dict:TDict[str, pd.DataFrame],
                     conn:Connection=None):
        relations_topsort = self.topsort_relnames(names=relations_dict.keys(),
                                                  conn=conn)
        for relname in relations_topsort:
            self.fast_insert(name=relname, df=relations_dict[relname], 
                             conn=conn)
    
    @transaction()
    def mfast_upsert(self, relations_dict:TDict[str, pd.DataFrame],
                     conn:Connection=None):
        relations_topsort = self.topsort_relnames(names=relations_dict.keys(),
                                                  conn=conn)
        for relname in relations_topsort:
            self.fast_upsert(name=relname, df=relations_dict[relname],
                             conn=conn)
    
    @transaction()
    def fast_upsert(self, name:str, df:pd.DataFrame, conn:Connection=None):
        if df.empty: # engine complains
            return
        qtable = self.get_qtable(name=name)
        fast_upsert(df=df, qual_table=qtable,
                    index_cols=self.get_index_cols(qtable=qtable), 
                    include_index=False, conn=conn)
    
    @transaction()
    def read(self, qtable:TUnion[str, Table], index:pd.Index=None,
             set_index:bool=False, conn:Connection=None) -> pd.DataFrame:
        if isinstance(qtable, str):
            table_obj = self[qtable]
        elif isinstance(qtable, Table):
            table_obj = qtable
        else:
            raise ValueError()
        query = table_obj.select()
        if index is not None:
            in_clause = get_in_clause_rhs(index_like=index)
            query = query.where(table_obj.c[Consts.index_col].in_(in_clause))
        result = self.read_rp(conn.execute(query))
        if set_index:
            primary_key_columns = table_obj.primary_key.columns.values()
            assert len(primary_key_columns) == 1
            primary_key = primary_key_columns[0].name
            result = result.set_index(primary_key)
        return self.postprocess_df(result)

    def read_rp(self, rp:Result) -> pd.DataFrame:
        df = pd.DataFrame(rp.fetchall(), columns=rp.keys())
        return self.postprocess_df(df)

    def delete_rows(self, name:str, index:TList[str], index_col:str, 
                    conn:Connection=None):
        tableobj = self.get_tableobj(name=name)
        query = (sql.delete(tableobj).
                 where(tableobj.c[index_col].  in_(get_in_clause_rhs(index))))
        conn.execute(query)

    ############################################################################ 
    ### low-level interface
    ############################################################################ 
    ### schemas 
    @transaction()
    def create_schema(self, name:str, conn:Connection=None):
        if not conn.dialect.has_schema(conn, name):
            conn.execute(sqlalchemy.schema.CreateSchema(name=name))
        
    ### tables
    def get_qtable(self, name:str) -> str:
        return f'{self.GRAPH_SCHEMA}.{name}'

    def get_tableobj(self, name:str) -> Table:
        return self[self.get_qtable(name=name)]

    @transaction()
    def has_table(self, name: str, conn: Connection = None) -> bool:
        return self.has_qtable(qtable=self.get_qtable(name=name), conn=conn)

    def has_column(self, name: str, col: str):
        tableobj = self.get_tableobj(name=name)
        return col in [elt.name for elt in tableobj.columns]

    @transaction()
    def has_qtable(self, qtable:str, conn:Connection=None) -> bool:
        schema, table_name = qtable.split('.')
        engine = self.get_engine()
        return engine.dialect.has_table(conn, table_name, schema=schema)
    
    @transaction()
    def drop_qtable(self, qtable:str, conn:Connection=None):
        schema, table = qtable.split('.')
        logging.debug(f'Dropping table {qtable}...')
        query = f'DROP TABLE "{schema}"."{table}"'
        conn.execute(query)

    @transaction()
    def get_count(self, table:str, conn:Connection=None) -> int:
        qual_table = self.get_qtable(name=table)
        query = f'SELECT COUNT(*) FROM {qual_table};'
        result = self.read_rp(conn.execute(query))
        return result['count'].item()
    
    def tables(self, return_qual_names:bool=False) -> TList[str]:
        qual_tables = [table for table in self.sql_meta.tables.keys()
                if table.split('.')[0] == self.GRAPH_SCHEMA]
        if return_qual_names:
            return qual_tables
        else:
            return [qual_table.split('.')[1] for qual_table in qual_tables]
        
    def get_nx_graph(self) -> nx.MultiDiGraph:
        """
        Return a graph over relation *names* (unqualified tables).
        
        Keys in the graph are (source column, target column) pairs.
        """
        res = nx.MultiDiGraph()
        all_arrows = []
        for qtable in self.tables(return_qual_names=True):
            all_arrows += self.get_out_fks(qtable=qtable, return_qual_names=False)
        for s_table, s_col, t_table, t_col in all_arrows:
            res.add_node(s_table)
            res.add_node(t_table)
            res.add_edge(s_table, t_table, key=(s_col, t_col))
        return res
    
    @transaction()
    def topsort_relnames(self, names:TIter[str], forward:bool=True,
                         conn:Connection=None) -> TList[str]:
        """
        Args:
            forward (bool, optional): If True, return names starting from tables
            without out-edges; if False, reverse order.
        """
        self.reload_meta(conn=conn)
        self.reload_sql_meta(conn=conn)
        nx_graph = self.get_nx_graph()
        # check that the relation names are actually present
        existing_names = set(nx_graph.nodes)
        if not set(names).issubset(existing_names):
            raise ValueError(f'Got relation names {set.difference(set(names), existing_names)} not in relation storage')
        nx_graph = nx_graph.subgraph(nodes=list(names))
        res = list(topological_sort(G=nx_graph))
        if forward:
            return res[::-1]
        else:
            return res

    ### columns
    @transaction()
    def get_columns(self, qtable: str, conn: Connection = None) -> TList[str]:
        query = f'SELECT * FROM {qtable} WHERE false'
        result = self.fast_select(query=query, conn=conn)
        return result.columns.values.tolist()
    
    @transaction()
    def create_column(self, qtable: str, name: str, dtype: str,
                      with_default: bool = False, default_value: TAny = None,
                      fk_qtable:str=None, fk_col:str=None,
                      conn: Connection = None):
        schema, table = qtable.split('.')
        quoted_table = f'"{schema}"."{table}"'
        if fk_qtable is not None:
            assert fk_col is not None
            fk_name = f'{fk_qtable}.{fk_col}'.replace('.', '_')
            fk_line = f'CONSTRAINT {fk_name} REFERENCES {fk_qtable} ("{fk_col}")'
        else:
            fk_line = ''
        if with_default:
            #! in psql, single quotes are used to indicate string literals
            default_line = f"DEFAULT '{default_value}'"
        else:
            default_line = ''
        query = f"""
        ALTER TABLE {quoted_table}
            ADD COLUMN "{name}" {dtype}
            {fk_line}
            {default_line}
        """
        conn.execute(query)
        self.reload_sql_meta(conn=conn) #! important

    @transaction()
    def drop_column(self, qtable:str, name:str, conn:Connection = None):
        schema, table = qtable.split('.')
        logging.debug(f'Dropping column {name} from table {qtable}...')
        query = f'ALTER TABLE "{schema}"."{table}" DROP COLUMN "{name}"'
        conn.execute(query)
    
    @transaction()
    def create_fk(self, s_qtable: str, s_col: str, t_qtable: str, t_col: str,
                  fk_name: str=None, deferrable: bool = False,
                  initially_deferred: bool = False,
                  cascade_delete: bool = False, conn: Connection = None):
        if fk_name is None:
            fk_name = f'{s_qtable}_{s_col}_{t_qtable}_{t_col}'
        s_schema, s_table = s_qtable.split('.')
        t_schema, t_table = t_qtable.split('.')
        s_quoted_table = f'"{s_schema}"."{s_table}"'
        t_quoted_table = f'"{t_schema}"."{t_table}"'
        if not deferrable:
            defer = ''
        else:
            if initially_deferred:
                defer = 'DEFERRABLE INITIALLY DEFERRED'
            else:
                defer = 'DEFERRABLE INITIALLY IMMEDIATE'
        if cascade_delete:
            on_delete = 'ON DELETE cascade'
        else:
            on_delete = ''
        query = f"""
        ALTER TABLE {s_quoted_table}
            ADD CONSTRAINT "{fk_name}"
            FOREIGN KEY ("{s_col}")
            REFERENCES {t_quoted_table}("{t_col}")
            {on_delete}
            {defer};
        """
        conn.execute(query)

    def get_out_fks(self, qtable:str, 
                    return_qual_names:bool=True,
                    ) -> TSet[TTuple[str, str, str, str]]:
        table_obj = self[qtable]
        foreign_keys = table_obj.foreign_key_constraints
        results = set()
        for fk in foreign_keys:
            sk_qual = qtable
            sk = sk_qual if return_qual_names else sk_qual.split('.')[1]
            tk_qual = self.get_qtable(fk.referred_table.name)
            tk = tk_qual if return_qual_names else tk_qual.split('.')[1]
            col = fk.column_keys[0]
            t_col = fk.elements[0].column.name
            results.add((sk, col, tk, t_col))
        return results

    def get_index_cols(self, qtable:str) -> TList[str]:
        T = self[qtable]
        indexes = list(T.indexes)
        primary_key = T.primary_key
        if len(indexes) > 1:
            raise NotImplementedError()
        elif len(indexes) == 1:
            index = indexes[0]
            index_cols = [col.name for col in index.columns]
        else:
            if len(primary_key.columns) == 1:
                index_cols = [primary_key.columns[0].name]
            elif len(primary_key.columns) == 0:
                index_cols = []
            else:
                raise NotImplementedError()
        return index_cols
    
    ############################################################################ 
    def __getitem__(self, qtable_id:TUnion[str, TTuple[str, str]]) -> Table:
        if isinstance(qtable_id, tuple):
            schema, table = qtable_id
            qtable_id = f'{schema}.{table}'
        return self.sql_meta.tables[qtable_id]

    @property
    def _connection_string(self) -> str:
        # get a connection string from DB settings 
        return get_connection_string(db_name=self.db_name,
                                     user=self.psql_config.user,
                                     password=self.psql_config.password,
                                     host=self.psql_config.host, 
                                     port=self.psql_config.port)