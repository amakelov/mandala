from sqlalchemy.engine.base import Connection
from sqlalchemy.sql.selectable import Select, CompoundSelect
from sqlalchemy.engine.result import Result
from sqlalchemy.dialects import postgresql

from .utils import transaction

from ...common_imports import *
from ...util.common_ut import get_uid
from ...core.config import PSQLConfig


################################################################################
### helper functions
################################################################################
def get_db_root(psql_config:PSQLConfig):
    return PSQLInterface(autocommit=True, psql_config=psql_config)

def get_connection_string(db_name:str, user:str, password:str,
                          host:str, port:int) -> str:
    return f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}'

################################################################################
### interface to postgres
################################################################################
class PSQLInterface(object):
    # to connect to a postgres server

    def __init__(self, psql_config:PSQLConfig, autocommit:bool=False,):
        self.host = psql_config.host
        self.user = psql_config.user
        self.port = psql_config.port
        self.password = psql_config.password
        self.root_db_name = psql_config.root_db_name
        self.autocommit = autocommit
        self.connection_string = f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.root_db_name}'
        self.engine = self.get_engine()
        self.block_subtransactions = False
    
    ############################################################################ 
    ### utils
    ############################################################################ 
    def get_raw_conn(self, autocommit:bool=False) -> Connection:
        conn = psycopg2.connect(host=self.host, 
                                user=self.user,
                                password=self.password,
                                database=self.root_db_name)
        conn.autocommit = autocommit
        return conn
    
    def get_engine(self) -> TAny:
        """
        you might think that setting autocommit=True in conn.execution_options
        would be enough to tell sqlalchemy to use autocommit mode, however
        that's not the case; we must do it at the level of engine creation. See
        [this](https://www.oddbird.net/2014/06/14/sqlalchemy-postgres-autocommit/)
        for more details
        """
        if self.autocommit:
            engine = sqlalchemy.create_engine(self.connection_string, 
                                            isolation_level='AUTOCOMMIT', 
                                            connect_args={'connect_timeout': 5})
        else:
            engine = sqlalchemy.create_engine(self.connection_string, 
                                              connect_args={'connect_timeout': 5})
        return engine
        
    def read_rp(self, rp:Result) -> pd.DataFrame:
        df = pd.DataFrame(rp.fetchall(), columns=rp.keys())
        return df

    @transaction()
    def read(self, query:str, index_col:str=None,
             conn:Connection=None) -> pd.DataFrame:
        rp = conn.execute(query)
        df = self.read_rp(rp=rp)
        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df

    ############################################################################ 
    ### managing databases
    ############################################################################ 
    @staticmethod
    def _is_unnamed_db(db_name:str) -> bool:
        return db_name.startswith('v5_') and len(db_name) == 35
    
    @transaction()
    def create_db(self, db_name:str, conn:Connection=None):
        logging.debug('Creating database {}...'.format(db_name))
        conn.execute('CREATE DATABASE {}'.format(db_name))  
    
    @transaction()
    def kill_active_connections(self, db_name:str, conn:Connection=None):
        query = f"""
        SELECT 
            pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE
            pg_stat_activity.datname = '{db_name}'
        AND pid <> pg_backend_pid()       
        """
        conn.execute(query)
    
    @transaction(retry=False) #! not retriable because it is itself used in retry loop
    def drop_db(self, db_name:str, must_exist:bool=True, 
                kill_connections:bool=False, conn:Connection=None):
        assert db_name not in ('postgres', 'template0', 'template1')
        logging.info('Dropping database {}...'.format(db_name))
        if kill_connections:
            self.kill_active_connections(db_name=db_name, conn=conn)
        if must_exist:
            conn.execute('DROP DATABASE {}'.format(db_name))  
        else:
            conn.execute(f'DROP DATABASE IF EXISTS {db_name}')
    
    def get_all_dbs(self) -> TSet[str]:
        df = self.read(query='SELECT datname FROM pg_database')
        return set(df.datname.values)
    
    def exists_db(self, db_name:str) -> bool:
        return db_name in self.get_all_dbs()

    @transaction()
    def _drop_unnamed(self, conn:Connection=None):
        #! NEVER CALL THIS
        all_dbs = self.get_all_dbs()
        for db_name in all_dbs:
            if PSQLInterface._is_unnamed_db(db_name):
                self.drop_db(db_name, conn=conn)
    
################################################################################
### fast operations
################################################################################
def fast_select(query:TUnion[str, Select]=None, qual_table:str=None, 
                index_col:str=None, cols:TList[str]=None, 
                conn:Connection=None) -> pd.DataFrame:
    """
    Some notes:
        - loading an empty table with an index (index_col=something) will not
        display the index name(s), but they are in the (empty) index
    """
    logging.debug('Fastread does not handle dtypes')
    # quote table name 
    if query is None:
        assert qual_table is not None
        if '.' in qual_table:
            schema, table = qual_table.split('.')
            quoted_table = f'"{schema}"."{table}"'
        else:
            quoted_table = f'"{qual_table}"'
        if cols is not None:
            cols_string = ', '.join([f'"{col}"' for col in cols])
            query = f'SELECT {cols_string} FROM {quoted_table}'
        else:
            query = f'SELECT * FROM {quoted_table}'
    head = 'HEADER'
    if isinstance(query, (Select, CompoundSelect)):
        #! the query object must be converted to a pure postgresql-compatible
        #! string for this to work, and in particular to render bound parameters
        # in-line using the literal_binds kwarg and the particular dialect
        query_string = query.compile(bind=conn.engine,
                                     compile_kwargs={'literal_binds': True}, 
                                     dialect=postgresql.dialect())
    elif isinstance(query, str):
        query_string = query
    else:
        raise NotImplementedError()
    copy_sql = f"""COPY ({query_string}) TO STDOUT WITH CSV {head}"""
    buffer = io.StringIO()
    # Note that we need to use a *raw* connection in this method, which can be
    # accessed as conn.connection
    with conn.connection.cursor() as curs:
        curs.copy_expert(copy_sql, buffer)
        buffer.seek(0)
        df:pd.DataFrame = pd.read_csv(buffer)
        if index_col is not None:
            df = df.set_index(index_col)
    return df

def fast_insert(df:pd.DataFrame, qual_table:str, conn:Connection=None,
              columns:TList[str]=None, include_index:bool=True):
    """
    In psycopg 2.9, they changed the .copy_from() method, so that table names
    are now quoted. This means that it won't work with a schema-qualified name.
    This method fixes this by using copy_expert(), as directed by the psycopg2
    docs. 
    """
    if columns is None:
        columns = df.columns
    if '.' in qual_table:
        schema, table = qual_table.split('.')
        quoted_table = f'"{schema}"."{table}"'
    else:
        quoted_table = f'"{qual_table}"'
    start_time = time.time()
    # save dataframe to an in-memory buffer
    buffer = io.StringIO()
    if include_index:
        df = df.reset_index()
    df.to_csv(buffer, header=False, index=False, columns=columns, na_rep='')
    buffer.seek(0)
    columns_string = ', '.join('"{}"'.format(k) for k in columns)
    query = f"""COPY {quoted_table}({columns_string}) FROM STDIN WITH CSV"""
    # Note that we need to use a *raw* connection in this method, which can be
    # accessed as conn.connection
    with conn.connection.cursor() as curs:
        curs.copy_expert(sql=query, file=buffer)
    end_time = time.time()
    nrows = df.shape[0]
    total_time = end_time - start_time
    logging.debug(f'Inserted {nrows} rows, {nrows/total_time} rows/second') 
    
def fast_upsert(df:pd.DataFrame, qual_table:str,
                index_cols:TList[str], columns:TList[str]=None, 
                include_index:bool=True, conn:Connection=None):
    """
    code based on 
    https://stackoverflow.com/questions/46934351/python-postgresql-copy-command-used-to-insert-or-update-not-just-insert
    """
    if include_index:
        df = df.reset_index()
    #! importantly, columns are set after potentially resetting the index
    if columns is None:
        columns = list(df.columns)
    if '.' in qual_table:
        schema, table = qual_table.split('.')
        quoted_table = f'"{schema}"."{table}"'
    else:
        schema = ''
        table = qual_table
        quoted_table = f'"{qual_table}"'

    # create a temporary table with same columns as target table
    # temp_qual_table = f'{schema}.{table}__copy'
    temp_uid = get_uid()[:16]
    temp_qual_table = f'{schema}_{table}__copy_{temp_uid}'
    temp_index_name = f'{schema}_{table}__temp_index_{temp_uid}'
    create_temp_table_query = f"""
    create temporary table {temp_qual_table} as (select * from {quoted_table} limit
    0);
    """
    conn.execute(create_temp_table_query)

    # if provided, create indices on the table
    if index_cols is not None:
        create_temp_index_query = f"""
        CREATE INDEX {temp_index_name} ON {temp_qual_table}({','.join(index_cols)});
        """
        conn.execute(create_temp_index_query)

    # copy data into this table
    fast_insert(df=df, qual_table=temp_qual_table, conn=conn, columns=columns, 
                include_index=include_index)
    
    # comma-separated lists of various things
    target_cols_string = f"{', '.join(columns)}"
    source_cols_string = f"{', '.join([f'{temp_qual_table}.{col}' for col in columns])}"
    index_cols_string = f"{', '.join([f'{col}' for col in index_cols])}"
    # update existing records
    index_conditions = ' AND '.join([f'{qual_table}.{col} = {temp_qual_table}.{col}' for col in index_cols])
    update_query = f"""
    UPDATE {quoted_table}
    SET 
        ({target_cols_string}) = ({source_cols_string})
    FROM
        {temp_qual_table}
    WHERE
        {index_conditions}
    """
    conn.execute(update_query)

    # insert new records
    insert_query = f"""
    INSERT INTO {quoted_table}({target_cols_string})
        ( 
        SELECT {source_cols_string}
        FROM 
        {temp_qual_table} LEFT JOIN {quoted_table} USING({index_cols_string})
        WHERE {table} IS NULL);
    """
    conn.execute(insert_query)