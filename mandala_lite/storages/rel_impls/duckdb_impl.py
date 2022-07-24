import functools

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection as Connection
from pypika import Query, Column
from pypika.queries import QueryBuilder

from ..rels import RelStorage
from ...common_imports import *
from ...core.config import Config


class Transaction:
    def __init__(self):
        pass

    def __call__(self, method) -> "method":
        @functools.wraps(method)
        def inner(instance: "DuckDBStorage", *args, conn: Connection = None, **kwargs):
            if conn is None:
                # new transaction
                conn = instance._get_connection()
                result = method(instance, *args, conn=conn, **kwargs)
                instance._end_transaction(conn=conn)
                return result
            else:
                # nest in existing transaction
                result = method(instance, *args, conn=conn, **kwargs)
                return result

        return inner


transaction = Transaction


class DuckDBRelStorage(RelStorage):
    UID_DTYPE = "VARCHAR"  # TODO - change this
    VREF_TABLE = Config.vref_table
    TEMP_PANDAS_TABLE = "__pandas__"

    def __init__(self, address: str = ":memory:"):
        self.address = address
        self.in_memory = address == ":memory:"
        if self.in_memory:
            self._conn = duckdb.connect(self.address)
        self.init()
        # if not self.exists_db(address=self.address):
        #     self.init()

    # @staticmethod
    # def exists_db(address:str) -> bool:
    #     if address == ":memory:":
    #         return True
    #     else:
    #         return os.path.exists(address)

    def _get_connection(self) -> Connection:
        return self._conn if self.in_memory else duckdb.connect(database=self.address)

    def _end_transaction(self, conn: Connection):
        if not self.in_memory:
            conn.close()

    @transaction()
    def init(self, conn: Connection = None):
        self.create_relation(
            name=self.VREF_TABLE,
            columns=[],
            conn=conn,
        )

    @transaction()
    def get_tables(self, conn: Connection = None) -> List[str]:
        return conn.execute("SHOW TABLES;").fetchdf()["name"].values.tolist()

    @transaction()
    def get_data(self, table: str, conn: Connection = None) -> pd.DataFrame:
        return conn.execute(f"SELECT * FROM {table};").fetchdf()

    @transaction()
    def get_all_data(self, conn: Connection = None) -> Dict[str, pd.DataFrame]:
        tables = self.get_tables(conn=conn)
        data = {}
        for table in tables:
            data[table] = self.get_data(table, conn)
        return data

    ############################################################################
    ### schema management
    ############################################################################
    @transaction()
    def create_relation(
        self,
        name: str,
        columns: List[tuple[str, str]],
        conn: Connection = None,
    ):
        """
        Create a (memoization) table with given columns.

        Importantly, this *always* creates a primary key on the UID column.
        """
        query = (
            Query.create_table(table=name)
            .columns(
                Column(
                    column_name=Config.uid_col,
                    column_type=self.UID_DTYPE,
                ),
                *[
                    Column(
                        column_name=c,
                        column_type=dtype if dtype is not None else self.UID_DTYPE,
                    )
                    for c, dtype in columns
                ],
            )
            .primary_key(Config.uid_col)
        )
        print(query)
        conn.execute(str(query))

    @transaction()
    def delete_relation(self, name: str, conn: Connection = None):
        """
        Delete a (memoization) table
        """
        query = Query.drop_table(table=name)
        conn.execute(str(query))

    @transaction()
    def create_column(
        self, relation: str, name: str, default_value: str, conn: Connection = None
    ):
        """
        Add a new column to a table.
        """
        query = f"ALTER TABLE {relation} ADD COLUMN {name} {self.UID_DTYPE} DEFAULT {default_value}"
        conn.execute(query=query)

    ############################################################################
    ### instance management
    ############################################################################
    @transaction()
    def insert(self, name: str, df: pd.DataFrame, conn: Connection = None):
        """
        Append rows to a table
        """
        conn.register(view_name=self.TEMP_PANDAS_TABLE, python_object=df)
        conn.execute(f"INSERT INTO '{name}' SELECT * FROM {self.TEMP_PANDAS_TABLE}")
        conn.unregister(view_name=self.TEMP_PANDAS_TABLE)

    @transaction()
    def upsert(self, name: str, df: pd.DataFrame, conn: Connection = None):
        """
        Upsert rows in a table based on index
        """
        conn.register(view_name=self.TEMP_PANDAS_TABLE, python_object=df)
        conn.execute(
            f"INSERT INTO '{name}' SELECT * FROM {self.TEMP_PANDAS_TABLE} WHERE {Config.uid_col} NOT IN (SELECT '{Config.uid_col}' FROM '{name}')"
        )
        conn.unregister(view_name=self.TEMP_PANDAS_TABLE)

    @transaction()
    def delete(self, name: str, index: List[str], conn: Connection = None):
        """
        Delete rows from a table based on index
        """
        conn.execute(f"DELETE FROM '{name}' WHERE {Config.uid_col} IN ({index})")

    ############################################################################
    ### queries
    ############################################################################
    @transaction()
    def execute(
        self,
        query: Union[str, Query],
        parameters: list[Any] = [],
        conn: Connection = None,
    ) -> pd.DataFrame:
        if isinstance(query, QueryBuilder):
            query = str(query)
        return conn.execute(query, parameters=parameters).fetchdf()
