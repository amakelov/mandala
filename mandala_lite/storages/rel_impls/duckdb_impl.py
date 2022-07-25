import duckdb
import pyarrow as pa
from duckdb import DuckDBPyConnection as Connection
from pypika import Query, Column

from .bases import RelStorage
from .utils import Transactable, transaction
from ...common_imports import *
from ...core.config import Config


class DuckDBRelStorage(RelStorage, Transactable):
    UID_DTYPE = "VARCHAR"  # TODO - change this
    VREF_TABLE = Config.vref_table
    EVENT_LOG_TABLE = Config.event_log_table
    TEMP_ARROW_TABLE = "__arrow__"
    SPECIAL_TABLES = (EVENT_LOG_TABLE, TEMP_ARROW_TABLE)

    def __init__(self, address: str = ":memory:"):
        self.address = address
        self.in_memory = address == ":memory:"
        if self.in_memory:
            self._conn = duckdb.connect(self.address)
        self.init()

    def _get_connection(self) -> Connection:
        return self._conn if self.in_memory else duckdb.connect(database=self.address)

    def _end_transaction(self, conn: Connection):
        if not self.in_memory:
            conn.close()

    @transaction()
    def init(self, conn: Connection = None):
        self.create_relation(
            name=self.VREF_TABLE,
            columns=[("value", "blob")],
            conn=conn,
        )
        # Initialize the event log.
        # The event log is just a list of UIDs that changed, for now.
        self.create_relation(self.EVENT_LOG_TABLE, [("table", "varchar")])

    @transaction()
    def get_call_tables(self, conn: Connection = None) -> List[str]:
        tables = self.get_tables(conn=conn)
        return [
            t for t in tables if t not in self.SPECIAL_TABLES and t != self.VREF_TABLE
        ]

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
            data[table] = self.get_data(table=table, conn=conn)
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
        Create a table with given columns, with a primary key named `Config.uid_col`.
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
    def _get_cols(self, relation: str, conn: Connection = None) -> List[str]:
        """
        Duckdb-specific method to get the *ordered* columns of a table.
        """
        return (
            self.execute_arrow(query=f'DESCRIBE "{relation}";', conn=conn)
            .column("column_name")
            .to_pylist()
        )

    @transaction()
    def insert(self, relation: str, pt: pa.Table, conn: Connection = None):
        """
        Append rows to a table
        """
        if pt.empty:
            return
        table_cols = self._get_cols(relation=relation, conn=conn)
        assert set(pt.column_names) == set(table_cols)
        cols_string = ", ".join([f'"{column_name}"' for column_name in pt.column_names])
        conn.register(view_name=self.TEMP_ARROW_TABLE, python_object=pt)
        conn.execute(
            f'INSERT INTO "{relation}" ({cols_string}) SELECT * FROM {self.TEMP_ARROW_TABLE}'
        )
        conn.unregister(view_name=self.TEMP_ARROW_TABLE)

    @transaction()
    def upsert(self, relation: str, ta: pa.Table, conn: Connection = None):
        """
        Upsert rows in a table based on index
        """
        if len(ta) == 0:
            return
        table_cols = self._get_cols(relation=relation, conn=conn)
        assert set(ta.column_names) == set(table_cols)
        cols_string = ", ".join([f'"{column_name}"' for column_name in ta.column_names])
        conn.register(view_name=self.TEMP_ARROW_TABLE, python_object=ta)
        query = f'INSERT INTO "{relation}" ({cols_string}) SELECT * FROM {self.TEMP_ARROW_TABLE} WHERE "{Config.uid_col}" NOT IN (SELECT "{Config.uid_col}" FROM "{relation}")'
        conn.execute(query)
        conn.unregister(view_name=self.TEMP_ARROW_TABLE)

    @transaction()
    def delete(self, relation: str, index: List[str], conn: Connection = None):
        """
        Delete rows from a table based on index
        """
        conn.execute(f'DELETE FROM "{relation}" WHERE {Config.uid_col} IN ({index})')

    ############################################################################
    ### queries
    ############################################################################
    @transaction()
    def execute_arrow(
        self,
        query: Union[str, Query],
        parameters: list[Any] = None,
        conn: Connection = None,
    ) -> pa.Table:
        if parameters is None:
            parameters = []
        if not isinstance(query, str):
            query = str(query)
        return conn.execute(query, parameters=parameters).fetch_arrow_table()

    @transaction()
    def execute_df(
        self,
        query: Union[str, Query],
        parameters: list[Any] = None,
        conn: Connection = None,
    ) -> pd.DataFrame:
        if parameters is None:
            parameters = []
        if not isinstance(query, str):
            query = str(query)
        return conn.execute(query, parameters=parameters).fetchdf()
