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
    TEMP_ARROW_TABLE = "__arrow__"

    def __init__(self, address: str = ":memory:"):
        self.address = address
        self.in_memory = address == ":memory:"
        if self.in_memory:
            self._conn = duckdb.connect(self.address)

    def _get_connection(self) -> Connection:
        return self._conn if self.in_memory else duckdb.connect(database=self.address)

    def _end_transaction(self, conn: Connection):
        if not self.in_memory:
            conn.close()

    @transaction()
    def get_tables(self, conn: Connection = None) -> List[str]:
        return conn.execute("SHOW TABLES;").fetchdf()["name"].values.tolist()

    @transaction ()
    def table_exists(self, relation: str, conn: Connection = None) -> bool:
        return relation in self.get_tables(conn=conn)

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
        columns: List[tuple[str, Optional[str]]],
        primary_key: Optional[str] = None,
        conn: Connection = None,
    ):
        """
        Create a table with given columns, with an optional primary key
        """
        query = Query.create_table(table=name).if_not_exists().columns(
            *[
                Column(
                    column_name=c,
                    column_type=dtype if dtype is not None else self.UID_DTYPE,
                )
                for c, dtype in columns
            ],
        )
        if primary_key is not None:
            query = query.primary_key(primary_key)
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
        query = f"ALTER TABLE {relation} ADD COLUMN {name} {self.UID_DTYPE} DEFAULT '{default_value}'"
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
    def _get_primary_keys(self, relation: str, conn: Connection = None) -> List[str]:
        """
        Duckdb-specific method to get the primary key of a table.
        """
        constraint_type = "PRIMARY KEY"
        df = self.execute_df(query=f"SELECT * FROM duckdb_constraints();", conn=conn)
        df = df[["table_name", "constraint_type", "constraint_column_names"]]
        df = df[
            (df["table_name"] == relation) & (df["constraint_type"] == constraint_type)
        ]
        if len(df) == 0:
            return []
        elif len(df) == 1:
            return df["constraint_column_names"].item()
        else:
            raise NotImplementedError(f"Multiple primary keys for {relation}")

    @transaction()
    def insert(self, relation: str, ta: pa.Table, conn: Connection = None):
        """
        Append rows to a table
        """
        if len(ta) == 0:
            return
        table_cols = self._get_cols(relation=relation, conn=conn)
        assert set(ta.column_names) == set(table_cols)
        cols_string = ", ".join([f'"{column_name}"' for column_name in ta.column_names])
        conn.register(view_name=self.TEMP_ARROW_TABLE, python_object=ta)
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
        # TODO this a temporary hack until we get function signature sync working!
        if not self.table_exists(relation, conn=conn):
            self.create_relation(relation, [(col, None) for col in ta.column_names], primary_key=Config.uid_col, conn=conn)
        table_cols = self._get_cols(relation=relation, conn=conn)
        assert set(ta.column_names) == set(table_cols)
        cols_string = ", ".join([f'"{column_name}"' for column_name in ta.column_names])
        primary_keys = self._get_primary_keys(relation=relation, conn=conn)
        if len(primary_keys) != 1:
            raise NotImplementedError()
        primary_key = primary_keys[0]
        conn.register(view_name=self.TEMP_ARROW_TABLE, python_object=ta)
        query = f'INSERT INTO "{relation}" ({cols_string}) SELECT * FROM {self.TEMP_ARROW_TABLE} WHERE "{primary_key}" NOT IN (SELECT "{primary_key}" FROM "{relation}")'
        conn.execute(query)
        conn.unregister(view_name=self.TEMP_ARROW_TABLE)

    @transaction()
    def delete(self, relation: str, index: List[str], conn: Connection = None):
        """
        Delete rows from a table based on index
        """
        primary_keys = self._get_primary_keys(relation=relation, conn=conn)
        if len(primary_keys) != 1:
            raise NotImplementedError()
        primary_key = primary_keys[0]
        in_str = ", ".join([f"'{i}'" for i in index])
        conn.execute(f'DELETE FROM "{relation}" WHERE {primary_key} IN ({in_str})')

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
    def execute_no_results(
        self,
        query: Union[str, Query],
        parameters: list[Any] = None,
        conn: Connection = None,
    ) -> None:
        if parameters is None:
            parameters = []
        if not isinstance(query, str):
            query = str(query)
        return conn.execute(query, parameters=parameters)

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
