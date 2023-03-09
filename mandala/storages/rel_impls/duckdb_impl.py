import duckdb
import pyarrow as pa
from duckdb import DuckDBPyConnection as Connection
from pypika import Query, Column

from .bases import RelStorage
from .utils import Transactable, transaction
from ...core.utils import get_uid
from ...common_imports import *


class DuckDBRelStorage(RelStorage, Transactable):
    UID_DTYPE = "VARCHAR"  # TODO - change this
    TEMP_ARROW_TABLE = "__arrow__"

    def __init__(self, address: Optional[str] = None, _read_only: bool = False):
        self.in_memory = address is None
        self._read_only = _read_only
        if self.in_memory:
            self._conn = duckdb.connect(database=":memory:", read_only=self._read_only)
        else:
            self.path = address

    ############################################################################
    ### transaction interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return (
            self._conn
            if self.in_memory
            else duckdb.connect(database=self.path, read_only=self._read_only)
        )

    def _end_transaction(self, conn: Connection):
        if not self.in_memory:
            conn.close()

    ############################################################################
    ###
    ############################################################################
    @transaction()
    def get_tables(self, conn: Optional[Connection] = None) -> List[str]:
        return self.execute_df(query="SHOW TABLES;", conn=conn)["name"].tolist()

    @transaction()
    def table_exists(self, relation: str, conn: Optional[Connection] = None) -> bool:
        return relation in self.get_tables(conn=conn)

    @transaction()
    def get_data(self, table: str, conn: Optional[Connection] = None) -> pd.DataFrame:
        return self.execute_df(query=f"SELECT * FROM {table};", conn=conn)

    @transaction()
    def get_count(self, table: str, conn: Optional[Connection] = None) -> int:
        df = self.execute_df(query=f"SELECT COUNT(*) FROM {table};", conn=conn)
        return df["count_star()"].item()

    @transaction()
    def get_all_data(
        self, conn: Optional[Connection] = None
    ) -> Dict[str, pd.DataFrame]:
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
        columns: List[Tuple[str, Optional[str]]],  # [(col name, type), ...]
        defaults: Dict[str, Any],  # {col name: default value, ...}
        primary_key: Optional[str] = None,
        if_not_exists: bool = True,
        conn: Optional[Connection] = None,
    ):
        """
        Create a table with given columns, with an optional primary key.
        Columns are given as tuples of (name, type).
        Columns without a dtype are assumed to be of type `self.UID_DTYPE`.
        """
        query = Query.create_table(table=name).columns(
            *[
                Column(
                    column_name=column_name,
                    column_type=column_type
                    if column_type is not None
                    else self.UID_DTYPE,
                    default=defaults.get(column_name, None),
                    # nullable=False,
                )
                for column_name, column_type in columns
            ],
        )
        if if_not_exists:
            query = query.if_not_exists()
        if primary_key is not None:
            query = query.primary_key(primary_key)
        conn.execute(str(query))
        logger.debug(
            f'Created table "{name}" with columns {[elt[0] for elt in columns]}'
        )

    @transaction()
    def create_column(
        self,
        relation: str,
        name: str,
        default_value: str,
        conn: Optional[Connection] = None,
    ):
        """
        Add a new column to a table.
        """
        query = f"ALTER TABLE {relation} ADD COLUMN {name} {self.UID_DTYPE} DEFAULT '{default_value}'"
        conn.execute(query=query)
        logger.debug(f'Added column "{name}" to table "{relation}"')

    @transaction()
    def drop_column(self, relation: str, name: str, conn: Optional[Connection] = None):
        """
        Drop a column from a table.
        """
        query = f"ALTER TABLE {relation} DROP COLUMN {name}"
        conn.execute(query=query)
        logger.debug(f'Dropped column "{name}" from table "{relation}"')

    @transaction()
    def rename_relation(
        self, name: str, new_name: str, conn: Optional[Connection] = None
    ):
        """
        Rename a table
        """
        query = f"ALTER TABLE {name} RENAME TO {new_name};"
        conn.execute(query)
        logger.debug(f'Renamed table "{name}" to "{new_name}"')

    @transaction()
    def rename_column(
        self, relation: str, name: str, new_name: str, conn: Optional[Connection] = None
    ):
        """
        Rename a column
        """
        query = f'ALTER TABLE {relation} RENAME "{name}" TO "{new_name}";'
        conn.execute(query)
        logger.debug(f'Renamed column "{name}" of table "{relation}" to "{new_name}"')

    @transaction()
    def rename_columns(
        self, relation: str, mapping: Dict[str, str], conn: Optional[Connection] = None
    ):
        # factorize the renaming into two maps that can be applied atomically
        part_1 = {k: get_uid() for k in mapping.keys()}
        part_2 = {part_1[k]: v for k, v in mapping.items()}
        for k, v in part_1.items():
            self.rename_column(relation=relation, name=k, new_name=v, conn=conn)
        for k, v in part_2.items():
            self.rename_column(relation=relation, name=k, new_name=v, conn=conn)
        if len(mapping) > 0:
            logger.debug(f'Renamed columns of table "{relation}" via mapping {mapping}')

    ############################################################################
    ### instance management
    ############################################################################
    @transaction()
    def _get_cols(self, relation: str, conn: Optional[Connection] = None) -> List[str]:
        """
        Duckdb-specific method to get the *ordered* columns of a table.
        """
        return (
            self.execute_arrow(query=f'DESCRIBE "{relation}";', conn=conn)
            .column("column_name")
            .to_pylist()
        )

    @transaction()
    def _get_primary_keys(
        self, relation: str, conn: Optional[Connection] = None
    ) -> List[str]:
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
            raise NotImplementedError()
        elif len(df) == 1:
            return df["constraint_column_names"].item()
        else:
            raise NotImplementedError(f"Multiple primary keys for {relation}")

    @transaction()
    def insert(self, relation: str, ta: pa.Table, conn: Optional[Connection] = None):
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
    def upsert(
        self,
        relation: str,
        ta: pa.Table,
        key_cols: Optional[List[str]] = None,
        conn: Optional[Connection] = None,
    ):
        """
        Upsert rows in a table based on primary key.

        TODO: currently does NOT update matching rows
        """
        if len(ta) == 0:
            return
        if not self.table_exists(relation, conn=conn):
            raise RuntimeError()
        col_names = ta.column_names if isinstance(ta, pa.Table) else ta.columns
        cols_string = ", ".join([f'"{column_name}"' for column_name in col_names])
        conn.register(view_name=self.TEMP_ARROW_TABLE, python_object=ta)
        if key_cols is None:
            primary_keys = self._get_primary_keys(relation=relation, conn=conn)
            if len(primary_keys) != 1:
                raise NotImplementedError()
            primary_key = primary_keys[0]
            query = f'INSERT INTO "{relation}" ({cols_string}) SELECT * FROM {self.TEMP_ARROW_TABLE} WHERE "{primary_key}" NOT IN (SELECT "{primary_key}" FROM "{relation}")'
        else:
            key_cols_str = ", ".join([f'"{column_name}"' for column_name in key_cols])
            key_cols_str = f"({key_cols_str})"
            query = f'INSERT INTO "{relation}" ({cols_string}) SELECT * FROM {self.TEMP_ARROW_TABLE} WHERE {key_cols_str} NOT IN (SELECT {key_cols_str} FROM "{relation}")'
        conn.execute(query)
        conn.unregister(view_name=self.TEMP_ARROW_TABLE)

    @transaction()
    def delete(
        self, relation: str, index: List[str], conn: Optional[Connection] = None
    ):
        """
        Delete rows from a table based on index
        """
        primary_keys = self._get_primary_keys(relation=relation, conn=conn)
        if len(primary_keys) != 1:
            raise NotImplementedError()
        primary_key = primary_keys[0]
        in_str = ", ".join([f"'{i}'" for i in index])
        query = f'DELETE FROM "{relation}" WHERE {primary_key} IN ({in_str})'
        conn.execute(query)

    ############################################################################
    ### queries
    ############################################################################
    @transaction()
    def execute_arrow(
        self,
        query: Union[str, Query],
        parameters: List[Any] = None,
        conn: Optional[Connection] = None,
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
        parameters: List[Any] = None,
        conn: Optional[Connection] = None,
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
        parameters: List[Any] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        if parameters is None:
            parameters = []
        if not isinstance(query, str):
            query = str(query)
        return conn.execute(query, parameters=parameters).fetchdf()
