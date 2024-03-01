import sqlite3
from pandas.api.types import is_string_dtype
import pyarrow as pa
from pypika import Query, Column

from .bases import RelStorage
from .utils import Transactable, transaction
from ...core.utils import get_uid
from ...common_imports import *


class SQLiteRelStorage(RelStorage, Transactable):
    UID_DTYPE = "VARCHAR"

    def __init__(
        self,
        address: Optional[str] = None,
        _read_only: bool = False,
        autocommit: bool = False,
        journal_mode: str = "WAL",
        page_size: int = 32768,
        mmap_size_MB: int = 256,
        cache_size_pages: int = 1000,
        synchronous: str = "normal",
    ):
        self.journal_mode = journal_mode
        self.page_size = page_size
        self.mmap_size_MB = mmap_size_MB
        self.cache_size_pages = cache_size_pages
        self.synchronous = synchronous
        self.autocommit = autocommit
        self._read_only = _read_only

        self.in_memory = address is None
        if self.in_memory:
            self._id = get_uid()
            self._connection_address = f"file:{self._id}?mode=memory&cache=shared"
            self._conn = sqlite3.connect(
                str(self._connection_address), isolation_level=None, uri=True
            )
            with self._conn:
                self.apply_optimizations(self._conn)
        else:
            self._connection_address = address

    def get_optimizations(self) -> List[str]:
        """
        This needs some explaining:
            - you cannot change `page_size` after setting `journal_mode = WAL`
            - `journal_mode = WAL` is persistent across database connections
            - `cache_size` is in pages when positive, in kB when negative
        """
        if self.mmap_size_MB is None:
            mmap_size = 0
        else:
            mmap_size = self.mmap_size_MB * 1024**2
        pragma_dict = OrderedDict(
            [
                # 'temp_store': 'memory',
                ("synchronous", self.synchronous),
                ("page_size", self.page_size),
                ("cache_size", self.cache_size_pages),
                ("journal_mode", self.journal_mode),
                ("mmap_size", mmap_size),
                ("foreign_keys", "ON"),
            ]
        )
        lines = [f"PRAGMA {k} = {v};" for k, v in pragma_dict.items()]
        return lines

    def apply_optimizations(self, conn: sqlite3.Connection):
        opts = self.get_optimizations()
        for line in opts:
            conn.execute(line)

    def read_cursor(self, c: sqlite3.Cursor) -> pd.DataFrame:
        if c.description is None:
            assert len(c.fetchall()) == 0
            return pd.DataFrame()
        colnames = [col[0] for col in c.description]
        df = pd.DataFrame(c.fetchall(), columns=colnames)
        return self.postprocess_df(df)

    def postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, dtype in df.dtypes.items():
            if not is_string_dtype(dtype):
                df[col] = df[col].astype(str)
        return df

    ############################################################################
    ### transaction interface
    ############################################################################
    def _get_connection(self) -> sqlite3.Connection:
        if self.in_memory:
            return self._conn
        else:
            return sqlite3.connect(
                str(self._connection_address), isolation_level=None  # "IMMEDIATE"
            )

    def _end_transaction(self, conn: sqlite3.Connection):
        conn.commit()
        if not self.in_memory:
            conn.close()

    ############################################################################
    ###
    ############################################################################
    @transaction()
    def get_tables(self, conn: Optional[sqlite3.Connection] = None) -> List[str]:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        cur = conn.cursor()
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]

    @transaction()
    def table_exists(
        self, relation: str, conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        return relation in self.get_tables(conn=conn)

    @transaction()
    def get_data(
        self, table: str, conn: Optional[sqlite3.Connection] = None
    ) -> pd.DataFrame:
        return self.execute_df(query=f"SELECT * FROM {table};", conn=conn)

    @transaction()
    def get_count(self, table: str, conn: Optional[sqlite3.Connection] = None) -> int:
        query = f"SELECT COUNT(*) FROM {table};"
        return int(self.execute_df(query=query, conn=conn).iloc[0, 0])

    @transaction()
    def get_all_data(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, pd.DataFrame]:
        return {
            table: self.get_data(table, conn=conn)
            for table in self.get_tables(conn=conn)
        }

    ############################################################################
    ### schema management
    ############################################################################
    @transaction()
    def create_relation(
        self,
        name: str,
        columns: List[Tuple[str, Optional[str]]],  # [(col name, type), ...]
        defaults: Dict[str, Any],  # {col name: default value, ...}
        primary_key: Optional[Union[str, List[str]]] = None,
        if_not_exists: bool = True,
        conn: Optional[sqlite3.Connection] = None,
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
            if isinstance(primary_key, str):
                query = query.primary_key(primary_key)
            else:
                query = query.primary_key(*primary_key)
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
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Add a new column to a table.
        """
        query = (
            f"ALTER TABLE {relation} ADD COLUMN {name} TEXT DEFAULT '{default_value}'"
        )
        conn.execute(query)
        logger.debug(f'Created column "{name}" in table "{relation}"')

    @transaction()
    def drop_column(
        self, relation: str, name: str, conn: Optional[sqlite3.Connection] = None
    ):
        """
        Drop a column from a table.
        """
        query = f'ALTER TABLE {relation} DROP COLUMN "{name}"'
        conn.execute(query)
        logger.debug(f'Dropped column "{name}" from table "{relation}"')

    @transaction()
    def rename_relation(
        self, name: str, new_name: str, conn: Optional[sqlite3.Connection] = None
    ):
        """
        Rename a table
        """
        query = f"ALTER TABLE {name} RENAME TO {new_name};"
        conn.execute(query)
        logger.debug(f'Renamed table "{name}" to "{new_name}"')

    @transaction()
    def rename_column(
        self,
        relation: str,
        name: str,
        new_name: str,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Rename a column
        """
        query = f'ALTER TABLE {relation} RENAME COLUMN "{name}" TO "{new_name}";'
        conn.execute(query)
        logger.debug(f'Renamed column "{name}" in table "{relation}" to "{new_name}"')

    @transaction()
    def rename_columns(
        self,
        relation: str,
        mapping: Dict[str, str],
        conn: Optional[sqlite3.Connection] = None,
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
    def _get_cols(
        self, relation: str, conn: Optional[sqlite3.Connection] = None
    ) -> List[str]:
        """
        get the *ordered* columns of a table.
        """
        query = f"PRAGMA table_info({relation})"
        df = self.execute_df(query=query, conn=conn)
        return list(df["name"])

    @transaction()
    def _get_primary_keys(
        self, relation: str, conn: Optional[sqlite3.Connection] = None
    ) -> List[str]:
        """
        get the primary key of a table.
        """
        query = f"PRAGMA table_info({relation})"
        df = self.execute_df(query=query, conn=conn)
        return list(df[df["pk"].apply(int) == 1]["name"])

    @transaction()
    def insert(
        self, relation: str, ta: pa.Table, conn: Optional[sqlite3.Connection] = None
    ):
        """
        Append rows to a table
        """
        df = ta.to_pandas()
        if df.empty:
            return
        columns = df.columns.tolist()
        col_str = ", ".join([f'"{col}"' for col in columns])
        placeholder_str = ",".join(["?" for _ in columns])
        query = f"INSERT INTO {relation}({col_str}) VALUES ({placeholder_str})"
        parameters = list(df.itertuples(index=False))
        conn.executemany(query, parameters)

    @transaction()
    def upsert(
        self,
        relation: str,
        ta: pa.Table,
        key_cols: Optional[List[str]] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Upsert rows in a table based on primary key.
        """
        if isinstance(ta, pa.Table):
            df = ta.to_pandas()
        else:
            df = ta
        if df.empty:  # engine complains
            return
        columns = df.columns.tolist()
        col_str = ", ".join([f'"{col}"' for col in columns])
        placeholder_str = ",".join(["?" for _ in columns])
        query = (
            f"INSERT OR REPLACE INTO {relation}({col_str}) VALUES ({placeholder_str})"
        )
        parameters = list(df.itertuples(index=False))
        conn.executemany(query, parameters)

    @transaction()
    def delete(
        self,
        relation: str,
        where_col: str,
        where_values: List[str],
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Delete rows from a table where `where_col` is in `where_values`
        """
        query = f"DELETE FROM {relation} WHERE {where_col} IN ({','.join(['?']*len(where_values))})"
        conn.execute(query, where_values)

    @transaction()
    def vacuum(self, warn: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        ! this needs a lot of free space on disk to work (~2x db size)
        """
        if warn:
            total_db_size = (
                self.execute_df("PRAGMA page_count").astype(float).iloc[0, 0]
                * self.page_size
            )
            question = "Vacuuming a database of size {:.2f} MB, this may take a long time and requires ~2x as much free space on disk, are you sure?".format(
                total_db_size / 1024**2
            )
            # ask the user if they are sure
            user_input = input(question + " (y/n): ")
            if user_input.lower() != "y":
                logging.info("Aborting vacuuming.")
                return
        conn.execute("VACUUM")

    @transaction()
    def execute_df(
        self,
        query: Union[str, Query],
        parameters: List[Any] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> pd.DataFrame:
        if parameters is None:
            parameters = []
        cursor = conn.execute(str(query), parameters)
        return self.postprocess_df(self.read_cursor(cursor))

    @transaction()
    def execute_arrow(
        self,
        query: Union[str, Query],
        parameters: List[Any] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> pa.Table:
        if isinstance(query, Query):
            query = str(query)
        df = self.execute_df(query=query, parameters=parameters, conn=conn)
        return pa.Table.from_pandas(df)

    @transaction()
    def execute_no_results(
        self,
        query: Union[str, Query],
        parameters: List[Any] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        if parameters is None:
            parameters = []
        if not isinstance(query, str):
            query = str(query)
        return conn.execute(query, parameters)
