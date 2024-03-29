from abc import ABC, abstractmethod
import sqlite3
from pypika import Query, Column
from ...common_imports import *
from .utils import Connection
import pyarrow as pa


class RelStorage(ABC):
    """
    Responsible for the low-level (i.e., unaware of mandala-specific concepts)
    interactions with the relational part of the storage, such as creating and
    extending tables, running queries, etc. This is intended to be a pretty
    generic, minimal database interface, supporting just the things we need.

    It's deliberately referred to as "relational storage" as opposed to a
    "relational database" because simpler implementations exist.
    """

    @abstractmethod
    def create_relation(
        self,
        name: str,
        columns: List[Tuple[str, Optional[str]]],  # [(col name, type), ...]
        defaults: Dict[str, Any],  # {col name: default value, ...}
        primary_key: Optional[Union[str, List[str]]] = None,
        if_not_exists: bool = True,
        conn: Optional[Any] = None,
    ):
        """
        Create a relation with the given name and columns.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_column(self, relation: str, name: str, default_value: str):
        raise NotImplementedError()

    @abstractmethod
    def insert(self, name: str, df: pd.DataFrame):
        """
        Append rows to a table
        """
        raise NotImplementedError()

    @abstractmethod
    def upsert(self, relation: str, ta: pa.Table, conn: Optional[Connection] = None):
        """
        Upsert rows in a table based on index
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(
        self,
        relation: str,
        where_col: str,
        where_values: List[str],
        conn: Optional[Connection] = None,
    ):
        """
        Delete rows from a table where `where_col` is in `where_values`
        """
        raise NotImplementedError()

    @abstractmethod
    def get_data(
        self, table: str, conn: Optional[sqlite3.Connection] = None
    ) -> pd.DataFrame:
        """
        Fetch data from a table.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_df(
        self,
        query: Union[str, Query],
        parameters: List[Any] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> pd.DataFrame:
        """
        Execute a query and return the result as a DataFrame.
        """
        raise NotImplementedError()
