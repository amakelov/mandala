from ...common_imports import *
from ...core.config import Config
import duckdb
import pypika
from pypika import Query, Table, Field, Column
import functools
from duckdb import DuckDBPyConnection as Connection


class Transaction:
    def __init__(self):
        pass

    def __call__(self, method) -> "method":
        @functools.wraps(method)
        def inner(instance: "DuckDBStorage", *args, conn: Connection = None, **kwargs):
            if conn is None:
                # new transaction
                if instance.in_memory:
                    conn = instance.conn
                else:
                    conn = instance.get_connection()
                result = method(instance, *args, conn=conn, **kwargs)
                if not instance.in_memory:
                    conn.close()
                return result
            else:
                # nest in existing transaction
                result = method(instance, *args, conn=conn, **kwargs)
                return result

        return inner


transaction = Transaction


class DuckDBStorage:
    def __init__(self, address: str = ":memory:"):
        self.address = address
        self.in_memory = address == ":memory:"
        if self.in_memory:
            self.conn = duckdb.connect(self.address)

    def get_connection(self) -> Connection:
        return duckdb.connect(database=self.address)

    ############################################################################
    ### schema management
    ############################################################################
    @transaction()
    def create_relation(self, name: str, columns: List[str], conn: Connection = None):
        """
        Create a (memoization) table with given columns
        """
        query = (
            Query.create_table(table=name)
            .columns(
                Column(
                    column_name=Config.uid_col,
                    column_type="VARCHAR",
                ),
                *[
                    Column(
                        column_name=c,
                        column_type="VARCHAR",
                    )
                    for c in columns
                ]
            )
            .primary_key(Config.uid_col)
        )
        conn.execute(str(query))
