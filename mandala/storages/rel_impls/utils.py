from ...common_imports import *
from ...core.config import Config
from abc import ABC, abstractmethod
import functools

if Config.has_duckdb:
    from duckdb import DuckDBPyConnection as Connection
else:

    class Connection:
        pass


class Transactable(ABC):
    @abstractmethod
    def _get_connection(self) -> Connection:
        raise NotImplementedError()

    @abstractmethod
    def _end_transaction(self, conn: Connection):
        raise NotImplementedError()


class Transaction:
    def __init__(self):
        pass

    def __call__(self, method) -> "method":
        @functools.wraps(method)
        def inner(instance: Transactable, *args, conn: Connection = None, **kwargs):
            transaction_started_here = False
            if conn is None:
                # new transaction
                conn = instance._get_connection()
                instance._current_conn = conn
                transaction_started_here = True
                # conn.execute("BEGIN IMMEDIATE")
            try:
                result = method(instance, *args, conn=conn, **kwargs)
                if transaction_started_here:
                    instance._end_transaction(conn=conn)
                return result
            except Exception as e:
                if transaction_started_here:
                    conn.rollback()
                    instance._end_transaction(conn=conn)
                raise e
            # instance._end_transaction(conn=conn)
            # return result
            # else:
            #     # nest in existing transaction
            #     result = method(instance, *args, conn=conn, **kwargs)
            #     return result

        return inner


transaction = Transaction
