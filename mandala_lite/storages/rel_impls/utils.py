from ...common_imports import *
from abc import ABC, abstractmethod
import functools
from duckdb import DuckDBPyConnection as Connection


class Transactable(ABC):
    @abstractmethod
    def _get_connection(self) -> Connection:
        raise NotImplementedError()

    @abstractmethod
    def _end_transaction(self, conn: Connection):
        raise NotImplementedError()


class DuckDBTransaction:
    def __init__(self):
        pass

    def __call__(self, method) -> "method":
        @functools.wraps(method)
        def inner(instance: Transactable, *args, conn: Connection = None, **kwargs):
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


transaction = DuckDBTransaction
