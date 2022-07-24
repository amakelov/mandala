from typing import Generic

from pypika import Table, Query, Parameter

from .rels import RelStorage, RelAdapter
from ..common_imports import *
from ..core.config import Config


class KVStore:
    """
    Interface for key-value stores for Python objects (keyed by strings).
    """

    def exists(self, k: str) -> bool:
        raise NotImplementedError()

    def set(self, k: str, v: Any) -> None:
        raise NotImplementedError()

    def get(self, k: str) -> Any:
        raise NotImplementedError()

    def delete(self, k: str) -> None:
        raise NotImplementedError()

    def keys(self) -> List[str]:
        raise NotImplementedError()

    ############################################################################
    ### mmethods are useful syntactic sugar
    ############################################################################
    def mget(self, ks: List[str]) -> List[Any]:
        return [self.get(k=k) for k in ks]

    def mset(self, kvs: Dict[str, Any]):
        for k, v in kvs.items():
            self.set(k=k, v=v)

    def mexists(self, ks: List[str]) -> List[bool]:
        return [self.exists(k=k) for k in ks]

    def mdelete(self, ks: List[str]):
        for k in ks:
            self.delete(k=k)


class InMemoryStorage(KVStore):
    """
    Simple in-memory implementation for local testing and/or buffering

    NOTE: delegates error handling
    """

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.dirty_entries: set[str] = set()

    def exists(self, k: str) -> bool:
        return k in self.data

    def set(self, k: str, v: Any):
        self.data[k] = v
        self.dirty_entries.add(k)

    def get(self, k: str) -> Any:
        return self.data[k]

    def delete(self, k: str):
        del self.data[k]

    def keys(self) -> List[str]:
        return list(self.data.keys())
