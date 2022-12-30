from ..common_imports import *
from multiprocessing import Manager
from multiprocessing.managers import DictProxy


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


class InMemoryStorage(KVStore):
    """
    Simple in-memory implementation for local testing and/or buffering

    NOTE: delegates error handling
    """

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.dirty_entries: set[str] = set()

    def __repr__(self):
        return f"InMemoryStorage(data={self.data})"

    def exists(self, k: str) -> bool:
        return k in self.data

    def set(self, k: str, v: Any):
        self.data[k] = v
        self.dirty_entries.add(k)

    def get(self, k: str) -> Any:
        return self.data[k]

    def delete(self, k: str):
        del self.data[k]
        self.dirty_entries.remove(k)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    @property
    def is_clean(self) -> bool:
        return len(self.dirty_entries) == 0


class MultiProcInMemoryStorage(KVStore):
    def __init__(self):
        manager = Manager()
        self.data: DictProxy[str, Any] = manager.dict()
        self.dirty_entries: DictProxy[str, None] = manager.dict()

    def __repr__(self):
        return f"MultiProcInMemoryStorage(data={self.data})"

    def exists(self, k: str) -> bool:
        return k in self.data.keys()

    def set(self, k: str, v: Any):
        self.data[k] = v
        self.dirty_entries[k] = None

    def get(self, k: str) -> Any:
        return self.data[k]

    def delete(self, k: str):
        del self.data[k]
        self.dirty_entries.pop(k)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    @property
    def is_clean(self) -> bool:
        return len(self.dirty_entries) == 0
