from ..common_imports import *
from typing import Generic, TypeVar
from multiprocessing import Manager
from multiprocessing.managers import DictProxy


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class KVCache(Generic[_KT, _VT]):
    """
    Interface for key-value stores for Python objects (keyed by strings).
    """

    def __init__(self):
        self.dirty_entries = set()

    def exists(self, k: _KT) -> bool:
        raise NotImplementedError

    def set(self, k: _KT, v: _VT) -> None:
        raise NotImplementedError

    def get(self, k: _KT) -> Any:
        raise NotImplementedError

    def __getitem__(self, k: _KT) -> _VT:
        raise NotImplementedError

    def __setitem__(self, k: _KT, v: _VT) -> None:
        raise NotImplementedError

    def delete(self, k: _KT) -> None:
        raise NotImplementedError

    def keys(self) -> List[_KT]:
        raise NotImplementedError

    def items(self) -> Iterable[Tuple[_KT, _VT]]:
        raise NotImplementedError

    def evict_all(self) -> None:
        """
        Remove all entries from the cache.
        """
        raise NotImplementedError

    def clear_all(self) -> None:
        """
        Mark all entries as clean.
        """
        raise NotImplementedError


class InMemoryStorage(KVCache):
    """
    Simple in-memory implementation for local testing and/or buffering
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

    def __getitem__(self, k: str) -> Any:
        return self.data[k]

    def __setitem__(self, k: str, v: Any) -> None:
        self.data[k] = v
        self.dirty_entries.add(k)

    def delete(self, k: str):
        del self.data[k]
        self.dirty_entries.remove(k)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self.data.items()

    def evict_all(self) -> None:
        self.data = {}
        self.dirty_entries = set()

    def clear_all(self) -> None:
        self.dirty_entries = set()

    @property
    def is_clean(self) -> bool:
        return len(self.dirty_entries) == 0


class MultiProcInMemoryStorage(KVCache):
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
