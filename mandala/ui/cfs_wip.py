import textwrap
from ..common_imports import *
from ..core.config import *
from ..core.tps import Type
from ..core.model import Ref, FuncOp, Call
from ..core.builtins_ import StructOrientations, Builtins
from ..storages.rel_impls.utils import Transactable, transaction, Connection
from ..queries.graphs import copy_subgraph
from ..core.prov import propagate_struct_provenance
from ..queries.weaver import CallNode, ValNode, PaddedList
from ..queries.viz import GraphPrinter, visualize_graph
from .funcs import FuncInterface
from .storage import Storage, ValueLoader
from .cfs_utils import estimate_uid_storage, convert_bytes_to



class CanonicalCF:
    """
    Some experiments with more general cfs
    """
    pass



class PaddedList(Sequence[Optional[T]]):
    """
    A list-like object that is backed by a list of values and a list of indices,
    and has length `length`. When indexed, it returns the value from the list at
    the corresponding index, or None if the index is not in the list of indices.
    """

    def __init__(self, support: Dict[int, T], length: int):
        self.support = support
        self.length = length

    def __repr__(self) -> str:
        return f"PaddedList({self.tolist()})"

    def tolist(self) -> List[Optional[T]]:
        return [self.support.get(i, None) for i in range(self.length)]

    def copy(self) -> "PaddedList":
        return PaddedList(support=self.support.copy(), length=self.length)

    def copy_item(self, i: int, times: int, inplace: bool = False) -> "PaddedList":
        res = self if inplace else self.copy()
        if i not in res.support:
            res.length = res.length + times
        else:
            for j in range(res.length, res.length + times):
                res.support[j] = res.support[i]
            res.length += times
        return res
    
    def change_length(self, length: int, inplace: bool = False) -> "PaddedList":
        res = self if inplace else self.copy()
        res.length = length
        return res

    def append_items(self, items: List[T], inplace: bool = False) -> "PaddedList":
        res = self if inplace else self.copy()
        for i, item in enumerate(items):
            res.support[res.length + i] = item
        res.length += len(items)
        return res

    @staticmethod
    def from_list(lst: List[Optional[T]], length: Optional[int] = None) -> "PaddedList":
        if length is None:
            length = len(lst)
        items = {i: v for i, v in enumerate(lst) if v is not None}
        return PaddedList(support=items, length=length)

    def dropna(self) -> List[T]:
        return [self.support[k] for k in sorted(self.support.keys())]

    @staticmethod
    def padded_like(plist: "PaddedList[T1]", values: List[T]) -> "PaddedList[T]":
        support = {i: values[j] for j, i in enumerate(sorted(plist.support.keys()))}
        return PaddedList(support=support, length=len(plist))

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Optional[T]]:
        return (self.support.get(i, None) for i in range(self.length))

    def __getitem__(
        self, idx: Union[int, slice, List[bool], np.ndarray]
    ) -> Union[T, "PaddedList"]:
        if isinstance(idx, int):
            return self.support.get(idx, None)
        elif isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, (list, np.ndarray)):
            return self.masked(idx)
        else:
            raise NotImplementedError(
                "Indexing only supported for integers, slices, and boolean arrays"
            )

    def masked(self, mask: Union[List[bool], np.ndarray]) -> "PaddedList":
        """
        Return a new `PaddedList` object with the values masked by the given
        boolean array, and the indices updated accordingly.
        """
        if len(mask) != self.length:
            raise ValueError("Boolean mask must have the same length as the list")
        result_items = {}
        cur_masked_idx = 0
        for mask_idx, m in enumerate(mask):
            if m:
                if mask_idx in self.support:
                    result_items[cur_masked_idx] = self.support[mask_idx]
                cur_masked_idx += 1
        return PaddedList(support=result_items, length=cur_masked_idx)

    def keep_only(self, indices: Set[int]) -> "PaddedList":
        """
        Return a new `PaddedList` object with the values masked by the given
        list of indices.
        """
        items = {i: v for i, v in self.support.items() if i in indices}
        return PaddedList(support=items, length=self.length)

