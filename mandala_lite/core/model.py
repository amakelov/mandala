import pandas as pd

from .config import Config
from ..common_imports import *
from .utils import Hashing
from .sig import Signature


class ValueRef:
    """
    Wraps objects with storage metadata.

    This is the object passed between memoized functions (ops).
    """

    def __init__(self, uid: str, obj: Any, in_memory: bool):
        self.uid = uid
        self.obj = obj
        self.in_memory = in_memory

    def __repr__(self) -> str:
        if self.in_memory:
            return f"ValueRef({self.obj}, uid={self.uid})"
        else:
            return f"ValueRef(in_memory=False, uid={self.uid})"

    def detached(self) -> "ValueRef":
        """
        Return a *copy* of this `ValueRef` without the pointer to the underlying
        object.

        (Correspondingly, this is marked as not `in_memory`.)
        """
        return ValueRef(uid=self.uid, obj=None, in_memory=False)


def wrap(obj: Any, uid: Optional[str] = None) -> ValueRef:
    """
    Wraps a value as a `ValueRef`, if it isn't one already.

    The uid is either explicitly set or a content hash is generated. Note that
    content hashing may take non-trivial time for large objects. When `obj` is
    already a `ValueRef` and `uid` is provided, an error is raised.
    """
    if isinstance(obj, ValueRef):
        if uid is not None:
            # protect against accidental misuse
            raise ValueError(f"Cannot change uid of ValueRef: {obj}")
        return obj
    else:
        uid = Hashing.get_content_hash(obj) if uid is None else uid
        return ValueRef(uid=uid, obj=obj, in_memory=True)


T = TypeVar("T")


def unwrap(obj: Union[T, ValueRef]) -> T:
    """
    If an object is a `ValueRef`, returns the wrapped object; otherwise, return
    the object itself.
    """
    if not isinstance(obj, ValueRef):
        return obj
    else:
        return obj.obj


class Call:
    """
    Represents the inputs, outputs and uid of a call to an operation.

    The inputs to an operation are represented as a dictionary, and the outputs
    are a (possibly empty) list, mirroring how Python functions have named
    inputs but nameless outputs for functions. This convention is followed
    throughout mandala to stick as close as possible to the object being
    modeled.

    The uid is a unique identifier for the call derived from the inputs and the
    identity of the operation.
    """

    def __init__(
        self,
        uid: str,
        inputs: Dict[str, ValueRef],
        outputs: List[ValueRef],
        op: "FuncOp",
    ):
        self.uid = uid
        self.inputs = inputs
        self.outputs = outputs
        self.op = op

    def detached(self) -> "Call":
        """
        Returns a "detached" *copy* of this call, meaning that the inputs and
        outputs are replaced by detached *copies* of the original inputs and
        outputs.

        This is just a simple way to extract the metadata of the call without
        coming up with a separate encoding.
        """
        return Call(
            uid=self.uid,
            inputs={k: v.detached() for k, v in self.inputs.items()},
            outputs=[v.detached() for v in self.outputs],
            op=self.op,
        )

    @staticmethod
    def from_row(row: pd.DataFrame) -> "Call":
        columns = list(row.columns)
        output_columns = [column for column in columns if column.startswith("output")]
        input_columns = [
            column
            for column in columns
            if column not in output_columns and column != Config.uid_col
        ]
        return Call(
            uid=row[Config.uid_col],
            inputs={
                k: ValueRef(row[k].item(), obj=None, in_memory=False)
                for k in input_columns
            },
            outputs=[
                ValueRef(row[k].item(), obj=None, in_memory=False)
                for k in sorted(output_columns, key=lambda x: int(x[7:]))
            ],
            op=None,
        )


class FuncOp:
    """
    Operation that models function execution.

    The `is_synchronized` attribute is responsible for keeping track of whether
    this operation has been connected to the storage.

    The synchronization process is responsible for verifying that the function
    signature last stored is compatible with the current signature, and
    performing the necessary updates to the stored signature.

    See also:
        - `Signature`
    """

    def __init__(self, func: Callable, version: int = 0, is_super: bool = False):
        self.func = func
        self.py_sig = inspect.signature(self.func)
        self.sig = Signature.from_py(
            sig=inspect.signature(func),
            name=func.__name__,
            version=version,
            is_super=is_super,
        )
        # TODO: use this
        self.is_synchronized = False

    def compute(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Computes the function with the given *unwrapped* inputs, named by
        internal input names.
        """
        inv_mapping = {v: k for k, v in self.sig.ext_to_int_input_map.items()}
        inputs = {inv_mapping[k]: v for k, v in inputs.items()}
        result = self.func(**inputs)
        if self.sig.n_outputs == 0:
            assert result is None, "Function returned non-None value"
            return []
        elif self.sig.n_outputs == 1:
            return [result]
        else:
            return list(result)
