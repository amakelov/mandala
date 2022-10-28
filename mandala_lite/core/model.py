import pyarrow as pa

from .config import Config
from ..common_imports import *
from .utils import Hashing
from .sig import Signature


class Delayed:
    pass


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

    @staticmethod
    def is_delayed(vref: "ValueRef") -> bool:
        return isinstance(vref.obj, Delayed)

    @staticmethod
    def make_delayed() -> "ValueRef":
        return ValueRef(uid="", obj=Delayed(), in_memory=False)


def wrap(obj: Any, uid: Optional[str] = None) -> ValueRef:
    """
    Wraps a value as a `ValueRef`, if it isn't one already.

    The uid is either explicitly set, or a content hash is generated. Note that
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
    Represents the data of a call to an operation (inputs, outputs, and call UID).

    The inputs to an operation are represented as a dictionary, and the outputs
    are a (possibly empty) list, mirroring how Python functions have named
    inputs but nameless (but ordered) outputs. This convention is followed
    throughout to stick as close as possible to the object being modeled (a
    Python function).

    The UID is a unique identifier for the call derived *deterministically* from
    the inputs' UIDs and the "identity" of the operation.
    """

    def __init__(
        self,
        uid: str,
        inputs: Dict[str, ValueRef],
        outputs: List[ValueRef],
        func_op: "FuncOp",
    ):
        self.uid = uid
        self.inputs = inputs
        self.outputs = outputs
        self.func_op = func_op

    @staticmethod
    def from_row(row: pa.Table, func_op: "FuncOp") -> "Call":
        """
        Generate a `Call` from a single-row table encoding the UID, input and
        output UIDs.

        NOTE: this does not include the objects for the inputs and outputs to the call!
        """
        columns = row.column_names
        output_columns = [
            column for column in columns if column.startswith(Config.output_name_prefix)
        ]
        input_columns = [
            column
            for column in columns
            if column not in output_columns and column != Config.uid_col
        ]
        return Call(
            uid=row.column(Config.uid_col)[0].as_py(),
            inputs={
                k: ValueRef(row.column(k)[0].as_py(), obj=None, in_memory=False)
                for k in input_columns
            },
            outputs=[
                ValueRef(row.column(k)[0].as_py(), obj=None, in_memory=False)
                for k in sorted(output_columns, key=lambda x: int(x[7:]))
            ],
            func_op=func_op,
        )

    def set_input_values(self, inputs: Dict[str, ValueRef]) -> "Call":
        res = copy.deepcopy(self)
        assert set(inputs.keys()) == set(res.inputs.keys())
        for k, v in inputs.items():
            current = res.inputs[k]
            assert v.in_memory and not current.in_memory
            current.obj = v.obj
            current.in_memory = True
        return res

    def set_output_values(self, outputs: List[ValueRef]) -> "Call":
        res = copy.deepcopy(self)
        assert len(outputs) == len(res.outputs)
        for i, v in enumerate(outputs):
            current = res.outputs[i]
            assert v.in_memory and not current.in_memory
            current.obj = v.obj
            current.in_memory = True
        return res


class FuncOp:
    """
    Operation that models function execution.

    The `is_synchronized` attribute is responsible for keeping track of whether
    this operation has been connected to the storage.

    The synchronization process is responsible for verifying that the function
    signature last stored is compatible with the current signature, and
    performing the necessary updates to the stored signature.

    See also:
        - `mandala_lite.core.sig.Signature`
    """

    def __init__(self, func: Callable, version: int = 0, ui_name: Optional[str] = None):
        # `ui_name` is useful for simulating multi-user scenarios in tests
        self.func = func
        self.py_sig = inspect.signature(self.func)
        ui_name = self.func.__name__ if ui_name is None else ui_name
        self.sig = Signature.from_py(
            sig=inspect.signature(func), name=ui_name, version=version
        )

    def compute(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Computes the function on the given *unwrapped* inputs. Returns a list of
        `self.sig.n_outputs` outputs (after checking they are the number
        expected by the interface).

        This expects the inputs to be named using *internal* input names.
        """
        result = self.func(**inputs)
        if self.sig.n_outputs == 0:
            assert (
                result is None
            ), f"Operation {self} has zero outputs, but its function returned {result}"
            return []
        elif self.sig.n_outputs == 1:
            return [result]
        else:
            assert isinstance(
                result, tuple
            ), f"Operation {self} has multiple outputs, but its function returned a non-tuple: {result}"
            assert (
                len(result) == self.sig.n_outputs
            ), f"Operation {self} has {self.sig.n_outputs} outputs, but its function returned a tuple of length {len(result)}"
            return list(result)

    @staticmethod
    def _from_data(sig: Signature, f: Optional[Callable] = None) -> "FuncOp":
        """
        Create a `FuncOp` object based on a signature and maybe a function. For
        internal use only.
        """
        if f is None:
            f = lambda *args, **kwargs: None
            res = FuncOp(func=f, version=sig.version, ui_name=sig.ui_name)
            res.sig = sig
            res.func = None
        else:
            res = FuncOp(func=f, version=sig.version, ui_name=sig.ui_name)
            res.sig = sig
        return res
