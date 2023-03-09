from typing import Type
import textwrap
from .config import Config, parse_output_idx
from ..common_imports import *
from .utils import Hashing, get_uid
from .sig import (
    Signature,
    _postprocess_outputs,
    _get_arg_annotations,
    _get_return_annotations,
)
from .tps import Type, AnyType, ListType, DictType, SetType

from ..deps.tracers import TracerABC, DecTracer

if Config.has_torch:
    import torch
    from .integrations import sig_from_jit_script


class Delayed:
    pass


class TransientObj:
    def __init__(self, obj: Any, unhashable: bool = False):
        self.obj = obj
        self.unhashable = unhashable


def get_transient_uid(content_hash: str) -> str:
    return f"__transient__.{content_hash}"


def is_transient_uid(uid: str) -> bool:
    return uid.startswith("__transient__")


################################################################################
### refs
################################################################################
class Ref:
    def __init__(self, uid: str, obj: Any, in_memory: bool):
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory

    @property
    def obj(self) -> Any:
        return self._obj

    @staticmethod
    def from_uid(uid: str) -> "Ref":
        from .builtins_ import Builtins

        if Builtins.is_builtin_uid(uid=uid):
            builtin_id, uid = Builtins.parse_builtin_uid(uid=uid)
            return Builtins.spawn_builtin(builtin_id=builtin_id, uid=uid)
        elif is_transient_uid(uid=uid):
            return ValueRef(uid=uid, obj=None, in_memory=False, transient=True)
        else:
            return ValueRef(uid=uid, obj=None, in_memory=False)

    def is_delayed(self) -> bool:
        return isinstance(self.obj, Delayed)

    @staticmethod
    def make_delayed(RefCls) -> "Ref":
        return RefCls(uid="", obj=Delayed(), in_memory=False)

    def attach(self, reference: "Ref"):
        assert self.uid == reference.uid and reference.in_memory
        self._obj = reference.obj
        self.in_memory = True

    def _auto_attach(self, shallow: bool = True):
        if not self.in_memory:
            from ..ui.contexts import GlobalContext

            context = GlobalContext.current
            assert context is not None
            assert context.storage is not None
            storage = context.storage
            storage.rel_adapter.mattach(vrefs=[self], shallow=shallow)

    def detached(self) -> "Ref":
        return self.__class__(uid=self.uid, obj=None, in_memory=False)

    @property
    def _uid_suffix(self) -> str:
        return self.uid.split(".")[-1]

    @property
    def _short_uid(self) -> str:
        return self._uid_suffix[:3] + "..."

    def __repr__(self, shorten: bool = False) -> str:
        if self.in_memory:
            obj_repr = repr(self.obj)
            if shorten:
                obj_repr = textwrap.shorten(obj_repr, width=50, placeholder="...")
            if "\n" in obj_repr:
                obj_repr = f'\n{textwrap.indent(obj_repr, "    ")}'
            return f"{self.__class__.__name__}({obj_repr}, uid={self._short_uid})"
        else:
            return f"{self.__class__.__name__}(in_memory=False, uid={self._short_uid})"


class ValueRef(Ref):
    """
    Wraps objects with storage metadata.

    This is the object passed between memoized functions (ops).
    """

    def __init__(self, uid: str, obj: Any, in_memory: bool, transient: bool = False):
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory
        self.transient = transient

    def dump(self) -> "ValueRef":
        if not self.transient:
            return ValueRef(uid=self.uid, obj=self.obj, in_memory=True)
        return ValueRef(
            uid=self.uid, obj=TransientObj(obj=None), in_memory=False, transient=True
        )

    ############################################################################
    ### magic methods forwarding
    ############################################################################
    def _init_magic(self) -> Callable:
        if not Config.enable_ref_magics:
            raise RuntimeError(
                "Ref magic methods (typecasting/comparison operators/binary operators) are disabled; enable with Config.enable_ref_magics = True"
            )
        if Config.warnings:
            logging.warning(
                f"Automatically unwrapping `Ref` to run magic method (typecasting/comparison operators/binary operators)."
            )
        from .wrapping import unwrap

        self._auto_attach(shallow=True)
        return unwrap

    ### typecasting
    def __bool__(self) -> bool:
        self._init_magic()
        return self.obj.__bool__()

    def __int__(self) -> int:
        self._init_magic()
        return self.obj.__int__()

    def __index__(self) -> int:
        self._init_magic()
        return self.obj.__index__()

    ### comparison
    def __lt__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__lt__(unwrap(other))

    def __le__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__le__(unwrap(other))

    def __eq__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__eq__(unwrap(other))

    def __hash__(self) -> int:
        return id(self)

    def __ne__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__ne__(unwrap(other))

    def __gt__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__gt__(unwrap(other))

    def __ge__(self, other: Any) -> bool:
        unwrap = self._init_magic()
        return self.obj.__ge__(unwrap(other))

    ### binary operations
    def __add__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__add__(unwrap(other))

    def __sub__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__sub__(unwrap(other))

    def __mul__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__mul__(unwrap(other))

    def __floordiv__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__floordiv__(unwrap(other))

    def __truediv__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__truediv__(unwrap(other))

    def __mod__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__mod__(unwrap(other))

    def __or__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__or__(unwrap(other))

    def __and__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__and__(unwrap(other))

    def __xor__(self, other: Any) -> Any:
        unwrap = self._init_magic()
        return self.obj.__xor__(unwrap(other))


################################################################################
### calls
################################################################################
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
        inputs: Dict[str, Ref],
        outputs: List[Ref],
        func_op: "FuncOp",
        transient: bool,
        semantic_version: Optional[str] = None,
        content_version: Optional[str] = None,
    ):
        self.uid = uid
        self.semantic_version = semantic_version
        self.content_version = content_version
        self.inputs = inputs
        self.outputs = outputs
        self.transient = transient  # if outputs contain transient objects
        self.func_op = FuncOp._from_data(sig=func_op.sig, f=func_op.func)
        self.func_op.func = None

    def __repr__(self) -> str:
        tuples: List[Tuple[str, str]] = [
            ("uid", self.uid),
        ]
        if self.semantic_version is not None:
            tuples.append(("semantic_version", self.semantic_version))
        if self.content_version is not None:
            tuples.append(("content_version", self.content_version))
        tuples.extend(
            [
                ("inputs", textwrap.shorten(str(self.inputs), width=80)),
                ("outputs", textwrap.shorten(str(self.outputs), width=80)),
                ("func_op", self.func_op),
            ]
        )
        data_str = ",\n".join([f"    {k}={v}" for k, v in tuples])
        return f"Call(\n{data_str}\n)"

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
            if column not in output_columns and column not in Config.special_call_cols
        ]
        process_boolean = lambda x: True if x == "1" else False
        return Call(
            uid=row.column(Config.uid_col)[0].as_py(),
            semantic_version=row.column(Config.semantic_version_col)[0].as_py(),
            content_version=row.column(Config.content_version_col)[0].as_py(),
            transient=process_boolean(row.column(Config.transient_col)[0].as_py()),
            inputs={
                k: Ref.from_uid(uid=row.column(k)[0].as_py()) for k in input_columns
            },
            outputs=[
                Ref.from_uid(uid=row.column(k)[0].as_py())
                for k in sorted(output_columns, key=parse_output_idx)
            ],
            func_op=func_op,
        )

    def set_input_values(self, inputs: Dict[str, Ref]) -> "Call":
        res = copy.deepcopy(self)
        assert set(inputs.keys()) == set(res.inputs.keys())
        for k, v in inputs.items():
            current = res.inputs[k]
            current._obj = v.obj
            current.in_memory = True
        return res

    def set_output_values(self, outputs: List[Ref]) -> "Call":
        res = copy.deepcopy(self)
        assert len(outputs) == len(res.outputs)
        for i, v in enumerate(outputs):
            current = res.outputs[i]
            current._obj = v.obj
            current.in_memory = True
        return res

    def detached(self) -> "Call":
        return Call(
            uid=self.uid,
            inputs={k: v.detached() for k, v in self.inputs.items()},
            outputs=[v.detached() for v in self.outputs],
            func_op=self.func_op,
            semantic_version=self.semantic_version,
            content_version=self.content_version,
            transient=self.transient,
        )


################################################################################
### ops
################################################################################
class FuncOp:
    """
    Operation that models function execution.

    The `is_synchronized` attribute is responsible for keeping track of whether
    this operation has been connected to the storage.

    The synchronization process is responsible for verifying that the function
    signature last stored is compatible with the current signature, and
    performing the necessary updates to the stored signature.

    See also:
        - `mandala.core.sig.Signature`
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        sig: Optional[Signature] = None,
        version: Optional[int] = None,
        ui_name: Optional[str] = None,
        is_super: bool = False,
        n_outputs: Optional[int] = None,
        _is_builtin: bool = False,
    ):
        self.is_super = is_super
        self._is_builtin = _is_builtin
        if func is None:
            self.sig = sig
            self.py_sig = None
            self.func = None
        else:
            self.func = func
            if Config.has_torch and isinstance(func, torch.jit.ScriptFunction):
                sig, py_sig = sig_from_jit_script(self.func, version=version)
                ui_name = sig.ui_name
            else:
                py_sig = inspect.signature(self.func)
                ui_name = self.func.__name__ if ui_name is None else ui_name
            self.py_sig = py_sig
            self.sig = Signature.from_py(
                sig=self.py_sig, name=ui_name, version=version, _is_builtin=_is_builtin
            )
        if n_outputs is not None:
            self.sig.n_outputs = n_outputs

    @property
    def input_annotations(self) -> Dict[str, Any]:
        return _get_arg_annotations(func=self.func, support=list(self.sig.input_names))

    @property
    def input_types(self) -> Dict[str, Type]:
        return {k: Type.from_annotation(v) for k, v in self.input_annotations.items()}

    @property
    def output_annotations(self) -> List[Any]:
        return _get_return_annotations(func=self.func, support_size=self.sig.n_outputs)

    @property
    def output_types(self) -> List[Type]:
        return [Type.from_annotation(a) for a in self.output_annotations]

    def _set_func(self, func: Callable) -> None:
        # set the function only
        self.func = func
        self.py_sig = inspect.signature(self.func)

    def compute(
        self,
        inputs: Dict[str, Any],
        tracer: Optional[TracerABC] = None,
    ) -> Tuple[List[Any], Optional[TracerABC]]:
        if tracer is not None:
            with tracer:
                if isinstance(tracer, DecTracer):
                    node = tracer.register_call(func=self.func)
                result = self.func(**inputs)
                if isinstance(tracer, DecTracer):
                    tracer.register_return(node=node)
        else:
            result = self.func(**inputs)
        return _postprocess_outputs(sig=self.sig, result=result), tracer

    @staticmethod
    def _from_data(sig: Signature, f: Optional[Callable] = None) -> "Op":
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

    def get_call_uid(self, wrapped_inputs: Dict[str, Ref]) -> str:
        # get call UID using *internal names* to guarantee the same UID will be
        # assigned regardless of renamings
        hashable_input_uids = {}
        for k, v in wrapped_inputs.items():
            # ignore the inputs that were added to the function and have their
            # default values
            internal_k = self.sig.ui_to_internal_input_map[k]
            if internal_k in self.sig._new_input_defaults_uids:
                if self.sig._new_input_defaults_uids[internal_k] == v.uid:
                    continue
            hashable_input_uids[internal_k] = v.uid
        call_uid = Hashing.get_content_hash(
            obj=[
                hashable_input_uids,
                self.sig.versioned_internal_name,
            ]
        )
        return call_uid


################################################################################
### wrapping
################################################################################
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
    elif not isinstance(obj, TransientObj):
        uid = Hashing.get_content_hash(obj) if uid is None else uid
        return ValueRef(uid=uid, obj=obj, in_memory=True)
    else:
        if obj.unhashable:
            uid = get_uid()
        else:
            uid = Hashing.get_content_hash(obj.obj) if uid is None else uid
        uid = get_transient_uid(content_hash=uid)
        return ValueRef(uid=uid, obj=obj.obj, in_memory=True, transient=True)


from .builtins_ import ListRef, DictRef, SetRef

TP_TO_CLS = {
    AnyType: ValueRef,
    ListType: ListRef,
    DictType: DictRef,
    SetType: SetRef,
}


def make_delayed(tp: Type) -> Ref:
    return TP_TO_CLS[type(tp)](uid="", obj=Delayed(), in_memory=False)
