from typing import Type
from .config import Config, dump_output_name, parse_output_idx
from ..common_imports import *
from .utils import Hashing
from .sig import (
    Signature,
    _postprocess_outputs,
    _get_arg_annotations,
    _get_return_annotations,
)
from .tps import Type, AnyType, ListType, DictType, SetType
from .deps import DependencyGraph, Tracer, TerminalData

if Config.has_torch:
    import torch
    from .integrations import sig_from_jit_script


class Delayed:
    pass


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
            from ..ui.main import GlobalContext

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

    def __repr__(self) -> str:
        if self.in_memory:
            return f"{self.__class__.__name__}({self.obj}, uid={self._short_uid})"
        else:
            return f"{self.__class__.__name__}(in_memory=False, uid={self._short_uid})"


class ValueRef(Ref):
    """
    Wraps objects with storage metadata.

    This is the object passed between memoized functions (ops).
    """

    def __init__(self, uid: str, obj: Any, in_memory: bool):
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory

    def dump(self) -> "ValueRef":
        return ValueRef(uid=self.uid, obj=self.obj, in_memory=True)

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
    ):
        self.uid = uid
        self.inputs = inputs
        self.outputs = outputs
        self.func_op = FuncOp._from_data(sig=func_op.sig, f=func_op.func)
        self.func_op.func = None

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
            assert v.in_memory and not current.in_memory
            current._obj = v.obj
            current.in_memory = True
        return res

    def set_output_values(self, outputs: List[Ref]) -> "Call":
        res = copy.deepcopy(self)
        assert len(outputs) == len(res.outputs)
        for i, v in enumerate(outputs):
            current = res.outputs[i]
            assert v.in_memory and not current.in_memory
            current._obj = v.obj
            current.in_memory = True
        return res


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
        deps_root: Optional[Path] = None,
    ) -> Tuple[List[Any], Optional[DependencyGraph]]:
        """
        Computes the function on the given *unwrapped* inputs. Returns a list of
        `self.sig.n_outputs` outputs (after checking they are the number
        expected by the interface).

        This expects the inputs to be named using *internal* input names.
        """
        if deps_root is not None:
            graph = DependencyGraph()
            # ds = DependencyState(roots=[deps_root, Config.mandala_path], origin=(self.sig.internal_name, self.sig.version))
            tracer = Tracer(
                graph=graph, strict=True, paths=[deps_root, Config.mandala_path]
            )
            if sys.gettrace() is None:
                with tracer:
                    result = self.func(**inputs)
            else:
                current_trace = sys.gettrace()
                Tracer.break_signal(
                    data=Tracer.generate_terminal_data(
                        func=self.func,
                        internal_name=self.sig.internal_name,
                        version=self.sig.version,
                    )
                )
                sys.settrace(None)
                with tracer:
                    result = self.func(**inputs)
                sys.settrace(current_trace)
        else:
            result = self.func(**inputs)
            graph = None
        return _postprocess_outputs(sig=self.sig, result=result), graph

    # async def compute_async(
    #     self,
    #     inputs: Dict[str, Any],
    #     deps_root: Optional[Path] = None,
    #     ignore_dependency_tracking_errors: bool = False,
    # ) -> Tuple[List[Any], Optional[DependencyState]]:
    #     """
    #     Computes the function on the given *unwrapped* inputs. Returns a list of
    #     `self.sig.n_outputs` outputs (after checking they are the number
    #     expected by the interface).

    #     This expects the inputs to be named using *internal* input names.
    #     """
    #     if deps_root is not None:
    #         raise NotImplementedError()
    #     result = (
    #         await self.func(**inputs)
    #         if inspect.iscoroutinefunction(self.func)
    #         else self.func(**inputs)
    #     )
    #     ds = None
    #     return _postprocess_outputs(sig=self.sig, result=result), ds

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
    else:
        uid = Hashing.get_content_hash(obj) if uid is None else uid
        return ValueRef(uid=uid, obj=obj, in_memory=True)


from .builtins_ import ListRef, DictRef, SetRef

TP_TO_CLS = {
    AnyType: ValueRef,
    ListType: ListRef,
    DictType: DictRef,
    SetType: SetRef,
}


def make_delayed(tp: Type) -> Ref:
    return TP_TO_CLS[type(tp)](uid="", obj=Delayed(), in_memory=False)
