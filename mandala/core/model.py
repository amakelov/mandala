from collections.abc import Sequence

from .config import Config, dump_output_name, parse_output_idx
from ..common_imports import *
from .utils import Hashing
from .sig import Signature, _postprocess_outputs
from .tps import Type, AnyType, ListType
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
        self.obj = obj
        self.in_memory = in_memory

    @staticmethod
    def from_uid(uid: str) -> "Ref":
        if ListRef._is_list_uid(uid=uid):
            return ListRef(uid=uid, obj=None, in_memory=False)
        else:
            return ValueRef(uid=uid, obj=None, in_memory=False)

    def is_delayed(self) -> bool:
        return isinstance(self.obj, Delayed)

    @staticmethod
    def make_delayed(RefCls: type["Ref"]) -> "Ref":
        return RefCls(uid="", obj=Delayed(), in_memory=False)


class ValueRef(Ref):
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
        return ValueRef(uid=self.uid, obj=None, in_memory=False)

    def attach(self, reference: "ValueRef"):
        assert self.uid == reference.uid
        self.obj = reference.obj
        self.in_memory = True

    def dump(self) -> "ValueRef":
        return ValueRef(uid=self.uid, obj=self.obj, in_memory=True)


class ListRef(Ref, Sequence):
    def __init__(self, uid: str, obj: Optional[List[Ref]], in_memory: bool):
        assert ListRef._is_list_uid(uid=uid)
        self.uid = uid
        self.obj = obj
        self.in_memory = in_memory

    @staticmethod
    def _is_list_uid(uid: str) -> bool:
        return uid.startswith("__list__.")

    def __repr__(self) -> str:
        if self.in_memory:
            return f"ListRef({self.obj}, uid={self.uid})"
        else:
            return f"ListRef(in_memory=False, uid={self.uid})"

    def detached(self) -> "ListRef":
        return ListRef(uid=self.uid, obj=None, in_memory=False)

    def attach(self, reference: "ListRef"):
        assert self.uid == reference.uid
        self.obj = reference.obj
        self.in_memory = True

    def dump(self) -> "ListRef":
        return ListRef(
            uid=self.uid, obj=[vref.detached() for vref in self.obj], in_memory=True
        )

    ############################################################################
    ### list interface
    ############################################################################
    def __getitem__(self, idx: int) -> Ref:
        assert self.in_memory
        return self.obj[idx]

    def __iter__(self):
        assert self.in_memory
        return iter(self.obj)

    def __len__(self) -> int:
        assert self.in_memory
        return len(self.obj)


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
            current.obj = v.obj
            current.in_memory = True
        return res

    def set_output_values(self, outputs: List[Ref]) -> "Call":
        res = copy.deepcopy(self)
        assert len(outputs) == len(res.outputs)
        for i, v in enumerate(outputs):
            current = res.outputs[i]
            assert v.in_memory and not current.in_memory
            current.obj = v.obj
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

    def _set_func(self, func: Callable) -> None:
        # set the function only
        self.func = func
        self.py_sig = inspect.signature(self.func)

    def _generate_terminal_data(self) -> TerminalData:
        module_name, func_name = Tracer.get_func_key(func=self.func)
        data = TerminalData(
            internal_name=self.sig.internal_name,
            version=self.sig.version,
            module_name=module_name,
            func_name=func_name,
        )
        return data

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
                Tracer.break_signal(data=self._generate_terminal_data())
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
### builtins
################################################################################
class Builtins:
    @staticmethod
    def list_func(lst: List[Any], elt: Any, idx: Any):
        assert lst[idx] is elt

    @staticmethod
    def dict_func(dct: Dict[str, Any], key: str, val: Any):
        assert dct[key] is val

    @staticmethod
    def make_list_uid(uid: str) -> str:
        return f"__list__.{uid}"

    list_op = FuncOp(func=list_func, _is_builtin=True, version=0, ui_name="__list__")
    # dict_op = FuncOp(func=dict_func, _is_builtin=True, version=0, ui_name="__dict__")

    OPS = [list_op]

    @staticmethod
    def construct_list(elts: List[Ref]) -> Tuple[ListRef, List[Call]]:
        uid = Builtins.make_list_uid(
            Hashing.get_content_hash([elt.uid for elt in elts])
        )
        lst = ListRef(uid=uid, obj=elts, in_memory=True)
        idxs = [wrap(idx) for idx in range(len(elts))]
        calls = []
        for elt, idx in zip(elts, idxs):
            wrapped_inputs = {"lst": lst, "elt": elt, "idx": idx}
            call_uid = Builtins.list_op.get_call_uid(wrapped_inputs=wrapped_inputs)
            calls.append(
                Call(
                    uid=call_uid,
                    inputs=wrapped_inputs,
                    outputs=[],
                    func_op=Builtins.list_op,
                )
            )
        return lst, calls


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


def wrap_constructive(obj: Any, annotation: Any) -> Tuple[Ref, List[Call]]:
    tp = Type.from_annotation(annotation=annotation)
    # check the type
    if isinstance(tp, ListType) and not (
        isinstance(obj, list) or isinstance(obj, ListRef)
    ):
        raise ValueError(f"Expecting a list, got {type(obj)}")
    calls = []
    if isinstance(obj, Ref):
        return obj, calls
    if isinstance(tp, AnyType):
        return wrap(obj=obj), calls
    elif isinstance(tp, ListType):
        assert isinstance(obj, list)
        elt_results = [wrap_constructive(elt, annotation=tp.elt_type) for elt in obj]
        elt_calls = [c for _, cs in elt_results for c in cs]
        elt_vrefs = [v for v, _ in elt_results]
        calls.extend(elt_calls)
        result, construction_calls = Builtins.construct_list(elts=elt_vrefs)
        calls.extend(construction_calls)
        return result, calls
    else:
        raise ValueError()


def wrap_dict(
    objs: Dict[str, Any], annotations: Dict[str, Any]
) -> Tuple[Dict[str, Ref], List[Call]]:
    calls = []
    wrapped_objs = {}
    for k, v in objs.items():
        wrapped_obj, wrapping_calls = wrap_constructive(
            obj=v, annotation=annotations[k]
        )
        wrapped_objs[k] = wrapped_obj
        calls.extend(wrapping_calls)
    return wrapped_objs, calls


def wrap_list(objs: List[Any], annotations: List[Any]) -> Tuple[List[Ref], List[Call]]:
    calls = []
    wrapped_objs = []
    for i, v in enumerate(objs):
        wrapped_obj, wrapping_calls = wrap_constructive(
            obj=v, annotation=annotations[i]
        )
        wrapped_objs.append(wrapped_obj)
        calls.extend(wrapping_calls)
    return wrapped_objs, calls


################################################################################
### unwrapping
################################################################################
T = TypeVar("T")


def unwrap(obj: Union[T, Ref], through_collections: bool = True) -> T:
    """
    If an object is a `ValueRef`, returns the wrapped object; otherwise, return
    the object itself.

    If `through_collections` is True, recursively unwraps objects in lists,
    tuples, sets, and dict values.
    """
    if isinstance(obj, ValueRef):
        return unwrap(obj.obj, through_collections=through_collections)
    elif isinstance(obj, ListRef):
        return [unwrap(elt, through_collections=through_collections) for elt in obj.obj]
        # return unwrap(obj.obj, through_collections=through_collections)
    elif isinstance(obj, tuple) and through_collections:
        return tuple(unwrap(v, through_collections=through_collections) for v in obj)
    elif isinstance(obj, set) and through_collections:
        return {unwrap(v, through_collections=through_collections) for v in obj}
    elif isinstance(obj, list) and through_collections:
        return [unwrap(v, through_collections=through_collections) for v in obj]
    elif isinstance(obj, dict) and through_collections:
        return {
            k: unwrap(v, through_collections=through_collections)
            for k, v in obj.items()
        }
    else:
        return obj