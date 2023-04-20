from typing import Type
import textwrap
from .config import Config, parse_output_idx, dump_output_name, is_output_name, MODES
from ..common_imports import *
from .utils import Hashing, get_uid, get_full_uid, parse_full_uid
from .sig import (
    Signature,
    _postprocess_outputs,
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
    def __init__(self, uid: str, obj: Any, in_memory: bool, transient: bool):
        self.uid = uid
        self._causal_uid = None
        self._obj = obj
        self.in_memory = in_memory
        self.transient = transient
        # runtime only
        self._query: Optional[ValQuery] = None

    @property
    def causal_uid(self) -> str:
        return self._causal_uid

    @causal_uid.setter
    def causal_uid(self, causal_uid: str):
        if self.causal_uid is not None and self.causal_uid != causal_uid:
            raise ValueError("causal_uid already set")
        self._causal_uid = causal_uid

    @property
    def full_uid(self) -> str:
        return get_full_uid(uid=self.uid, causal_uid=self.causal_uid)

    @staticmethod
    def parse_full_uid(full_uid: str) -> Tuple[str, str]:
        return parse_full_uid(full_uid=full_uid)

    @property
    def obj(self) -> Any:
        return self._obj

    @staticmethod
    def from_full_uid(full_uid: str) -> "Ref":
        uid, causal_uid = Ref.parse_full_uid(full_uid=full_uid)
        res = Ref.from_uid(uid=uid)
        return res

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

            context = GlobalContext.current
            assert context is not None
            if context.mode != MODES.run:
                return
            assert context.storage is not None
            storage = context.storage
            storage.rel_adapter.mattach(vrefs=[self], shallow=shallow)
            causify_down(ref=self, start=self.causal_uid, stop_at_causal=False)

    def detached(self) -> "Ref":
        return self.__class__(uid=self.uid, obj=None, in_memory=False)

    def unlinked(self, keep_causal: bool) -> "Ref":
        raise NotImplementedError

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

    @property
    def query(self) -> "ValQuery":
        if self._query is None:
            raise ValueError("Ref has no query")
        return self._query

    def pin(self, *values):
        assert self._query is not None
        if len(values) == 0:
            constraint = [self.full_uid]
        else:
            constraint = self.query.get_constraint(*values)
        self._query.constraint = constraint

    def unpin(self):
        assert self._query is not None
        self._query.constraint = None


class ValueRef(Ref):
    def __init__(self, uid: str, obj: Any, in_memory: bool, transient: bool = False):
        super().__init__(uid=uid, obj=obj, in_memory=in_memory, transient=transient)

    def dump(self) -> "ValueRef":
        if not self.transient:
            return ValueRef(uid=self.uid, obj=self.obj, in_memory=True)
        return ValueRef(
            uid=self.uid, obj=TransientObj(obj=None), in_memory=False, transient=True
        )

    def unlinked(self, keep_causal: bool) -> "Ref":
        res = ValueRef(
            uid=self.uid,
            obj=self.obj,
            in_memory=self.in_memory,
            transient=self.transient,
        )
        if keep_causal:
            res.causal_uid = self.causal_uid
        return res

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
    def __init__(
        self,
        uid: str,
        inputs: Dict[str, Ref],
        outputs: List[Ref],
        func_op: "FuncOp",
        transient: bool,
        causal_uid: Optional[str] = None,
        semantic_version: Optional[str] = None,
        content_version: Optional[str] = None,
    ):
        self.func_op = func_op.detached()
        self.uid = uid
        self.semantic_version = semantic_version
        self.content_version = content_version
        self.inputs = inputs
        self.outputs = outputs
        self.transient = transient  # if outputs contain transient objects
        if causal_uid is None:
            input_uids = {k: v.uid for k, v in self.inputs.items()}
            input_causal_uids = {k: v.causal_uid for k, v in self.inputs.items()}
            assert all([v is not None for v in input_causal_uids.values()])
            causal_uid = self.func_op.get_call_causal_uid(
                input_uids=input_uids,
                input_causal_uids=input_causal_uids,
                semantic_version=semantic_version,
            )
        self.causal_uid = causal_uid
        self._func_query = None

    @property
    def full_uid(self) -> str:
        return f"{self.uid}.{self.causal_uid}"

    def link(
        self,
        orientation: Optional[str] = None,
    ):
        if self._func_query is not None:
            return
        input_types, output_types = self.func_op.input_types, self.func_op.output_types
        for k, v in self.inputs.items():
            prepare_query(ref=v, tp=input_types[k])
        for i, v in enumerate(self.outputs):
            prepare_query(ref=v, tp=output_types[i])
        outputs = {dump_output_name(i): v.query for i, v in enumerate(self.outputs)}
        self._func_query = FuncQuery.link(
            inputs={k: v.query for k, v in self.inputs.items()},
            func_op=self.func_op.detached(),
            outputs=outputs,
            orientation=orientation,
            constraint=None,
        )

    def unlink(self):
        assert self._func_query is not None
        self._func_query.unlink()
        self._func_query = None

    @property
    def func_query(self) -> "FuncQuery":
        assert self._func_query is not None
        return self._func_query

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
        output_columns = [column for column in columns if is_output_name(column)]
        input_columns = [
            column
            for column in columns
            if column not in output_columns and column not in Config.special_call_cols
        ]
        process_boolean = lambda x: True if x == "1" else False
        return Call(
            uid=row.column(Config.uid_col)[0].as_py(),
            causal_uid=row.column(Config.causal_uid_col)[0].as_py(),
            semantic_version=row.column(Config.semantic_version_col)[0].as_py(),
            content_version=row.column(Config.content_version_col)[0].as_py(),
            transient=process_boolean(row.column(Config.transient_col)[0].as_py()),
            inputs={
                k: Ref.from_full_uid(full_uid=row.column(k)[0].as_py())
                for k in input_columns
            },
            outputs=[
                Ref.from_full_uid(full_uid=row.column(k)[0].as_py())
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
            causal_uid=self.causal_uid,
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
    def __init__(
        self,
        func: Optional[Callable] = None,
        sig: Optional[Signature] = None,
        version: Optional[int] = None,
        ui_name: Optional[str] = None,
        is_super: bool = False,
        n_outputs_override: Optional[int] = None,
        _is_builtin: bool = False,
    ):
        self.is_super = is_super
        self._is_builtin = _is_builtin
        if func is None:
            self.sig = sig
            self.py_sig = None
            self._func = None
            self._module = None
            self._qualname = None
        else:
            self._func = func
            self._module = func.__module__
            self._qualname = func.__qualname__
            if Config.has_torch and isinstance(func, torch.jit.ScriptFunction):
                sig, py_sig = sig_from_jit_script(self._func, version=version)
                ui_name = sig.ui_name
            else:
                py_sig = inspect.signature(self._func)
                ui_name = self._func.__name__ if ui_name is None else ui_name
            self.py_sig = py_sig
            self.sig = Signature.from_py(
                sig=self.py_sig, name=ui_name, version=version, _is_builtin=_is_builtin
            )
        self.n_outputs_override = n_outputs_override
        if n_outputs_override is not None:
            self.sig.n_outputs = n_outputs_override
            self.sig.output_annotations = [Any] * n_outputs_override

    @property
    def func(self) -> Callable:
        assert self._func is not None
        return self._func

    @property
    def is_builtin(self) -> bool:
        return self._is_builtin

    @func.setter
    def func(self, func: Optional[Callable]):
        self._func = func
        if func is not None:
            self.py_sig = inspect.signature(self._func)

    @property
    def input_annotations(self) -> Dict[str, Any]:
        assert self.sig is not None
        return self.sig.input_annotations

    @property
    def input_types(self) -> Dict[str, Type]:
        return {k: Type.from_annotation(v) for k, v in self.input_annotations.items()}

    @property
    def output_annotations(self) -> List[Any]:
        assert self.sig is not None
        return self.sig.output_annotations

    @property
    def output_types(self) -> List[Type]:
        return [Type.from_annotation(a) for a in self.output_annotations]

    def compute(
        self,
        inputs: Dict[str, Any],
        tracer: Optional[TracerABC] = None,
    ) -> List[Any]:
        if tracer is not None:
            with tracer:
                if isinstance(tracer, DecTracer):
                    node = tracer.register_call(func=self.func)
                result = self.func(**inputs)
                if isinstance(tracer, DecTracer):
                    tracer.register_return(node=node)
        else:
            result = self.func(**inputs)
        return _postprocess_outputs(sig=self.sig, result=result)

    @staticmethod
    def _from_data(
        sig: Signature,
        func: Callable,
    ) -> "FuncOp":
        """
        Create a `FuncOp` object based on a signature and maybe a function. For
        internal use only.
        """
        res = FuncOp(func=func, version=sig.version, ui_name=sig.ui_name)
        res.sig = sig
        return res

    @staticmethod
    def _from_sig(sig: Signature) -> "FuncOp":
        return FuncOp(func=None, sig=sig)

    def detached(self) -> "FuncOp":
        if self._func is None:
            return copy.deepcopy(self)
        result = FuncOp(
            func=None,
            sig=copy.deepcopy(self.sig),
            version=self.sig.version,
            ui_name=self.sig.ui_name,
            is_super=self.is_super,
            n_outputs_override=self.n_outputs_override,
            _is_builtin=self._is_builtin,
        )
        result._module = self._module
        result._qualname = self._qualname
        result.py_sig = self.py_sig  # signature objects are immutable
        return result

    def get_active_inputs(self, input_uids: Dict[str, str]) -> Dict[str, str]:
        """
        Return a dict of external -> internal input names for inputs that are
        not set to their default values.
        """
        res = {}
        for k, v in input_uids.items():
            internal_k = self.sig.ui_to_internal_input_map[k]
            if internal_k in self.sig._new_input_defaults_uids:
                internal_uid = Ref.parse_full_uid(
                    full_uid=self.sig._new_input_defaults_uids[internal_k]
                )[0]
                if internal_uid == v:
                    continue
            res[k] = internal_k
        return res

    def get_call_causal_uid(
        self,
        input_uids: Dict[str, str],
        input_causal_uids: Dict[str, str],
        semantic_version: Optional[str],
    ) -> str:
        active_inputs = self.get_active_inputs(input_uids=input_uids)
        return Hashing.get_content_hash(
            obj=[
                {
                    active_inputs[k]: v
                    for k, v in input_causal_uids.items()
                    if k in active_inputs.keys()
                },
                semantic_version,
                self.sig.versioned_internal_name,
            ]
        )

    def get_pre_call_uid(self, input_uids: Dict[str, str]) -> str:
        # get call UID using *internal names* to guarantee the same UID will be
        # assigned regardless of renamings
        active_inputs = self.get_active_inputs(input_uids=input_uids)
        hashable_input_uids = {
            active_inputs[k]: v
            for k, v in input_uids.items()
            if k in active_inputs.keys()
        }
        call_uid = Hashing.get_content_hash(
            obj=[
                hashable_input_uids,
                self.sig.versioned_internal_name,
            ]
        )
        return call_uid

    def get_call_uid(self, pre_call_uid: str, semantic_version: Optional[str]) -> str:
        return Hashing.get_content_hash((pre_call_uid, semantic_version))


################################################################################
### wrapping
################################################################################
def wrap_atom(obj: Any, uid: Optional[str] = None) -> ValueRef:
    """
    Wraps a value as a `ValueRef`, if it isn't one already.

    The uid is either explicitly set, or a content hash is generated. Note that
    content hashing may take non-trivial time for large objects. When `obj` is
    already a `ValueRef` and `uid` is provided, an error is raised.
    """
    if isinstance(obj, Ref) and not isinstance(obj, ValueRef):
        raise ValueError(f"Cannot wrap {obj} as a ValueRef")
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


from .builtins_ import ListRef, DictRef, SetRef, Builtins
from ..queries.weaver import ValQuery, StructOrientations, FuncQuery, prepare_query
from ..ui.contexts import GlobalContext
from .wrapping import causify_down, causify_atom
