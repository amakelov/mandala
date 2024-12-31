import textwrap
import functools
from collections import deque
from .common_imports import *
from .common_imports import sess
from typing import Literal
from .tps import *
from .config import *
from .utils import (
    parse_output_name,
)
from .utils import serialize, deserialize, get_content_hash

################################################################################
### model
################################################################################


class Ref:
    """
    Base class, should not be instantiated directly.
    """

    def __init__(self, cid: str, hid: str, in_memory: bool, obj: Optional[Any]) -> None:
        self.cid = cid
        self.hid = hid
        self.obj = obj
        self.in_memory = in_memory

    def with_hid(self, hid: str) -> "Ref":
        return type(self)(cid=self.cid, hid=hid, in_memory=self.in_memory, obj=self.obj)

    def __repr__(self) -> str:
        if self.in_memory:
            obj_repr = repr(self.obj)
            return f"Ref({obj_repr}, hid={self.hid[:3]}...)"
        else:
            return f"Ref(hid={self.hid[:3]}..., in_memory={self.in_memory})"

    def __hash__(self) -> int:
        return hash(self.hid)

    def detached(self) -> "Ref":
        """
        Return a *copy* of this ref
        """
        return type(self)(cid=self.cid, hid=self.hid, in_memory=False, obj=None)

    def attached(self, obj: Any) -> "Ref":
        """
        Return a *copy* of this ref that is in memory (taking the object from
        `obj`)
        """
        return type(self)(cid=self.cid, hid=self.hid, in_memory=True, obj=obj)

    def shallow_copy(self) -> "Ref":
        return type(self)(
            cid=self.cid, hid=self.hid, in_memory=self.in_memory, obj=self.obj
        )

class AtomRef(Ref):
    def __repr__(self) -> str:
        return "Atom" + super().__repr__()


################################################################################
### wrappers for values that should be treated in special ways
###############################################################################
class _Ignore:
    """
    Used to mark values that should be ignored by the storage, but still
    passed to the function. In this case, the wrapped value `value` will be 
    directly passed to the function.
    """
    def __init__(self, value: Any = None) -> None:
        self.value = value

class _NewArgDefault(_Ignore):
    """
    Like `_Ignore`, but should be used as default value when adding new
    arguments to functions to enable backwards compatibility.

    Has the additional effect that, if the default value is
    `_NewArgDefault(obj)`, then when the function is called with `obj`, the
    value of the argument will also be ignored by the storage.
    """
    pass


class ValuePointer:
    """
    Replace an object by a human-readable name from the point of view of the
    storage.

    A value wrapped in a `ValuePointer` will:
    - be identified via its `id` when hashing (when computing both the content
    and history IDs)
    - will not be saved in the storage
    - will be replaced by the underlying `obj` when passed to a memoized
    function 

    This is useful for passing large, complex objects (e.g., a machine learning
    dataset or model) that is immutable over the course of a project and is 
    either not serializable, or stored elsewhere and thus we don't need to 
    duplicate it in the storage.
    
    """
    def __init__(self, id: str, obj: Any):
        if not isinstance(id, str) or not id:
            raise ValueError("The `id` must be a non-empty string.")
        self.id = id
        self.obj = obj
    
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError("ValuePointer objects should not be compared.")
    
    def __hash__(self) -> int:
        raise NotImplementedError("ValuePointer objects should not be hashed.")
    
    def __repr__(self) -> str:
        obj_repr = repr(self.obj)
        return f"ValuePointer({self.id!r}, {obj_repr})"

    
T = TypeVar("T")
def Ignore(value: T = None) -> T:
    return _Ignore(value)

def NewArgDefault(value: T = None) -> T:
    return _NewArgDefault(value)

def unwrap_special_value(obj: ValuePointer | _Ignore | Any) -> Any:
    if isinstance(obj, ValuePointer):
        return obj.obj
    elif isinstance(obj, _Ignore):
        return obj.value
    else:
        return obj

################################################################################
### ops and calls
################################################################################
class Op:
    def __init__(
        self,
        name: str,
        f: Callable,
        nout: Union[Literal["var", "auto"], int] = "auto",
        output_names: Optional[List[str]] = None,
        version: Optional[int] = 0,
        ignore_args: Optional[Tuple[str,...]] = None, # ignore these arguments when hashing
        __structural__: bool = False,
        __allow_side_effects__: bool = False,
    ) -> None:
        self.name = name
        self.nout = nout
        self.version = version
        self.output_names = output_names
        self.ignore_args = ignore_args
        self.__structural__ = __structural__
        self.__allow_side_effects__ = __allow_side_effects__
        self.f = f
        #! make sure there's no overlap between the input and output names
        if f is not None:
            input_names = list(inspect.signature(f).parameters.keys())
            if output_names is not None:
                assert not any(name in output_names for name in input_names), (
                    f"Output names clash with input names for op {name}."
                )
            else:
                # to be on the safe side, check if any input name starts with "output_"
                assert not any(name.startswith("output_") for name in input_names), (
                    f"Output names must be specified for op {name} because the input names "
                    "contain 'output_'. Alternatively, rename the input arguments to not start with 'output_'."
                )

    def __repr__(self) -> str:
        return f"Op({self.name}, nout={self.nout}, output_names={self.output_names}, version={self.version})"

    @property
    def id(self) -> str:
        return self.name

    def _get_hashable_inputs(self, inputs: Dict[str, Ref]) -> Dict[str, Any]:
        return {k: v for k, v in inputs.items() if not isinstance(v.obj, _Ignore)}

    def get_call_history_id(self,
                            inputs: Dict[str, Ref],
                            semantic_version: Optional[str] = None,
                            ) -> str:
        """
        Combine the inputs' history IDs, the name of the op, and the semantic
        version to get a unique id for the call history.
        """
        hashable_inputs = self._get_hashable_inputs(inputs)
        obj = ({k: v.hid for k, v in hashable_inputs.items()}, self.name, self.version)
        if semantic_version is not None:
            obj = obj + (semantic_version,)
        return get_content_hash(obj)

    def get_call_content_id(self, inputs: Dict[str, Ref],
                            semantic_version: Optional[str] = None) -> str:
        hashable_inputs = self._get_hashable_inputs(inputs)
        obj = ({k: v.cid for k, v in hashable_inputs.items()}, self.name, self.version)
        if semantic_version is not None:
            obj = obj + (semantic_version,)
        return get_content_hash(obj)
    
    def get_pre_call_id(self, inputs: Dict[str, Ref]) -> str:
        """
        Combine the inputs' content IDs and the name of the op to get a unique
        id for the pre-call, to be used to search for matching semantic
        versions.
        """
        hashable_inputs = self._get_hashable_inputs(inputs)
        return get_content_hash((self.name, {k: v.cid for k, v in hashable_inputs.items()}))

    def get_output_history_ids(
        self, call_history_id: str, output_names: List[str]
    ) -> Dict[str, str]:
        return {k: get_content_hash((call_history_id, k)) for k in output_names}

    def get_ordered_outputs(self, output_dict: Dict[str, Any]) -> Tuple[Any, ...]:
        if (
            self.output_names is None
        ):  # output names must be generic output_0, output_1, etc.
            output_dict_by_int = {
                parse_output_name(name): value for name, value in output_dict.items()
            }
            return tuple([output_dict_by_int[i] for i in range(len(output_dict))])
        else:  # we use the order of the keys in self.output_names
            return tuple([output_dict[k] for k in self.output_names])

    def detached(self) -> "Op":
        return Op(
            name=self.name,
            f=None,
            nout=self.nout,
            output_names=self.output_names,
            version=self.version,
            __structural__=self.__structural__,
        )

    def __call__(self, *args, **kwargs) -> Union[Tuple[Ref, ...], Ref]:
        if Context.current_context is None:  # act as noop
            return self.f(*args, **kwargs)
        else:
            storage = Context.current_context.storage
            return storage.call(self, args, kwargs, config={"save_calls": True})


class Call:
    def __init__(
        self,
        op: Op,
        cid: str,
        hid: str,
        inputs: Dict[str, Ref],
        outputs: Dict[str, Ref],
        semantic_version: Optional[str] = None,
        content_version: Optional[str] = None,
    ) -> None:
        self.op = op
        self.cid = cid
        self.hid = hid
        self.inputs = inputs
        self.outputs = outputs
        self.semantic_version = semantic_version
        self.content_version = content_version

    def __repr__(self) -> str:
        return f"Call({self.op.name}, hid={self.hid[:3]}...)"

    def detached(self) -> "Call":
        """
        Return the call with the inputs, outputs and op detached.
        """
        return Call(
            op=self.op.detached(),
            cid=self.cid,
            hid=self.hid,
            inputs={k: v.detached() for k, v in self.inputs.items()},
            outputs={k: v.detached() for k, v in self.outputs.items()},
            semantic_version=self.semantic_version,
            content_version=self.content_version,
        )


def wrap_atom(obj: Any, history_id: Optional[str] = None) -> AtomRef:
    """
    Wrap a Python object in an AtomRef. If the object is already a Ref, return
    it unchanged. If `history_id` is not provided, it will be initialized
    from the object's content hash (thereby representing an object without any
    history).

    All operations that wrap a value (whether a collection or an atom) must 
    factor through this function.
    """
    if isinstance(obj, Ref):
        if not isinstance(obj, AtomRef):
            raise ValueError(f"Expected an AtomRef, got {type(obj)}")
        assert history_id is None
        return obj
    if isinstance(obj, ValuePointer):
        # we never directly hash the object, but rather the id
        uid = get_content_hash(obj.id) 
        return AtomRef(cid=uid, hid=uid, in_memory=True, obj=ValuePointer(id=obj.id, obj=None))
    uid = get_content_hash(obj)
    if history_id is None:
        history_id = get_content_hash(uid)
    return AtomRef(cid=uid, hid=history_id, in_memory=True, obj=obj)


################################################################################
### native support for some kinds of collections
################################################################################
class ListRef(Ref):
    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, i: int) -> Ref:
        assert self.in_memory
        return self.obj[i]

    def __repr__(self) -> str:
        return "List" + super().__repr__()

    def shape(self) -> "ListRef":
        return ListRef(
            cid=self.cid,
            hid=self.hid,
            in_memory=True,
            obj=[elt.detached() for elt in self.obj],
        )


class DictRef(Ref):
    """
    For now, we only support dictionaries where keys are strings. It's possible
    to extend keys to be `Ref` objects.
    """
    def __len__(self) -> int:
        return len(self.obj)
    
    def __getitem__(self, key: str) -> Ref:
        assert self.in_memory
        if isinstance(key, str):
            return self.obj[key]
        elif isinstance(key, Ref):
            assert key.in_memory
            return self.obj[key.obj]
        else:
            raise ValueError(key)
    
    def __repr__(self) -> str:
        return "Dict" + super().__repr__()
    
    def items(self) -> Iterable[Tuple[str, Ref]]:
        return self.obj.items()
    
    def values(self) -> Iterable[Ref]:
        return self.obj.values()
    
    def shape(self) -> "DictRef":
        return DictRef(
            cid=self.cid,
            hid=self.hid,
            in_memory=True,
            obj={k: v.detached() for k, v in self.obj.items()},
        )
        

class TupleRef(Ref):
    pass


class SetRef(Ref):
    def __repr__(self) -> str:
        return "Set" + super().__repr__()

    def __len__(self) -> int:
        return len(self.obj)

    def __iter__(self):
        return iter(self.obj)

    def __contains__(self, elt: Ref) -> bool:
        return elt in self.obj


def recurse_on_ref_collections(f: Callable, obj: Any, **kwargs: Any) -> Any:
    if isinstance(obj, AtomRef):
        return f(obj, **kwargs)
    elif isinstance(obj, (list, ListRef)):
        return [recurse_on_ref_collections(f, elt, **kwargs) for elt in obj]
    elif isinstance(obj, (dict, DictRef)):
        return {k: recurse_on_ref_collections(f, v, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(recurse_on_ref_collections(f, elt, **kwargs) for elt in obj)
    elif isinstance(obj, set):
        return {recurse_on_ref_collections(f, elt, **kwargs) for elt in obj}
    elif isinstance(obj, RefCollection):
        return ValueCollection(values=[recurse_on_ref_collections(f, ref, **kwargs) for ref in obj.refs])
    else:
        return obj


def __make_list__(**items: Any) -> MList[Any]:
    # items must be a dict with keys "elts_0", "elts_1", etc.
    elts = [items[f"elts_{i}"] for i in range(len(items))]
    return ListRef(
        cid=get_content_hash([elt.cid for elt in elts]),
        hid=get_content_hash([elt.hid for elt in elts]),
        in_memory=True,
        obj=elts,
    )

def __make_dict__(**kwargs: Any) -> dict:
    return DictRef(
        cid=get_content_hash(sorted([(k, v.cid) for k, v in kwargs.items()])),
        hid=get_content_hash(sorted([(k, v.hid) for k, v in kwargs.items()])),
        in_memory=True,
        obj=kwargs,
    )



def __make_set__(**kwargs: Any) -> MSet[Any]:
    elts = [kwargs[f"elts_{i}"] for i in range(len(kwargs))]
    return SetRef(
        cid=get_content_hash(sorted([elt.cid for elt in elts])),
        hid=get_content_hash(sorted([elt.hid for elt in elts])),
        in_memory=True,
        obj=set(elts),
    )


def __make_tuple__(*elts: Any) -> tuple:
    return tuple(elts)


def __list_getitem__(list: MList[Any], i: Any) -> Any:
    return list[i.obj]

def __dict_getitem__(dict: MDict[Any, Any], key: Any) -> Any:
    return dict[key]


__make_list__ = Op(name=__make_list__.__name__, f=__make_list__, __structural__=True, output_names=["list"])
__make_dict__ = Op(name=__make_dict__.__name__, f=__make_dict__, __structural__=True, output_names=["dict"])
__make_set__ = Op(name=__make_set__.__name__, f=__make_set__, __structural__=True, output_names=["set"])
__make_tuple__ = Op(name=__make_tuple__.__name__, f=__make_tuple__, __structural__=True, output_names=["tuple"])
__list_getitem__ = Op(name=__list_getitem__.__name__, f=__list_getitem__, __structural__=True, output_names=["list_item"])
__dict_getitem__ = Op(name=__dict_getitem__.__name__, f=__dict_getitem__, __structural__=True, output_names=["dict_value"])


def make_ref_set(resf: Iterable[Ref]) -> SetRef:
    return __make_set__.f(**{f"elts_{i}": elt for i, elt in enumerate(resf)})


################################################################################
### interfaces
################################################################################
def op(
    output_names: Union[Optional[List[str]], Callable] = None,
    nout: Union[Literal["var", "auto"], int] = "auto",
    ignore_args: Optional[Tuple[str,...]] = None,
    __structural__: bool = False,
    __allow_side_effects__: bool = False,
):
    """
    Decorator used to make a function memoized by the storage. Some options:

    - `ignore_args` is a tuple of argument names (keyword or positional) that
    should be ignored when hashing the function. This is useful when the
    function has arguments that are not relevant to the output, like a batch
    size.
    """
    def decorator(f: Callable, output_names = None) -> 'f': # some IDE magic to make it recognize that @op(f) has the same type as f
        res = Op(
            f.__name__,
            f,
            output_names=output_names,
            nout=nout,
            ignore_args=ignore_args,
            __structural__=__structural__,
            __allow_side_effects__=__allow_side_effects__,
        )
        return functools.wraps(f)(res) # more magic 

    if callable(output_names):
        return decorator(output_names, None)
    else:
        return lambda f: decorator(f, output_names)


class Context:

    current_context: Optional["Context"] = None
    _profiling_stats: Dict[str, float] = {
        'total_time': 0.0,
        'get_call_time': 0.0,
        'call_exists_time': 0.0,
    }

    @staticmethod
    def reset_profiling_stats():
        Context._profiling_stats = {
            'total_time': 0.0,
            'get_call_time': 0.0,
            'call_exists_time': 0.0,
        }

    def __init__(self, storage: "Storage") -> None:
        self.storage = storage

    def __enter__(self) -> "Storage":
        Context.current_context = self
        return self.storage

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        Context.current_context = None


class RefCollection:
    def __init__(self, refs: Iterable[Ref]):
        self.refs = sorted(refs, key=lambda ref: ref.hid)
    
    def __repr__(self):
        return f"RefCollection({self.refs})"


class ValueCollection:
    def __init__(self, values: List[None]):
        self.values = values
    
    def __repr__(self):
        return f"ValueCollection({self.values})"


class CallCollection:
    def __init__(self, calls: Iterable[Call]):
        self.calls = sorted(calls, key=lambda call: call.hid)
    
    def __repr__(self):
        return f"CallCollection({self.calls})"

