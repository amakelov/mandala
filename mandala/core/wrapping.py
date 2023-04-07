import typing
from ..common_imports import *
from .tps import Type, ListType, DictType, SetType, AnyType, StructType
from .utils import Hashing
from .model import Ref, Call, wrap_atom, ValueRef
from .builtins_ import ListRef, DictRef, SetRef, Builtins, StructRef


def typecheck(obj: Any, tp: Type):
    if isinstance(tp, ListType) and not (
        isinstance(obj, list) or isinstance(obj, ListRef)
    ):
        raise ValueError(f"Expecting a list, got {type(obj)}")
    elif isinstance(tp, DictType) and not (
        isinstance(obj, dict) or isinstance(obj, DictRef)
    ):
        raise ValueError(f"Expecting a dict, got {type(obj)}")
    elif isinstance(tp, SetType) and not (
        isinstance(obj, set) or isinstance(obj, SetRef)
    ):
        raise ValueError(f"Expecting a set, got {type(obj)}")


def causify_atom(ref: ValueRef):
    if ref.causal_uid is not None:
        return
    ref.causal_uid = Hashing.get_content_hash(ref.uid)


def causify_down(ref: Ref, start: str, stop_at_causal: bool = True):
    """
    In-place top-down assignment of causal hashes to a Ref and its children.
    Requires causal hashes to not be present initially.
    """
    assert start is not None
    if ref.causal_uid is not None and stop_at_causal:
        return
    if isinstance(ref, ValueRef):
        ref.causal_uid = start
    elif isinstance(ref, ListRef):
        if ref.in_memory:
            for i, elt in enumerate(ref.obj):
                causify_down(elt, start=Hashing.hash_list([start, i]))
        ref.causal_uid = start
    elif isinstance(ref, DictRef):
        if ref.in_memory:
            for k, elt in ref.obj.items():
                causify_down(elt, start=Hashing.hash_list([start, k]))
        ref.causal_uid = start
    elif isinstance(ref, SetRef):
        # sort by uid to ensure deterministic ordering
        if ref.in_memory:
            elts_by_uid = {elt.uid: elt for elt in ref.obj}
            sorted_uids = sorted({elt.uid for elt in ref.obj})
            for i, uid in enumerate(sorted_uids):
                causify_down(elts_by_uid[uid], start=Hashing.hash_list([start, uid]))
        ref.causal_uid = start
    else:
        raise ValueError(f"Unknown ref type {type(ref)}")


def decausify(ref: Ref, stop_at_first_missing: bool = False):
    """
    In-place recursive removal of causal hashes from a Ref
    """
    if ref._causal_uid is None and stop_at_first_missing:
        return
    ref._causal_uid = None
    if isinstance(ref, StructRef):
        for elt in ref.children():
            decausify(elt, stop_at_first_missing=stop_at_first_missing)


def wrap_constructive(obj: Any, annotation: Any) -> Tuple[Ref, List[Call]]:
    tp = Type.from_annotation(annotation=annotation)
    typecheck(obj=obj, tp=tp)
    calls = []
    if isinstance(obj, Ref):
        res = obj, calls
    elif isinstance(tp, AnyType):
        res = wrap_atom(obj=obj), calls
        causify_atom(ref=res[0])
    elif isinstance(tp, StructType):
        RefCls: Union[
            typing.Type[ListRef], typing.Type[DictRef], typing.Type[SetRef]
        ] = Builtins.REF_CLASSES[tp.struct_id]
        assert type(obj) == tp.model
        recursive_result = RefCls.map(
            obj=obj, func=lambda elt: wrap_constructive(elt, annotation=tp.elt_type)
        )
        wrapped_elts = RefCls.map(obj=recursive_result, func=lambda elt: elt[0])
        recursive_calls = RefCls.elts(
            RefCls.map(obj=recursive_result, func=lambda elt: elt[1])
        )
        calls.extend([c for cs in recursive_calls for c in cs])
        obj: StructRef = RefCls(obj=wrapped_elts, uid=None, in_memory=True)
        obj.causify_up()
        obj_calls = obj.get_calls()
        calls.extend(obj_calls)
        res = obj, calls
    else:
        raise ValueError(f"Unknown type {tp}")
    return res


def wrap_inputs(
    objs: Dict[str, Any],
    annotations: Dict[str, Any],
) -> Tuple[Dict[str, Ref], List[Call]]:
    calls = []
    wrapped_objs = {}
    for k, v in objs.items():
        wrapped_obj, wrapping_calls = wrap_constructive(
            obj=v,
            annotation=annotations[k],
        )
        wrapped_objs[k] = wrapped_obj
        calls.extend(wrapping_calls)
    return wrapped_objs, calls


def causify_outputs(refs: List[Ref], call_causal_uid: str):
    assert isinstance(call_causal_uid, str)
    for i, ref in enumerate(refs):
        causify_down(ref=ref, start=Hashing.hash_list([call_causal_uid, str(i)]))


def wrap_outputs(
    objs: List[Any],
    annotations: List[Any],
) -> Tuple[List[Ref], List[Call]]:
    calls = []
    wrapped_objs = []
    for i, v in enumerate(objs):
        wrapped_obj, wrapping_calls = wrap_constructive(
            obj=v,
            annotation=annotations[i],
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
    if isinstance(obj, ValueRef) and obj.transient:
        return obj.obj
    if isinstance(obj, Ref) and not obj.in_memory:
        from ..ui.contexts import GlobalContext

        if GlobalContext.current is None:
            raise ValueError(
                "Cannot unwrap a Ref with `in_memory=False` outside a context"
            )
        storage = GlobalContext.current.storage
        storage.rel_adapter.mattach(vrefs=[obj])
    if isinstance(obj, ValueRef):
        return obj.obj
    elif isinstance(obj, StructRef):
        return type(obj).map(
            obj=obj.obj,
            func=lambda elt: unwrap(elt, through_collections=through_collections),
        )
    elif type(obj) is tuple and through_collections:
        return tuple(unwrap(v, through_collections=through_collections) for v in obj)
    elif type(obj) is set and through_collections:
        return {unwrap(v, through_collections=through_collections) for v in obj}
    elif type(obj) is list and through_collections:
        return [unwrap(v, through_collections=through_collections) for v in obj]
    elif type(obj) is dict and through_collections:
        return {
            k: unwrap(v, through_collections=through_collections)
            for k, v in obj.items()
        }
    else:
        return obj


def contains_transient(ref: Ref) -> bool:
    if isinstance(ref, ValueRef):
        return ref.transient
    elif isinstance(ref, StructRef):
        return any(contains_transient(elt) for elt in ref.children())
    else:
        raise ValueError(f"Unexpected ref type {type(ref)}")


def contains_not_in_memory(ref: Ref) -> bool:
    if isinstance(ref, ValueRef):
        return not ref.in_memory
    elif isinstance(ref, StructRef):
        return any(contains_not_in_memory(elt) for elt in ref.children())
    else:
        raise ValueError(f"Unexpected ref type {type(ref)}")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Ref):
        return (_sanitize_value(value.obj), value.in_memory, value.uid)
    try:
        hash(value)
        return value
    except TypeError:
        if isinstance(value, bytearray):
            return value.hex()
        elif isinstance(value, list):
            return tuple([_sanitize_value(v) for v in value])
        else:
            raise NotImplementedError(f"Got value of type {type(value)}")


def compare_dfs_as_relations(
    df_1: pd.DataFrame, df_2: pd.DataFrame, return_reason: bool = False
) -> Union[bool, Tuple[bool, str]]:
    if df_1.shape != df_2.shape:
        result, reason = False, f"Shapes differ: {df_1.shape} vs {df_2.shape}"
    if set(df_1.columns) != set(df_2.columns):
        result, reason = False, f"Columns differ: {df_1.columns} vs {df_2.columns}"
    # reorder columns of df_2 to match df_1
    df_2 = df_2[df_1.columns]
    # sanitize values to make them hashable
    df_1 = df_1.applymap(_sanitize_value)
    df_2 = df_2.applymap(_sanitize_value)
    # compare as sets of tuples
    result = set(map(tuple, df_1.itertuples(index=False))) == set(
        map(tuple, df_2.itertuples(index=False))
    )
    if result:
        reason = ""
    else:
        reason = f"Dataframe rows differ: {df_1} vs {df_2}"
    if return_reason:
        return result, reason
    else:
        return result
