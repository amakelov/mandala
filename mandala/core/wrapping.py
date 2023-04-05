import typing
from ..common_imports import *
from .tps import Type, ListType, DictType, SetType, AnyType, StructType
from .utils import Hashing
from .model import Ref, Call, wrap_atom, ValueRef
from .builtins_ import ListRef, DictRef, SetRef, Builtins


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


def wrap_causal(obj: Any, annotation: Any, start_uid: str) -> Tuple[Ref, List[Call]]:
    tp = Type.from_annotation(annotation=annotation)
    typecheck(obj=obj, tp=tp)
    calls = []
    if isinstance(obj, Ref):
        return obj, calls
    if isinstance(tp, AnyType):
        return wrap_atom(obj=obj, uid=start_uid), calls
    if isinstance(tp, ListType):
        elt_hashes = [Hashing.hash_list([start_uid, i]) for i in range(len(obj))]
        elt_refs = []
        for i, elt in enumerate(obj):
            elt, elt_calls = wrap_causal(
                obj=elt,
                annotation=tp.elt_type,
                start_uid=elt_hashes[i],
            )
            elt_refs.append(elt)
            calls.extend(elt_calls)
        res = ListRef(uid=start_uid, obj=elt_refs, in_memory=True)
        calls.extend(res.get_calls())
        return res, calls
    elif isinstance(tp, DictType):
        elt_hashes = {k: Hashing.hash_list([start_uid, k]) for k in obj.keys()}
        elt_refs = {}
        for k, elt in obj.items():
            elt, elt_calls = wrap_causal(
                obj=elt,
                annotation=tp.elt_type,
                start_uid=elt_hashes[k],
            )
            elt_refs[k] = elt
            calls.extend(elt_calls)
    elif isinstance(tp, SetType):
        elt_hashes = [Hashing.hash_list([start_uid, i]) for i in range(len(obj))]
        elt_refs = []
        for i, elt in enumerate(obj):
            elt, elt_calls = wrap_causal(
                obj=elt,
                annotation=tp.elt_type,
                start_uid=elt_hashes[i],
            )
            elt_refs.append(elt)
            calls.extend(elt_calls)
        res = SetRef(uid=start_uid, obj=elt_refs, in_memory=True)
        calls.extend(res.get_calls())
        return res, calls


def wrap_constructive(obj: Any, annotation: Any) -> Tuple[Ref, List[Call]]:
    tp = Type.from_annotation(annotation=annotation)
    typecheck(obj=obj, tp=tp)
    calls = []
    if isinstance(obj, Ref):
        return obj, calls
    if isinstance(tp, AnyType):
        return wrap_atom(obj=obj), calls
    if isinstance(tp, StructType):
        RefCls: Union[
            typing.Type[ListRef], typing.Type[DictRef], typing.Type[SetRef]
        ] = Builtins.REF_CLASSES[tp.struct_id]
        assert isinstance(obj, tp.model)
        recursive_result = RefCls.map(
            obj=obj, func=lambda elt: wrap_constructive(elt, annotation=tp.elt_type)
        )
        wrapped_elts = RefCls.map(obj=recursive_result, func=lambda elt: elt[0])
        recursive_calls = RefCls.elts(
            RefCls.map(obj=recursive_result, func=lambda elt: elt[1])
        )
        calls.extend([c for cs in recursive_calls for c in cs])
        obj = RefCls(obj=wrapped_elts, uid=None, in_memory=True)
        obj_calls = obj.get_calls()
        calls.extend(RefCls.elts(obj_calls))
        return obj, calls
    else:
        raise ValueError(f"Cannot wrap {type(obj)}")


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
        # return unwrap(obj.obj, through_collections=through_collections)
        return obj.obj
    elif isinstance(obj, ListRef):
        return [unwrap(elt, through_collections=through_collections) for elt in obj.obj]
        # return unwrap(obj.obj, through_collections=through_collections)
    elif isinstance(obj, DictRef):
        return {
            k: unwrap(v, through_collections=through_collections)
            for k, v in obj.obj.items()
        }
    elif isinstance(obj, SetRef):
        return {unwrap(v, through_collections=through_collections) for v in obj.obj}
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
    elif isinstance(ref, ListRef):
        return any(contains_transient(elt) for elt in ref.obj)
    elif isinstance(ref, DictRef):
        return any(contains_transient(v) for v in ref.obj.values())
    elif isinstance(ref, SetRef):
        return any(contains_transient(v) for v in ref.obj)
    else:
        raise ValueError(f"Unexpected ref type {type(ref)}")


def contains_not_in_memory(ref: Ref) -> bool:
    if isinstance(ref, ValueRef):
        return not ref.in_memory
    elif isinstance(ref, ListRef):
        return any(contains_not_in_memory(elt) for elt in ref.obj)
    elif isinstance(ref, DictRef):
        return any(contains_not_in_memory(v) for v in ref.obj.values())
    elif isinstance(ref, SetRef):
        return any(contains_not_in_memory(v) for v in ref.obj)
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
