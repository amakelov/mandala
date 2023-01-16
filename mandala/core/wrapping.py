from ..common_imports import *
from .tps import Type, ListType, DictType, SetType, AnyType
from .model import Ref, FuncOp, Call, wrap, ValueRef
from .builtins_ import ListRef, DictRef, SetRef, Builtins


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
    elif isinstance(tp, DictType):
        assert isinstance(obj, dict)
        elt_results = {
            k: wrap_constructive(v, annotation=tp.val_type) for k, v in obj.items()
        }
        elt_calls = [c for _, cs in elt_results.values() for c in cs]
        elt_vrefs = {k: v for k, (v, _) in elt_results.items()}
        calls.extend(elt_calls)
        result, construction_calls = Builtins.construct_dict(elts=elt_vrefs)
        calls.extend(construction_calls)
        return result, calls
    elif isinstance(tp, SetType):
        assert isinstance(obj, set)
        elt_results = [wrap_constructive(elt, annotation=tp.elt_type) for elt in obj]
        elt_calls = [c for _, cs in elt_results for c in cs]
        elt_vrefs = [v for v, _ in elt_results]
        calls.extend(elt_calls)
        result, construction_calls = Builtins.construct_set(elts=elt_vrefs)
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
    if isinstance(obj, Ref) and not obj.in_memory:
        from ..ui.main import GlobalContext

        if GlobalContext.current is None:
            raise ValueError(
                "Cannot unwrap a Ref with `in_memory=False` outside a context"
            )
        storage = GlobalContext.current.storage
        storage.rel_adapter.mattach(vrefs=[obj])
    if isinstance(obj, ValueRef):
        return unwrap(obj.obj, through_collections=through_collections)
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
