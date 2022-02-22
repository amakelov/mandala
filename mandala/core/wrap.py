from .config import CoreConfig
from .bases import ValueRef, Call, is_instance, get_content_hash_with_type
from .impl import (
    AtomRef, ListRef, DictRef, ConstructDict, ConstructList, GetItemList,
    DeconstructList, GetKeyDict
)
from .tps import (
    Type, AtomType, ListType, AnyType, BuiltinTypes, DictType,
    UnionType, isinstance_annotation, is_subtype
)
from .utils import AnnotatedObj

from ..common_imports import *
from ..util.common_ut import concat_homogeneous_lists, is_admissible_atom

################################################################################
### type inference 
################################################################################
def is_concrete_type_match(obj:TAny, tp:Type) -> bool:
    """
    Check whether `tp` is a concrete type that matches `obj`. 
    """
    if not tp.is_concrete:
        return False
    elif isinstance(tp, AnyType):
        return True
    elif isinstance(tp, AtomType):
        if isinstance(obj, ValueRef):
            return type(obj.obj()) is tp.annotation
        else:
            return type(obj) is tp.annotation
    elif isinstance(tp, ListType):
        if not isinstance(obj, list):
            return False
        return all(is_concrete_type_match(obj=elt, tp=tp.elt_type)
                   for elt in obj)
    elif isinstance(tp, DictType):
        if not isinstance(obj, dict):
            return False
        return all(is_concrete_type_match(obj=v, tp=tp.value_type)
                   for v in obj.values())
    else:
        raise NotImplementedError()
    
def match_to_concrete_type(obj:TAny, options:TList[Type]) -> TOption[Type]:
    """
    Greedily find the first concrete type matching this object from a list of
    options, proceeding from left to right. Useful for matching against a union
    """
    for option in options:
        if isinstance(option, UnionType):
            # recursive call
            result = match_to_concrete_type(obj=obj, options=option.operands)
            if result is not None:
                return result
        else:
            if is_concrete_type_match(obj=obj, tp=option):
                return option
    return None
    
def infer_concrete_type(obj:TAny, reference:Type=None, 
                        type_dict:TDict[type, Type]=None) -> Type:
    if isinstance(obj, ValueRef):
        tp = obj.get_type()
        if reference is not None and (not is_subtype(s=tp, t=reference)):
            raise TypeError(f'Value is wrapped as type {tp}, expecting {reference}')
        return tp
    type_dict = {} if type_dict is None else type_dict
    obj_tp = type(obj)
    if reference is not None:
        if isinstance(reference, AnyType) and obj_tp in type_dict:
            return type_dict[obj_tp]
        elif isinstance(reference, UnionType):
            result = match_to_concrete_type(obj=obj, options=reference.operands)
            if result is None:
                raise TypeError(f'Matching against type {reference} failed for value {obj} of type {type(obj)}')
            else:
                return result
        else:
            return reference
    if obj_tp in type_dict:
        return type_dict[obj_tp]
    return AnyType()

################################################################################
### atoms
################################################################################
def parse_annotations(obj:TAny, mark_transient:bool=False, 
                      mark_delayed_storage:bool=False,
                      ) -> TTuple[TAny, TDict[str, TAny]]:
    if isinstance(obj, AnnotatedObj):
        annotations = {
            'transient': obj.transient,
            'delayed_storage': obj.delayed_storage
        }
        return obj.obj, annotations
    else:
        annotations =  {
            'transient': mark_transient,
            'delayed_storage': mark_delayed_storage
        }
        return obj, annotations

def wrap_as_atom(obj:TAny, tp:Type, set_uid:bool=True,
                 mark_transient:bool=False, mark_delayed_storage:bool=False,
                 require_admissible:bool=False, typecheck:bool=True) -> AtomRef:
    """
    Importantly, both the object's identity and the type participate in this to
    ensure that objects with the same value but with different types have
    distinct uids
    """
    obj, annotations = parse_annotations(
        obj=obj, mark_transient=mark_transient,
        mark_delayed_storage=mark_delayed_storage
    )
    if typecheck:
        assert not tp.is_compound
        if not (tp.annotation is typing.Any):
            try: 
                success = isinstance_annotation(obj=obj, annotation=tp.annotation)
            except NotImplementedError:
                success = True
                logging.warning(f'Encountered annotation {tp.annotation} that cannot be type-checked')
            if not success:
                raise TypeError(f'Got {type(obj)}, expecting {tp.annotation}')
    if require_admissible:
        if not is_admissible_atom(obj=obj):
            raise ValueError()
    if set_uid:
        uid = get_content_hash_with_type(raw_obj=obj, tp=tp)
    else:
        uid = None
    return AtomRef(obj=obj, in_memory=True, uid=uid, tp=tp, 
                  persistable=(not annotations['transient']), 
                  delayed_storage=annotations['delayed_storage'])

################################################################################
### constructive wrapping
################################################################################
def wrap_constructive(obj:TAny, reference:Type=None, 
                      type_dict:TDict[type, Type]=None,
                      ) -> TTuple[ValueRef, TList[Call]]:
    """
    Given an object, wrap it by applying constructive representations of any
    NEW structs that are formed by this process. 
    
    Returns:
        - the wrapped object 
        - all newly generated calls to structural operations by this process

    NOTE:
        - if `obj` is already a vref, its type will not be changed; instead,
        an error will be raised if the vref is not an instance of the inferred
        type (from `reference`, etc.)
    """
    obj, annotations = parse_annotations(obj=obj)
    mark_transient = annotations['transient']
    mark_delayed_storage = annotations['delayed_storage']
    tp = infer_concrete_type(obj=obj, reference=reference, type_dict=type_dict)
    if isinstance(obj, ValueRef):
        if not is_instance(vref=obj, tp=tp):
            raise TypeError(f'Got {obj.get_type()}, expecting {tp}')
        return obj, []
    if not tp.is_compound:
        res = wrap_as_atom(obj=obj, tp=tp, mark_transient=mark_transient, 
                           mark_delayed_storage=mark_delayed_storage)
        return res, []
    elif isinstance(tp, ListType):
        assert type(obj) is list, f'Got {type(obj)}'
        if mark_transient:
            raise TypeError('Making compound values transient is not allowed')
        if mark_delayed_storage:
            raise TypeError('Delaying storage of compound values is not allowed')
        elt_results = [wrap_constructive(obj=elt, reference=tp.elt_type,
                                         type_dict=type_dict) for elt in obj]
        wrapped_elts = [x[0] for x in elt_results]
        elt_calls = concat_homogeneous_lists(lists=[x[1] for x in elt_results])
        op = ConstructList()
        construct_outputs, call = op.compute_wrapped(
            inputs={str(i): elt for i, elt in enumerate(wrapped_elts)},
            output_types={op._list_name: tp}
        )
        vref = construct_outputs[op._list_name]
        return vref, [call] + elt_calls
    elif isinstance(tp, DictType):
        assert type(obj) is dict
        if mark_transient:
            raise TypeError('Making compound values transient is not allowed')
        if mark_delayed_storage:
            raise TypeError('Delaying storage of compound values is not allowed')
        elt_results = {k: wrap_constructive(obj=v, reference=tp.value_type,
                                            type_dict=type_dict) 
                       for k, v in obj.items()}
        wrapped_elts = {k: v[0] for k, v in elt_results.items()}
        elt_calls = concat_homogeneous_lists(
            lists=[x[1] for x in elt_results.values()]
        )
        op = ConstructDict()
        construct_outputs, call = op.compute_wrapped(
            inputs={k: v for k, v in wrapped_elts.items()},
            output_types={op._dict_name: tp}
        )
        vref = construct_outputs[op._dict_name]
        return vref, [call] + elt_calls
    else:
        raise NotImplementedError(f'Got {tp}')
    
def wrap_detached(obj:TAny, reference:Type=None, annotation:TAny=None, 
         type_dict:TDict[type, Type]=None, through_collections:bool=False,
         return_calls:bool=False,
         ) -> TUnion[ValueRef, TTuple[ValueRef, TList[Call]]]:
    """
    A wrapping function without side effects to storage. Performs constructive
    wrapping of values and optionally returns calls (but does not save them). 
    """
    type_dict = {} if type_dict is None else type_dict
    if through_collections:
        type_dict[list] = ListType()
        type_dict[dict] = DictType()
    if annotation is not None:
        assert reference is None
        reference = Type.from_annotation(annotation=annotation)
    vref, calls = wrap_constructive(obj=obj, reference=reference,
                                    type_dict=type_dict)
    if return_calls:
        return vref, calls
    else:
        return vref

################################################################################
### deconstructive wrapping
################################################################################
def wrap_structure(obj:TAny, reference:Type=None,
           type_dict:TDict[type, Type]=None) -> ValueRef:
    """
    Given a value, wrap it as a value ref with blank UIDs. In
    particular, this assigns other properties of vrefs, such as:
        - type
        - structural type (list, dict, atom)
        - persistability

    NOTE:
        - by definition, compound values cannot be made transient. 
        - the same notes apply here as in `wrap_constructive`.
    """
    obj, annotations = parse_annotations(obj=obj)
    mark_transient = annotations['transient']
    mark_delayed_storage = annotations['delayed_storage']
    reference = AnyType() if reference is None else reference
    type_dict = {} if type_dict is None else type_dict
    if isinstance(obj, ValueRef):
        if not is_instance(vref=obj, tp=reference):
            raise TypeError(f'Got {obj.get_type()}, expecting {reference}')
        return obj
    tp = infer_concrete_type(obj=obj, reference=reference, type_dict=type_dict)
    if not tp.is_compound:
        return wrap_as_atom(obj=obj, tp=tp, set_uid=False,
                            mark_transient=mark_transient,
                            mark_delayed_storage=mark_delayed_storage)
    elif isinstance(tp, ListType):
        if mark_transient:
            raise TypeError('Making compound values transient is not allowed')
        if mark_delayed_storage:
            raise TypeError('Delaying storage of compound values is not allowed')
        return ListRef(obj=[wrap_structure(obj=elt,
                                           reference=tp.elt_type,
                                           type_dict=type_dict) for elt in obj], 
                       uid=None, tp=tp)
    elif isinstance(tp, DictType):
        if mark_transient:
            raise TypeError('Making compound values transient is not allowed')
        if mark_delayed_storage:
            raise TypeError('Delaying storage of compound values is not allowed')
        return DictRef(obj={k: wrap_structure(obj=v, reference=tp.value_type, 
                                              type_dict=type_dict)
                            for k, v in obj.items()}, uid=None, tp=tp)
    else:
        raise NotImplementedError()
    
def get_list_decomposition(list_ref:ListRef, write_uids:bool=False, 
                           as_many:bool=True) -> TList[Call]:
    """
    Produce the calls to `GetItemList` decomposing this list ref, and optionally
    induce UIDs for the elements of the list that do not have UIDs.
    """
    if as_many:
        assert list_ref.uid is not None
        ops = [GetItemList() for _ in list_ref]
        calls = []
        for i, op in enumerate(ops):
            inputs = {
                op._list_name: list_ref,
                op._idx_name: wrap_as_atom(obj=i, tp=BuiltinTypes.get(int))
            }
            _, call = op.compute_wrapped(inputs=inputs, _write_uids=write_uids)
            calls.append(call)
        return calls
    else:
        assert list_ref.uid is not None
        op = DeconstructList()
        inputs = {op._list_name: list_ref}
        _, call = op.compute_wrapped(inputs=inputs, _write_uids=write_uids)
        return [call]

def get_dict_decomposition(dict_ref:DictRef,
                           write_uids:bool=False) -> TList[Call]:
    """
    Produce the calls to `GetItemList` decomposing this list ref, and optionally
    induce UIDs for the elements of the list that do not have UIDs.
    """
    assert dict_ref.uid is not None
    ops = {k: GetKeyDict() for k in dict_ref.keys()}
    calls = []
    for k, op in ops.items():
        inputs = {
            op._dict_name: dict_ref,
            op._key_name: wrap_as_atom(obj=k, tp=BuiltinTypes.get(str))
        }
        _, call = op.compute_wrapped(inputs=inputs, _write_uids=write_uids)
        calls.append(call)
    return calls

def get_deconstruction_calls(obj:ValueRef, res:TList[Call]=None, 
                             write_uids:bool=False) -> TList[Call]:
    """
    Given a (fully-typed) vref, return all calls in its *de-constructive*
    representation, and generate UIDs for the constituents that are missing
    them. 

    NOTE:
        - see notes for `wrap_constructive`
    """
    if res is None:
        res = []
    if isinstance(obj, AtomRef):
        return res
    elif isinstance(obj, ListRef):
        res += get_list_decomposition(
            list_ref=obj,
            write_uids=write_uids,
            as_many=CoreConfig.decompose_struct_as_many
        )
        for elt in obj:
            get_deconstruction_calls(obj=elt, res=res, write_uids=write_uids)
    elif isinstance(obj, DictRef):
        res += get_dict_decomposition(dict_ref=obj, write_uids=write_uids)
        for elt in obj.values():
            get_deconstruction_calls(obj=elt, res=res, write_uids=write_uids)
    else:
        raise NotImplementedError()
    return res