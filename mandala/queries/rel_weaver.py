from abc import abstractmethod

from .weaver_bases import OpWeave, ValWeave

from ..common_imports import *
from ..core.utils import BackwardCompatible
from ..core.config import CoreConfig, CoreConsts, MODES
from ..core.tps import (
    Type, AtomType, ListType, DictType, AnyType,
    UnionType, get_union, BuiltinTypes, intersection
)
from ..core.bases import Operation, ValueRef, GlobalContext
from ..core.impl import (
    SimpleFunc, GetItemList, GetKeyDict, ConstructDict,
    ConstructList, DeconstructList
)

SPAWN_CONSTRAINTS = False
ListConsts = CoreConsts.List
DictConsts = CoreConsts.Dict

TIdxConstraint = TOption[TUnion[int, TList[int]]]

def attach_to_context(query_obj:TUnion['ValQuery', 'OpQuery']):
    if GlobalContext.exists():
        context = GlobalContext.get()
        context.attach_query(query_obj=query_obj)

class ConstraintIds(object):
    isin = 'isin'
    where = 'where'
    equals = 'equals'
    identical = 'identical'


################################################################################
### value queries
################################################################################
class ValQuery(ValWeave):

    def __init__(self, creator:'OpQuery'=None, created_as:str=None, 
                 consumers:TList['OpQuery']=None, consumed_as:TList[str]=None,
                 constraints:TList[TAny]=None, 
                 value:TAny=None, is_singleton:bool=False, 
                 tp:Type=None, display_name:str=None):
        self._creator = creator
        self._created_as = created_as
        self._consumers = [] if consumers is None else consumers
        self._consumed_as = [] if consumed_as is None else consumed_as

        self._constraints = [] if constraints is None else constraints
        self._value = value
        self._is_singleton = is_singleton
        self._tp = AnyType() if tp is None else tp
        self._display_name = display_name
        self._aliases = []
        attach_to_context(self)
    
    ############################################################################ 
    ### weaver interface implementation
    ############################################################################ 
    @property
    def creator(self) -> TOption['OpQuery']:
        return self._creator
    
    @property
    def created_as(self) -> TOption[str]:
        return self._created_as
    
    @property
    def consumers(self) -> TList['OpQuery']:
        return self._consumers
    
    @property
    def consumed_as(self) -> TList[str]:
        return self._consumed_as
    
    def set_creator(self, creator:'OpQuery', created_as:str):
        assert self.creator is None
        assert self.created_as is None
        self._creator = creator
        self._created_as = created_as
    
    def add_consumer(self, consumer:'OpQuery', consumed_as: str):
        self._consumers.append(consumer)
        self._consumed_as.append(consumed_as)
    
    ############################################################################ 
    @property
    def tp(self) -> Type:
        return self._tp

    @property
    def constraints(self) -> TList[TAny]:
        return self._constraints

    ############################################################################ 
    ### fluent interface
    ############################################################################ 
    def spawn(self, return_copy:bool=SPAWN_CONSTRAINTS) -> 'ValQuery':
        """
        The base on top of which new constraints are added. Can be a copy or 
        `self`.
        """
        if return_copy:
            # don't pass collections by reference 
            constraints = [_ for _ in self.constraints]
            consumers = [_ for _ in self.consumers]
            consumed_as = [_ for _ in self.consumed_as]
            ConstructorCls = type(self)
            return ConstructorCls(
                creator=self.creator, created_as=self.created_as,
                constraints=constraints, consumers=consumers,
                consumed_as=consumed_as,
                value=self._value, is_singleton=self._is_singleton,
                tp=self.tp, display_name=self.display_name
            )
        else:
            return self
        
    def isin(self, rng:TList[TAny]) -> 'ValQuery':
        res = self.spawn()
        res._constraints.append((ConstraintIds.isin, rng))
        return res
    
    def where(self, pred:TCallable[[TAny], bool]) -> 'ValQuery':
        res = self.spawn()
        res._constraints.append((ConstraintIds.where, pred))
        return res
    
    def equals(self, value:TAny) -> 'ValQuery':
        res = self.spawn()
        res._constraints.append((ConstraintIds.equals, value))
        return res
    
    def identical(self, values:TList[ValueRef]):
        res = self.spawn()
        res._constraints.append((ConstraintIds.identical, values))
        return res
    
    def set_display_name(self, display_name:str):
        if self._display_name is not None:
            raise ValueError(f'Setting the name of an already named value query is not allowed (current name = "{self._display_name}"')
        self._display_name = display_name
    
    def named(self, name:str) -> 'ValQuery':
        self.set_display_name(display_name=name)
        return self

    ############################################################################ 
    @property
    def sql_name(self) -> str:
        return f'_{id(self)}'
    
    @property
    def display_name(self) -> str:
        return self._display_name

    def __repr__(self) -> str:
        data = {}
        data['display_name'] = self.display_name
        constraint_reprs = self.constraints
        data['constraints'] = constraint_reprs
        if self.creator is not None:
            data['creator'] = self.creator.op.ui_name
        else:
            data['creator'] = None
        order = ('display_name', 'creator', 'constraints')
        args = ', '.join([f'{k}={data[k]}' for k in order])
        return f'ValQuery({args})'
    
    ############################################################################ 
    ### constructors
    ############################################################################ 
    @staticmethod
    def from_tp(tp:Type) -> 'ValQuery':
        if isinstance(tp, UnionType):
            operands = tp.operands
            ### perform automatic reduction for homogeneous operands
            if all(isinstance(op, ListType) for op in operands):
                new_tp = ListType(elt_type=get_union(
                    tps=[op.elt_type for op in operands])
                )
                return ListQuery(tp=new_tp)
            elif all(isinstance(op, DictType) for op in operands):
                new_tp = DictType(value_type=get_union(
                        tps=[op.value_type for op in operands])
                )
                return DictQuery(tp=new_tp)
            else:
                return ValQuery(tp=tp)
        if not tp.is_compound:
            return AtomQuery(tp=tp)
        elif isinstance(tp, ListType):
            return ListQuery(tp=tp)
        elif isinstance(tp, DictType):
            return DictQuery(tp=tp)
        else:
            return ValQuery(tp=tp)
        
    @staticmethod
    def from_annotation(annotation:TAny, display_name:str=None) -> 'ValQuery':
        tp = Type.from_annotation(annotation=annotation)
        res = ValQuery.from_tp(tp=tp)
        if display_name is not None:
            res.set_display_name(display_name=display_name)
        return res

    ############################################################################ 
    ### magics
    ############################################################################ 
    def _init_magic(self):
        assert CoreConfig.enable_query_magics

    def __lt__(self, other:TAny) -> 'ValQuery':
        self._init_magic()
        self.where(pred=lambda x: x < float(other))
        return self
    
    def __le__(self, other:TAny) -> 'ValQuery':
        self._init_magic()
        self.where(pred=lambda x: x <= float(other))
        return self
    
    #! overloading __eq__ can be problematic with hashing

    def __ne__(self, other:TAny) -> 'ValQuery':
        self._init_magic()
        self.where(pred=lambda x: x != other)
        return self
    
    def __gt__(self, other:TAny) -> 'ValQuery':
        self._init_magic()
        self.where(pred=lambda x: x > other)
        return self
    
    def __ge__(self, other:TAny) -> 'ValQuery':
        self._init_magic()
        self.where(pred=lambda x: x >= other)
        return self
    
    
class AtomQuery(ValQuery):
    
    @property
    def tp(self) -> AtomType:
        return self._tp
    

def get_tags_from_context() -> TDict[str, TAny]:
    if GlobalContext.exists():
        context = GlobalContext.get()
        if context.mode == MODES.query_delete:
            res = {'delete': True}
        else:
            res = {}
    else:
        res = {}
    return res

class ListQuery(ValQuery):

    @property
    def tp(self) -> ListType:
        return self._tp

    @staticmethod
    def parse_list_idx(obj:TUnion[ValQuery, int, TList[int]]=None) -> ValQuery:
        #! todo: slices
        builtin_int = BuiltinTypes.get(py_type=int)
        if obj is None:
            res = ValQuery.from_tp(tp=builtin_int)
        elif isinstance(obj, ValQuery):
            res = obj
        elif isinstance(obj, int):
            res = ValQuery.from_tp(tp=builtin_int).equals(obj)
        elif isinstance(obj, list):
            assert all(isinstance(x, int) for x in obj)
            # TODO: this step can be optimized by hashing 
            res = ValQuery.from_tp(tp=builtin_int).isin(rng=obj)
        else:
            raise NotImplementedError()
        return res
    
    def __getitem__(self, obj:TUnion[ValQuery, int, TList[int], None]) -> ValQuery:
        if CoreConfig.decompose_struct_as_many:
            idx_query = self.parse_list_idx(obj=obj)
            assert idx_query.tp == BuiltinTypes.get(py_type=int)
            get_item_query = GetItemQuery(tags=get_tags_from_context())
            res = get_item_query(lst=self, idx=idx_query)
            return res
        else:
            if isinstance(obj, ValQuery):
                raise ValueError()
            idx_constraint = obj
            deconstruct_list_query = DeconstructListQuery(
                tags=get_tags_from_context()
            )
            return deconstruct_list_query(lst=self, 
                                          idx_constraint=idx_constraint)

def _MakeList(containing:ValQuery, at_index:int=None,
             at_indices:TList[int]=None) -> ListQuery:
    construct_list_query = ConstructListQuery(tags=get_tags_from_context())
    if at_index is not None:
        assert at_indices is None
        idx_constraint = at_index
    else:
        if at_indices is not None:
            idx_constraint = at_indices
        else:
            idx_constraint = None
    return construct_list_query(prototype=containing,
                                idx_constraint=idx_constraint)
    
def MakeList(containing:TAny, at_index:int=None, at_indices:TList[int]=None) -> TAny:
    return _MakeList(containing=containing, at_index=at_index, at_indices=at_indices)

class DictQuery(ValQuery):

    @property
    def tp(self) -> DictType:
        return self._tp

    def parse_key_query(self, 
                        obj:TUnion[ValQuery, str, TList[str]]) -> ValQuery:
        builtin_str = BuiltinTypes.get(py_type=str)
        if not isinstance(obj, ValQuery):
            if isinstance(obj, str):
                obj = ValQuery.from_tp(tp=builtin_str).equals(obj)
            elif isinstance(obj, list):
                assert all(isinstance(x, str) for x in obj)
                obj = ValQuery.from_tp(tp=builtin_str).isin(rng=obj)
        return obj
    
    def __getitem__(self, obj:TUnion[ValQuery, str, TList[str]]) -> ValQuery:
        key_query = self.parse_key_query(obj=obj)
        assert key_query.tp == BuiltinTypes.get(py_type=str)
        get_key_query = GetKeyQuery(tags=get_tags_from_context())
        res = get_key_query(dct=self, key=key_query)
        return res
    
def MakeDict(containing:ValQuery, at_key:str=None, 
             at_keys:TList[str]=None) -> DictQuery:
    construct_dict_query = ConstructDictQuery(tags=get_tags_from_context())
    if at_key is not None:
        assert at_keys is None
        key_constraint = at_key
    else:
        if at_keys is not None:
            key_constraint = at_keys
        else:
            key_constraint = None
    return construct_dict_query(prototype=containing,
                                key_constraint=key_constraint)

################################################################################
### type inference
################################################################################
def wrap_as_vq(obj:TAny, reference:Type=None) -> ValQuery:
    if isinstance(obj, ValQuery):
        return obj
    if reference is None:
        return ValQuery(tp=AnyType()).equals(value=obj)
    return ValQuery(tp=reference).equals(value=obj)


################################################################################
### operation queries
################################################################################
class OpQuery(OpWeave):

    def __init__(self, 
                 inputs:TDict[str, ValQuery],
                 outputs:TDict[str, ValQuery]):
        self._inputs = inputs
        self._outputs = outputs
        attach_to_context(self)
    
    ############################################################################ 
    ### query-specific implementations and interfaces
    ############################################################################ 
    @property
    def tags(self) -> TDict[str, TAny]:
        raise NotImplementedError()

    @property
    def sql_name(self) -> str:
        return f'_{id(self)}'

    @property
    @abstractmethod
    def output_names(self) -> TList[str]:
        raise NotImplementedError()
    
    @property
    def inputs(self) -> TDict[str, ValQuery]:
        return self._inputs
    
    @property
    def outputs(self) -> TDict[str, ValQuery]:
        return self._outputs
    
    def weave_input(self, name: str, inp:ValQuery):
        assert name not in self._inputs
        inp.add_consumer(consumer=self, consumed_as=name)
        # inp.consumers.append(self)
        # inp.consumed_as.append(name)
        self._inputs[name] = inp
    
    def weave_output(self, name: str, outp:ValQuery):
        outp.set_creator(creator=self, created_as=name)
        self._outputs[name] = outp
    
    @abstractmethod
    def __call__(self, **inputs: TDict[str, TAny]) -> TDict[str, 'ValQuery']:
        return super().__call__(**inputs)


class FuncQuery(OpQuery):
    
    def __init__(self, op:SimpleFunc,
                 inputs:TDict[str, ValQuery]=None,
                 outputs:TDict[str, ValQuery]=None, 
                 tags:TDict[str, TAny]=None):
        self._op = op
        self._output_names = self._op.sig.output_names
        self._inputs = {} if inputs is None else inputs
        self._outputs = {} if outputs is None else outputs
        self._tags = {} if tags is None else tags
        attach_to_context(self)
    
    @property
    def tags(self) -> TDict[str, TAny]:
        return self._tags
    
    @property
    def op(self) -> SimpleFunc:
        return self._op
    
    @property
    def output_names(self) -> TList[str]:
        return self._output_names
    
    def __call__(self, *args, **kwargs) -> TDict[str, 'ValQuery']:
        signature = self.op.sig
        input_types = signature.bind_types(args=args, kwargs=kwargs)
        inputs = signature.bind_args(
            args=args,
            kwargs=kwargs,
            apply_defaults=CoreConfig.bind_defaults_in_queries
        )
        # remove BackwardCompatible constructs
        inputs = {k: v for k, v in inputs.items()
                  if not isinstance(v, BackwardCompatible)}
        output_types = signature.outputs
        wrapped_inputs = {name: wrap_as_vq(obj=obj,
                                           reference=input_types[name])
                          for name, obj in inputs.items()}
        # typecheck
        for k, v in wrapped_inputs.items():
            input_tp = v.tp
            signature_tp = input_types[k]
            if intersection(u=input_tp, v=signature_tp) is None:
                raise TypeError('Incompatible query')
        for k, v in wrapped_inputs.items():
            self.weave_input(name=k, inp=v)
        wrapped_outputs = {name: ValQuery.from_tp(tp=output_types[name])
                           for name in self.op.output_names}
        for k, v in wrapped_outputs.items():
            self.weave_output(name=k, outp=v)
        return wrapped_outputs

    def __repr__(self) -> str:
        return f'FuncQuery(op={self.op.ui_name}, inputs={self.inputs}, outputs={self.outputs})'
    

class GetItemQuery(FuncQuery):
    
    def __init__(self, lst:ListQuery=None, idx:ValQuery=None,
                 elt:ValQuery=None, tags:TDict[str, TAny]=None):
        self._inputs = {}
        self._outputs = {}
        self._tags = {} if tags is None else tags
        self._op = GetItemList()
        if lst is not None:
            self._inputs[ListConsts.LIST] = lst
        if idx is not None:
            self._inputs[ListConsts.IDX] = idx
        if elt is not None:
            self._outputs[ListConsts.ELT] = elt
        attach_to_context(self)
        
    @property
    def output_names(self) -> TList[str]:
        return [ListConsts.ELT]
    
    def __call__(self, lst:ListQuery, idx:ValQuery) -> ValQuery:
        self.weave_input(name=ListConsts.LIST, inp=lst)
        self.weave_input(name=ListConsts.IDX, inp=idx)
        outp = ValQuery.from_tp(tp=lst.tp.elt_type)
        self.weave_output(name=ListConsts.ELT, outp=outp)
        return outp


class ConstructListQuery(FuncQuery):
    def __init__(self, tags:TDict[str, TAny]=None):
        self._inputs = {}
        self._outputs = {}
        self._tags = {} if tags is None else tags
        self._op = ConstructList()
        self._idx_constraint:TIdxConstraint = None
        attach_to_context(self)
    
    @property
    def idx_constraint(self) -> TIdxConstraint:
        return self._idx_constraint
     
    @property
    def output_names(self) -> TList[str]:
        return [ListConsts.LIST]
    
    def __call__(self, prototype:ValQuery,
                 idx_constraint:TIdxConstraint=None) -> ListQuery:
        assert self._idx_constraint is None
        self.weave_input(name=ListConsts.ELT, inp=prototype)
        self._idx_constraint = idx_constraint
        outp:ListQuery = ListQuery.from_tp(tp=ListType(elt_type=prototype.tp))
        self.weave_output(name=ListConsts.LIST, outp=outp)
        return outp
    

class DeconstructListQuery(FuncQuery):
    def __init__(self, tags:TDict[str, TAny]=None):
        self._inputs = {}
        self._outputs = {}
        self._tags = {} if tags is None else tags
        self._op = DeconstructList()
        self._idx_constraint:TIdxConstraint = None
        attach_to_context(self)
    
    @property
    def idx_constraint(self) -> TIdxConstraint:
        return self._idx_constraint
     
    @property
    def output_names(self) -> TList[str]:
        return [ListConsts.ELT]
    
    def __call__(self, lst:ListQuery, idx_constraint:TIdxConstraint) -> ValQuery:
        self.weave_input(name=ListConsts.LIST, inp=lst)
        assert self._idx_constraint is None
        self._idx_constraint = idx_constraint
        outp = ValQuery.from_tp(tp=lst.tp.elt_type)
        self.weave_output(name=ListConsts.ELT, outp=outp)
        return outp


class GetKeyQuery(FuncQuery):
    def __init__(self, dct:DictQuery=None,
                 key:ValQuery=None, value:ValQuery=None, 
                 tags:TDict[str, TAny]=None):
        self._inputs = {}
        self._outputs = {}
        self._tags = {} if tags is None else tags
        self._op = GetKeyDict()
        if dct is not None:
            self._inputs[DictConsts.DICT] = dct
        if key is not None:
            self._inputs[DictConsts.KEY] = key
        if value is not None:
            self._outputs[DictConsts.VALUE] = value
        attach_to_context(self)
        
    @property
    def tags(self) -> TDict[str, TAny]:
        return self._tags
    
    @property
    def op(self) -> Operation:
        return self._op
    
    @property
    def output_names(self) -> TList[str]:
        return [DictConsts.VALUE]
    
    def __call__(self, dct:DictQuery, key:ValQuery) -> ValQuery:
        self.weave_input(name=DictConsts.DICT, inp=dct)
        self.weave_input(name=DictConsts.KEY, inp=key)
        outp = ValQuery.from_tp(tp=dct.tp.value_type)
        self.weave_output(name=DictConsts.VALUE, outp=outp)
        return outp


TKeyConstraint = TOption[TUnion[str, TList[str]]]
class ConstructDictQuery(FuncQuery):
    def __init__(self, tags:TDict[str, TAny]=None):
        self._inputs = {}
        self._outputs = {}
        self._tags = {} if tags is None else tags
        self._op = ConstructDict()
        self._key_constraint:TKeyConstraint = None
        attach_to_context(self)
    
    @property
    def key_constraint(self) -> TKeyConstraint:
        return self._key_constraint
     
    @property
    def output_names(self) -> TList[str]:
        return [DictConsts.DICT]
    
    def __call__(self, prototype:ValQuery,
                 key_constraint:TKeyConstraint=None) -> DictQuery:
        self.weave_input(name=DictConsts.VALUE, inp=prototype)
        assert self._key_constraint is None
        self._key_constraint = key_constraint
        outp:DictQuery = DictQuery.from_tp(tp=DictType(value_type=prototype.tp))
        self.weave_output(name=DictConsts.DICT, outp=outp)
        return outp