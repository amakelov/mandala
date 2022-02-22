from collections.abc import Sequence, Mapping

from .bases import (
    Operation, ValueRef, BaseSignature, TVRefCollection, VRefCollecUtils,
    Call, unwrap, get_content_hash_with_type
)
from .config import CoreConsts, CoreConfig
from .exceptions import VRefNotInMemoryError
from .idx import OpIndex, ValueIndex, BuiltinOpClasses
from .sig import BaseSignature, Signature, BaseSigMap, SigMap
from .tps import Type, ListType, AnyType, AtomType, BuiltinTypes, DictType
from .utils import AnnotatedObj

from ..common_imports import *
from ..util.common_ut import rename_dict_keys, invert_dict

UID_COL = CoreConsts.UID_COL

################################################################################
### calls
################################################################################
class BaseCall(Call): # todo - unfortunate name b/c backward compat
    def __init__(self, op:Operation, uid:str,
                 inputs:TDict[str, ValueRef], 
                 outputs:TDict[str, ValueRef],
                 metadata:TAny=None,
                 exec_interval:TTuple[float, float]=None):
        assert uid is not None
        self._op = op
        self._inputs = inputs
        self._outputs = outputs
        self._metadata = {} if metadata is None else metadata
        self._uid = uid
        self._exec_interval = exec_interval

    @property
    def op(self) -> Operation:
        return self._op
    
    @property
    def inputs(self) -> TDict[str, ValueRef]:
        return self._inputs
    
    @property
    def outputs(self) -> TDict[str, ValueRef]:
        return self._outputs
    
    @property
    def metadata(self) -> TDict[str, TAny]:
        return self._metadata
    
    @property
    def uid(self) -> str:
        # should raise when requested but no UID is present
        return self._uid
    
    @staticmethod
    def from_execution_data(op:Operation, uid:str, inputs:TDict[str, ValueRef],
                            outputs:TDict[str, ValueRef],
                            metadata:TAny) -> 'Call':
        return BaseCall(op=op, inputs=inputs, outputs=outputs, 
                        metadata=metadata, uid=uid)
    
    def detached(self) -> 'Call':
        return BaseCall(op=self.op.detached(), uid=self.uid, 
                        inputs={k: v.detached() for k, v in self.inputs.items()},
                        outputs={k: v.detached() for k, v in self.outputs.items()}, 
                        metadata=self.metadata, 
                        exec_interval=self.exec_interval)
    
    @property
    def exec_interval(self) -> TOption[TTuple[float, float]]:
        return self._exec_interval
    
    @exec_interval.setter
    def exec_interval(self, interval:TTuple[float, float]):
        assert self._exec_interval is None
        self._exec_interval = interval

################################################################################
### values
################################################################################
class AtomRef(ValueRef):
    def __init__(self, obj:TAny, in_memory:bool=True, uid:str=None,
                 persistable:bool=True, tp:Type=None, delayed_storage:bool=False):
        self._obj = obj
        self._in_memory = in_memory
        self._persistable = persistable
        self._uid = uid
        self._tp = AnyType() if tp is None else tp
        self._is_persisted = False
        self._delayed_storage = delayed_storage
        self._check_state()
    
    def _check_state(self):
        if (not self._persistable) and self.delayed_storage:
            raise ValueError('Invalid state:...')
        if (self._obj is not None) and (not self._in_memory):
            raise ValueError('Invalid state:...')
        if not isinstance(self._tp, (AtomType, AnyType)):
            raise ValueError('Invalid state:...')
    
    ############################################################################ 
    ### storage properties
    ############################################################################ 
    @property
    def in_memory(self) -> bool:
        return self._in_memory
    
    def set_in_memory(self, value: bool):
        self._in_memory = value
    
    @property
    def is_persistable(self) -> bool:
        return self._persistable
    
    def set_is_persistable(self, value: bool):
        self._persistable = value
    
    @property
    def delayed_storage(self) -> bool:
        return getattr(self, '_delayed_storage', False)
    
    ############################################################################ 
    ### detachment/attachment
    ############################################################################ 
    def detached(self) -> 'ValueRef':
        return AtomRef(obj=None, in_memory=False, uid=self.uid, 
                       persistable=self._persistable, tp=self.get_type(),
                       delayed_storage=self.delayed_storage)
    
    def eq_detached(self, other) -> bool:
        if not isinstance(other, AtomRef):
            return False
        return all(other.__dict__[k] == v 
                   for k, v in self.__dict__.items() if k != '_obj')
    
    def attached(self, obj: TAny) -> 'ValueRef':
        assert not self.in_memory
        return AtomRef(obj=obj, in_memory=True, uid=self.uid,
                       persistable=self._persistable, tp=self.get_type(),
                       delayed_storage=self.delayed_storage)
    
    def attach(self, obj:TAny):
        assert not self.in_memory
        self._obj = obj
        self._in_memory = True
    
    @property
    def uid(self) -> str:
        if self._uid is None:
            raise ValueError()
        return self._uid
    
    def _set_uid(self, uid: str):
        self._uid = uid
    
    def obj(self) -> TAny:
        if not self._in_memory:
            raise VRefNotInMemoryError(f'The value reference {self} was requested but not found in memory.')
        return self._obj

    @staticmethod
    def is_compound() -> bool:
        return False
    
    ############################################################################ 
    ### storage interface
    ############################################################################ 
    def get_residue(self) -> TTuple[TAny, TVRefCollection]:
        assert self.in_memory
        # return object residue + constituents
        return self.obj(), VRefCollecUtils.get_empty()
    
    @staticmethod
    def obj_from_residue(obj:TAny, constituents:TVRefCollection) -> TAny:
        return obj
    
    ### 
    def get_type(self) -> Type:
        return self._tp
    
    def set_type(self, tp:Type):
        assert isinstance(tp, (AtomType, AnyType))
        self._tp = tp

    def unwrap(self) -> TAny:
        self._auto_attach(shallow=True)
        return self.obj()
    
    def __repr__(self) -> str:
        obj_repr = repr(self._obj)
        short_obj = f'{repr(self._obj)[:20]}{"..." if len(obj_repr) > 20 else ""}'
        prefix = 'AtomRef'
        if not self.in_memory:
            return f'{prefix}(in_memory={self.in_memory}, type={self.get_type()})'
        else:
            return f'{prefix}({short_obj}, type={self.get_type()})'
    
    ############################################################################ 
    ### operator magics
    ############################################################################ 
    def _init_magic(self):
        assert CoreConfig.enable_vref_magics
        self._auto_attach(shallow=True)
    
    ### typecasting
    def __bool__(self) -> bool:
        self._init_magic()
        return self.obj().__bool__()
    
    def __int__(self) -> int:
        self._init_magic()
        return self.obj().__int__()
        
    def __index__(self) -> int:
        self._init_magic()
        return self.obj().__index__()
        
    ### comparison
    def __lt__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__lt__(other)
    
    def __le__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__le__(other)
    
    def __eq__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__eq__(other)
    
    def __ne__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__ne__(other)
    
    def __gt__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__gt__(other)
    
    def __ge__(self, other:TAny) -> bool:
        self._init_magic()
        return self.obj().__ge__(other)
    
    ### binary operations
    def __add__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__add__(other)
    
    def __sub__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__sub__(other)
    
    def __mul__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__mul__(other)
    
    def __floordiv__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__floordiv__(other)
    
    def __truediv__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__truediv__(other)
    
    def __mod__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__mod__(other)
    
    def __or__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__or__(other)
    
    def __and__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__and__(other)
    
    def __xor__(self, other:TAny) -> TAny:
        self._init_magic()
        return self.obj().__xor__(other)
    
    
class ListRef(ValueRef, Sequence):

    def __init__(self, obj:TList[ValueRef], in_memory:bool=True, uid:str=None,
                 persistable:bool=True, tp:ListType=None):
        assert all(isinstance(elt, ValueRef) for elt in obj)
        self._obj = obj
        self._in_memory = in_memory
        self._persistable = persistable
        self._uid = uid
        self._tp = ListType() if tp is None else tp
        self._is_persisted = False
        self._delayed_storage = False
    
    ### storage properties
    @property
    def in_memory(self) -> bool:
        return self._in_memory

    def set_in_memory(self, value: bool):
        self._in_memory = value

    @property
    def is_persistable(self) -> bool:
        return self._persistable

    def set_is_persistable(self, value: bool):
        self._persistable = value
    
    @property
    def delayed_storage(self) -> bool:
        return getattr(self, '_delayed_storage', False)
    
    ###
    def detached(self) -> 'ValueRef':
        return ListRef(obj=[], in_memory=False, uid=self.uid, 
                       persistable=self._persistable, tp=self.get_type())
    
    def eq_detached(self, other) -> bool:
        if not isinstance(other, ListRef):
            return False
        return (self._in_memory == other._in_memory
                and self._persistable == other._persistable
                and self._uid == other._uid and self._tp == other._tp)
    
    def attached(self, obj:TList[ValueRef]) -> 'ValueRef':
        assert not self.in_memory
        return ListRef(obj=obj, in_memory=True, uid=self.uid, 
                       persistable=self._persistable, tp=self.get_type())
    
    def attach(self, obj:TList[ValueRef]):
        assert not self.in_memory
        self._obj = obj
        self._in_memory = True
    
    @property
    def uid(self) -> str:
        if self._uid is None:
            raise ValueError()
        return self._uid
    
    def _set_uid(self, uid: str):
        self._uid = uid
    
    def obj(self) -> TList[ValueRef]:
        assert self._in_memory
        return self._obj

    @staticmethod
    def is_compound() -> bool:
        return True
    
    ### 
    def get_residue(self) -> TTuple[TAny, TVRefCollection]:
        assert self.in_memory
        return None, self.obj()
    
    @staticmethod
    def obj_from_residue(obj:TAny, constituents:TVRefCollection) -> TAny:
        return constituents
    
    ### 
    def get_type(self) -> ListType:
        return self._tp
    
    def set_type(self, tp:ListType):
        assert isinstance(tp, ListType)
        self._tp = tp

    ############################################################################ 
    ### list interface
    ############################################################################ 
    def __getitem__(self, idx:int) -> ValueRef:
        self._auto_attach(shallow=True)
        return self.obj()[idx]
    
    def __iter__(self):
        self._auto_attach(shallow=True)
        return iter(self.obj())
    
    def __len__(self) -> int:
        self._auto_attach(shallow=True)
        return len(self.obj())

    ### 
    def unwrap(self) -> TAny:
        self._auto_attach(shallow=False)
        return [elt.unwrap() for elt in self.obj()]

    def __repr__(self) -> str:
        obj_repr = repr(self._obj)
        short_obj = f'{repr(self._obj)[:20]}{"..." if len(obj_repr) > 20 else ""}'
        prefix = 'ListRef'
        if self.in_memory:
            return f'{prefix}({short_obj}, elt_type={self.get_type().elt_type})'
        else:
            return (f'{prefix}(in_memory={self.in_memory}, '
                    f'elt_type={self.get_type().elt_type})')


class DictRef(ValueRef, Mapping):

    def __init__(self, obj:TDict[str, ValueRef],
                 in_memory:bool=True, uid:str=None,
                 persistable:bool=True, tp:DictType=None):
        assert all(isinstance(k, str) for k in obj)
        assert all(isinstance(v, ValueRef) for v in obj.values())
        self._obj = obj
        self._in_memory = in_memory
        self._persistable = persistable
        self._uid = uid
        self._tp = DictType() if tp is None else tp
        self._is_persisted = False
        self._delayed_storage = False
    
    ### storage properties
    @property
    def in_memory(self) -> bool:
        return self._in_memory
    
    def set_in_memory(self, value:bool):
        self._in_memory = value
    
    @property
    def is_persistable(self) -> bool:
        return self._persistable
    
    def set_is_persistable(self, value: bool):
        self._persistable = value
    
    @property
    def delayed_storage(self) -> bool:
        return getattr(self, '_delayed_storage', False)
    
    ### 
    def detached(self) -> 'ValueRef':
        return DictRef(obj={}, in_memory=False, uid=self.uid, 
                       persistable=self._persistable, tp=self.get_type())
    
    def eq_detached(self, other) -> bool:
        if not isinstance(other, DictRef):
            return False
        return (self._in_memory == other._in_memory
                and self._persistable == other._persistable
                and self._uid == other._uid and self._tp == other._tp)
    
    def attached(self, obj:TDict[str, ValueRef]) -> 'ValueRef':
        assert not self.in_memory
        return DictRef(obj=obj, in_memory=True, uid=self.uid,
                       persistable=self._persistable, tp=self.get_type())
    
    def attach(self, obj:TDict[str, ValueRef]):
        assert not self.in_memory
        self._obj = obj
        self._in_memory = True

    @property
    def uid(self) -> str:
        if self._uid is None:
            raise ValueError()
        return self._uid
    
    def _set_uid(self, uid: str):
        self._uid = uid
    
    def obj(self) -> TDict[str, ValueRef]:
        assert self._in_memory
        return self._obj

    @staticmethod
    def is_compound() -> bool:
        return True

    ### 
    def get_residue(self) -> TTuple[TAny, TVRefCollection]:
        assert self.in_memory
        return None, self.obj()
    
    @staticmethod
    def obj_from_residue(obj:TAny, constituents:TVRefCollection) -> TAny:
        return constituents
    
    ### 
    def get_type(self) -> DictType:
        return self._tp
    
    def set_type(self, tp:DictType):
        assert isinstance(tp, DictType)
        self._tp = tp

    ### 
    def unwrap(self) -> TAny:
        self._auto_attach(shallow=False)
        return {k: v.unwrap() for k, v in self.obj().items()}

    def __repr__(self) -> str:
        obj_repr = repr(self._obj)
        short_obj = f'{repr(self._obj)[:20]}{"..." if len(obj_repr) > 20 else ""}'
        prefix = 'DictRef'
        if self.in_memory:
            return f'{prefix}({short_obj}, value_type={self.get_type().value_type})'
        else:
            return f'{prefix}(in_memory={self.in_memory}, value_type={self.get_type().value_type})'
    
    ### dict interface
    def __dir__(self) -> TList[str]:
        if not self.in_memory:
            return []
        else:
            return list(self.keys())

    def __getitem__(self, key:int) -> ValueRef:
        self._auto_attach(shallow=True)
        return self.obj()[key]
    
    def __len__(self) -> int:
        self._auto_attach(shallow=True)
        return len(self.obj())
    
    def __iter__(self):
        self._auto_attach(shallow=True)
        return iter(self.obj())
    
    def keys(self):
        if not self.in_memory:
            raise ValueError()
        return self.obj().keys()


ValueIndex.register(AtomRef)
ValueIndex.register(ListRef)
ValueIndex.register(DictRef)

################################################################################
### operations
################################################################################
class SimpleFunc(Operation): # todo - unfortunate name b/c backward compat
    VERSION_COL = '__version__'
    TAG_COL = '__tag__'
    CONTEXT_KW = '__context__'

    def __init__(self, func:TCallable=None, output_names:TList[str]=None, 
                 version:str=None, is_super:bool=False, unwrap_inputs:bool=True,
                 var_outputs:bool=False, 
                 mutations:TDict[str, int]=None):
        self._func = func
        self._var_outputs = var_outputs
        self._mutations = mutations
        if func is None:
            self._sig = None
            self._orig_sig = None
            self._sig_map = None
            self._name = None
            self._ui_name = None
        else:
            self._sig = Signature.from_callable(
                clbl=func,
                output_names=output_names,
                excluded_names=[self.CONTEXT_KW],
                fixed_outputs=not self._var_outputs
            )
            self._orig_sig = Signature.from_callable(
                clbl=func,
                output_names=output_names,
                excluded_names=[self.CONTEXT_KW],
                fixed_outputs=not self._var_outputs
            )
            self._sig_map = SigMap(source=self.orig_sig, target=self.sig, 
                                kwarg_map={k: k for k in self.sig.kw})
            self._name = self.func.__name__
            self._ui_name = self.func.__name__
        self._metadata_cols = ['__tag__', '__version__']
        self._version = '0' if version is None else version
        self._is_super = is_super
        self._unwrap_inputs = unwrap_inputs
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def mutations(self) -> TDict[str, int]:
        return self._mutations
    
    @property
    def unwrap_inputs(self) -> bool:
        if hasattr(self, '_unwrap_inputs'): # backward compat.
            return self._unwrap_inputs
        else:
            return True
    
    def set_sig(self, sigmap:BaseSigMap):
        assert sigmap.source == self.orig_sig
        self._sig = sigmap.target
        self._sig_map = sigmap
    
    def set_name(self, name:str):
        self._name = name
        
    @property
    def orig_sig(self) -> BaseSignature:
        return self._orig_sig
    
    @property
    def sig_map(self) -> BaseSigMap:
        return self._sig_map
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def ui_name(self) -> str:
        return self._ui_name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def is_builtin(self) -> bool:
        return False
    
    @property
    def is_super(self) -> bool:
        return self._is_super
    
    def _serialize(self) -> TDict[str, TAny]:
        data = {k: v for k, v in self.__dict__.items() if k != '_func'}
        #! importantly, remove non-permanent information
        assert '_ui_name' in data
        data['_ui_name'] = None
        assert '_orig_sig' in data
        data['_orig_sig'] = None
        assert '_sig_map' in data
        data['_sig_map'] = None
        return data
    
    @staticmethod
    def _deserialize(data:TDict[str, TAny]) -> 'SimpleFunc':
        res = SimpleFunc()
        for k, v in data.items():
            setattr(res, k, v)
        return res

    def detached(self) -> 'Operation':
        return SimpleFunc._deserialize(self._serialize())
    
    @property
    def func(self) -> TCallable:
        return self._func
    
    @property
    def output_names(self) -> TOption[TList[str]]:
        return self.sig.output_names
    
    @property
    def num_outputs(self) -> int:
        return len(self.output_names)
    
    def compute(self, inputs: TDict[str, TAny], context_representation:TAny=None, 
                context_arg:str=None) -> TDict[str, TAny]:
        # need to use the original function's signature for this 
        inv_map = self.sig_map.inverse()
        input_map = inv_map.map_input_names(input_names=inputs.keys())
        ui_inputs = rename_dict_keys(dct=inputs, mapping=input_map)
        if context_arg is not None:
            assert context_arg not in ui_inputs
            assert context_representation is not None
            ui_inputs[context_arg] = context_representation
        outcome = self.func(**ui_inputs)
        if self.sig.has_fixed_outputs:
            assert self.output_names is not None
            if self.num_outputs == 0:
                assert outcome is None
                return {}
            elif self.num_outputs == 1:
                return {self.output_names[0]: outcome}
            else:
                assert isinstance(outcome, tuple)
                assert len(outcome) == len(self.output_names)
                return {self.output_names[i]: elt for i, elt in enumerate(outcome)}
        else:
            if outcome is None:
                res = {}
            elif isinstance(outcome, tuple):
                res = {f'output_{i}': outcome[i] for i in range(len(outcome))}
            else:
                res = {'output_0': outcome}
            # enforce explicit type annotations
            for k, v in res.items():
                assert isinstance(v, AnnotatedObj)
                assert v.target_type is not None
            return res
    
    def compute_wrapped(self, inputs: TDict[str, ValueRef], 
                        input_types: TDict[str, Type],
                        output_types: TDict[str, Type],
                        type_dict: TDict[str, Type]) -> TDict[str, ValueRef]:
        raise NotImplementedError()
    
    def repr_call(self, call:BaseCall, output_names:TList[str],
                  input_reprs:TDict[str, str]) -> str:
        internal_to_ui_inputs = invert_dict(self.sig_map.fixed_inputs_map())
        outputs_part = ', '.join(output_names)
        args_part = ', '.join(
            [f'{internal_to_ui_inputs[input_name]}={input_reprs[input_name]}' 
             for input_name in call.inputs.keys()])
        op_part = f'{self.ui_name}({args_part})'
        return f'{outputs_part} = {op_part}'

    ############################################################################ 
    ### relation methods
    ############################################################################ 
    @classmethod
    def get_relation(cls, input_uids: TDict[str, str], 
                     output_uids: TDict[str, str],
                     metadata: TDict[str, TAny], 
                     call_uid:str) -> pd.DataFrame:
        data = {**input_uids, **output_uids}
        data[UID_COL] = call_uid
        data[cls.VERSION_COL] = metadata[cls.VERSION_COL]
        data[cls.TAG_COL] = metadata[cls.TAG_COL]
        assert all(isinstance(v, str) for v in data.values())
        return pd.DataFrame({k: [v] for k, v in data.items()})
    
    def get_cols(self) -> TOption[TIter[str]]:
        if not self.sig.is_fixed:
            return None
        return [UID_COL, *self.sig.fixed_input_names, 
                *self.output_names, *self._metadata_cols]
    
    def get_vref_cols(self) -> TIter[str]:
        return [*self.sig.fixed_input_names, *self.output_names]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return []
    
    def get_primary_key(self) -> TOption[str]:
        return UID_COL

    ############################################################################ 
    ### magics
    ############################################################################ 
    def __repr__(self) -> str:
        data = {
            'name': self.ui_name,
            'version': self.version,
            'signature': self.orig_sig
        }
        return (f'Operation(name={data["name"]}, version={data["version"]}, '
                f'signature={data["signature"]})')

################################################################################
### list operations
################################################################################
class GetItemList(Operation): # !deprecated
    _call_name = UID_COL
    _list_name = 'list'
    _idx_name = 'idx'
    _elt_name = 'elt'

    def __init__(self):
        self._sig = Signature(ord_poskw=[(self._list_name, ListType()),
                                         (self._idx_name, AnyType())],
                              ord_outputs=[('elt', AnyType())])
    
    @property
    def is_super(self) -> bool:
        return False
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def name(self) -> str:
        return OpIndex.get_impl_id(cls=type(self))
    
    @property
    def version(self) -> str:
        return '0'

    @property
    def is_builtin(self) -> bool:
        return True
    
    def detached(self) -> 'Operation':
        return GetItemList()

    def compute(self, inputs:TDict[str, TAny]) -> TDict[str, TAny]:
        lst = inputs[self._list_name]
        idx = inputs[self._idx_name]
        return {self._elt_name: lst[idx]}
    
    def compute_wrapped(self, inputs: TDict[str, ValueRef],
                        _write_uids:bool=False, 
                        _require_matching_uids:bool=False) -> TTuple[TDict[str, ValueRef], Call]:
        """
        Call the operation on vrefs and return the dictionary of outputs and the
        resulting call. Optionally, write the UID of the output that it would be
        assigned by the call, IF it has no UID. This UID could be either causal
        or content-based
        """
        call_start = time.time()
        lst, idx = inputs[self._list_name], inputs[self._idx_name]
        assert isinstance(lst, ListRef)
        assert isinstance(idx, AtomRef)
        input_uids = {k: v.uid for k, v in inputs.items()}
        result:ValueRef = lst.obj()[idx.obj()]
        outputs = {self._elt_name: result}
        op = GetItemList()
        raw_outputs = {k: unwrap(v) for k, v in outputs.items()}
        output_types = {k: v.get_type() for k, v in outputs.items()}
        call_uid, output_uids = self.get_call_and_output_uids(
            input_uids=input_uids,
            metadata={},
            raw_outputs=raw_outputs,
            output_types=output_types
        )
        result_uid = output_uids[self._elt_name]
        if _write_uids:
            if result._uid is None:
                result._set_uid(uid=result_uid)
        else:
            if _require_matching_uids:
                assert result.uid == result_uid
        call = BaseCall.from_execution_data(op=op, uid=call_uid, 
                                            inputs=inputs, outputs=outputs,
                                            metadata={})
        call_end = time.time()
        call.exec_interval = (call_start, call_end)
        return outputs, call
    
    def repr_call(self, call:'Call', output_names:TList[str], 
                  input_reprs:TDict[str, str]) -> str:
        assert len(output_names) == 1
        return f'{output_names[0]} = {input_reprs[self._list_name]}[{input_reprs[self._idx_name]}]'
        
    @classmethod
    def get_relation(cls, input_uids:TDict[str, str], 
                     output_uids:TDict[str, str], 
                     metadata:TDict[str, TAny], 
                     call_uid:str) -> pd.DataFrame:
        assert not metadata
        data = {**input_uids, **output_uids}
        data[cls._call_name] = call_uid
        assert all(isinstance(v, str) for v in data.values())
        return pd.DataFrame({k: [v] for k, v in data.items()})
    
    def get_cols(self) -> TIter[str]:
        return [self._call_name, self._list_name, 
                self._idx_name, self._elt_name]
    
    def get_vref_cols(self) -> TIter[str]:
        return [self._list_name, self._idx_name, self._elt_name]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return [[self._list_name, self._idx_name]]
    
    def get_primary_key(self) -> str:
        return self._call_name
    

class ConstructList(Operation):
    # todo - should really be properties
    _call_name = UID_COL
    _list_name = 'list'
    _idx_name = 'idx'
    _elt_name = 'elt'

    def __init__(self):
        self._sig = Signature(varkwarg=('elts', AnyType()), 
                              ord_outputs=[(self._list_name, ListType())])
    
    @property
    def is_builtin(self) -> bool:
        return True
    
    @property
    def is_super(self) -> bool:
        return False
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def name(self) -> str:
        return OpIndex.get_impl_id(cls=type(self))
    
    @property
    def version(self) -> str:
        return '0'

    def detached(self) -> 'Operation':
        return ConstructList()
    
    def compute(self, inputs: TDict[str, TAny]) -> TDict[str, TAny]:
        return {self._list_name: [inputs[str(i)] for i in range(len(inputs))]}

    def compute_wrapped(self, inputs:TDict[str, ValueRef],
                        output_types:TDict[str, Type],
                        ) -> TTuple[TDict[str, ValueRef], Call]:
        call_start = time.time()
        numeric_range = range(len(inputs))
        input_uids = {k: v.uid for k, v in inputs.items()}
        op = ConstructList()
        if output_types:
            assert set(output_types) == {self._list_name}
            res_tp = output_types[self._list_name]
            assert isinstance(res_tp, ListType)
            metadata = {'output_type': json.dumps(res_tp.dump())}
        else:
            res_tp = ListType()
            metadata = {}
        output_obj = [inputs[str(i)] for i in numeric_range]
        raw_outputs = {self._list_name: output_obj}
        output_types = {self._list_name: res_tp}
        call_uid, output_uids = self.get_call_and_output_uids(
            input_uids=input_uids, metadata=metadata, raw_outputs=raw_outputs,
            output_types=output_types
        )
        res_uid = output_uids[self._list_name]
        res = ListRef(obj=[inputs[str(i)] for i in numeric_range],
                      in_memory=True, uid=res_uid, persistable=True)
        res.set_type(tp=res_tp)
        outputs = {self._list_name: res}
        call = BaseCall.from_execution_data(op=op, uid=call_uid, inputs=inputs,
                                            outputs=outputs, metadata={})
        call_end = time.time()
        call.exec_interval = (call_start, call_end)
        return outputs, call
    
    def repr_call(self, call:'Call', output_names:TList[str],
                  input_reprs:TDict[str, str]) -> str:
        assert len(output_names) == 1
        sorted_inputs = sorted([int(k) for k in input_reprs])
        items_part = ', '.join([input_reprs[str(k)] for k in sorted_inputs])
        return f'{output_names[0]} = [{items_part}]'

    @classmethod
    def get_relation(cls, input_uids: TDict[str, str], 
                     output_uids: TDict[str, str],
                     metadata: TDict[str, TAny], 
                     call_uid:str) -> pd.DataFrame:
        assert not metadata
        list_uid = output_uids[cls._list_name]
        num_elts = len(input_uids)
        element_uids = [input_uids[str(i)] for i in range(num_elts)]
        if CoreConfig.include_constructive_indexing_in_rels:
            logging.warning('This feature is not fully implemented')
            idx_tp = BuiltinTypes.get(py_type=int)
            idx_uids = [get_content_hash_with_type(raw_obj=i, tp=idx_tp) 
                        for i in range(num_elts)]
        else:
            element_idxs = [str(i) for i in range(num_elts)]
            idx_uids = element_idxs 
        return pd.DataFrame({
            cls._list_name: list_uid,
            cls._elt_name: element_uids,
            cls._idx_name: idx_uids,
            cls._call_name: call_uid
        })
    
    def get_cols(self) -> TIter[str]:
        return [self._call_name, self._list_name, 
                self._idx_name, self._elt_name]
    
    def get_vref_cols(self) -> TIter[str]:
        return [self._list_name, self._elt_name]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return [[self._list_name, self._idx_name]]
    
    def get_primary_key(self) -> TOption[str]:
        return None


class DeconstructList(Operation):
    _call_name = UID_COL
    _list_name = 'list'
    _idx_name = 'idx'
    _elt_name = 'elt'

    def __init__(self):
        self._sig = Signature(ord_poskw=[(self._list_name, ListType())], 
                              fixed_outputs=False)
    
    @property
    def is_builtin(self) -> bool:
        return True
    
    @property
    def is_super(self) -> bool:
        return False
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def name(self) -> str:
        return OpIndex.get_impl_id(cls=type(self))
    
    @property
    def version(self) -> str:
        return '0'

    def detached(self) -> 'Operation':
        return DeconstructList()
    
    def compute(self, inputs: TDict[str, TAny]) -> TDict[str, TAny]:
        lst = inputs[self._list_name]
        return {str(i): lst[i] for i in range(len(lst))}

    def compute_wrapped(self, inputs:TDict[str, ListRef],
                        _write_uids:bool=False,
                        _require_matching_uids:bool=False) -> TTuple[TDict[str, ValueRef], Call]:
        call_start = time.time()
        lst = inputs[self._list_name]
        assert isinstance(lst, ListRef)
        input_uids = {k: v.uid for k, v in inputs.items()}
        outputs = {str(i): lst.obj()[i] for i in range(len(lst))}
        op = DeconstructList()
        raw_outputs = {k: unwrap(v) for k, v in outputs.items()}
        output_types = {k: v.get_type() for k, v in outputs.items()}
        call_uid, output_uids = self.get_call_and_output_uids(input_uids=input_uids, 
                                                              metadata={},
                                                              raw_outputs=raw_outputs,
                                                              output_types=output_types)
        if _write_uids:
            for k, v in outputs.items():
                if v._uid is None:
                    v._set_uid(uid=output_uids[k])
        else:
            if _require_matching_uids:
                assert all(output_uids[k] == outputs[k].uid for k in output_uids)
        call = BaseCall.from_execution_data(op=op, uid=call_uid, inputs=inputs, 
                                            outputs=outputs, metadata={})
        call_end = time.time()
        call.exec_interval = (call_start, call_end)
        return outputs, call
    
    def repr_call(self, call:'Call', output_names:TList[str], input_reprs:TDict[str, str]) -> str:
        raise NotImplementedError()

    @classmethod
    def get_relation(cls, input_uids: TDict[str, str], 
                     output_uids: TDict[str, str],
                     metadata: TDict[str, TAny], 
                     call_uid:str) -> pd.DataFrame:
        """
        The relation returned is the same as the one for construct list
        """
        return ConstructList.get_relation(input_uids=output_uids,
                                          output_uids=input_uids, 
                                          metadata={}, call_uid=call_uid)
    
    def get_cols(self) -> TIter[str]:
        return [self._call_name, self._list_name,
                self._idx_name, self._elt_name]
    
    def get_vref_cols(self) -> TIter[str]:
        return [self._list_name, self._elt_name]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return [[self._list_name, self._idx_name]]
    
    def get_primary_key(self) -> TOption[str]:
        return None


################################################################################
### dict operations
################################################################################
class GetKeyDict(Operation):
    _call_name = UID_COL
    _dict_name = 'dict'
    _key_name = 'key'
    _value_name = 'value'

    def __init__(self):
        self._sig = Signature(ord_poskw=[(self._dict_name, DictType()),
                                    (self._key_name, AnyType())], 
                              ord_outputs=[('value', AnyType())])
    
    @property
    def is_builtin(self) -> bool:
        return True
    
    @property
    def is_super(self) -> bool:
        return False
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def name(self) -> str:
        return OpIndex.get_impl_id(cls=type(self))
    
    @property
    def version(self) -> str:
        return '0'

    def detached(self) -> 'Operation':
        return GetKeyDict()

    def compute(self, inputs:TDict[str, TAny]) -> TDict[str, TAny]:
        dct = inputs[self._dict_name]
        key = inputs[self._key_name]
        return {self._value_name: dct[key]}
    
    def compute_wrapped(self, inputs: TDict[str, ValueRef],
                        _write_uids:bool=False,
                        _require_matching_uids:bool=False,
                        ) -> TTuple[TDict[str, ValueRef], 'Call']:
        call_start = time.time()
        dct, key = inputs[self._dict_name], inputs[self._key_name]
        assert isinstance(dct, DictRef)
        assert isinstance(key, AtomRef)
        input_uids = {k: v.uid for k, v in inputs.items()}
        result:ValueRef = dct.obj()[key.obj()]
        outputs = {self._value_name: result}
        op = GetKeyDict()
        raw_outputs = {k: unwrap(v) for k, v in outputs.items()}
        output_types = {k: v.get_type() for k, v in outputs.items()}
        call_uid, output_uids = self.get_call_and_output_uids(
            input_uids=input_uids,
            metadata={},
            raw_outputs=raw_outputs,
            output_types=output_types
        )
        result_uid = output_uids[self._value_name]
        if _write_uids:
            if result._uid is None:
                result._set_uid(uid=result_uid)
        else:
            if _require_matching_uids:
                assert result.uid == result_uid
        call = BaseCall.from_execution_data(op=op, uid=call_uid, inputs=inputs,
                                            outputs=outputs, metadata={})
        call_end = time.time()
        call.exec_interval = (call_start, call_end)
        return outputs, call
    
    def repr_call(self, call:'Call', output_names:TList[str], 
                  input_reprs:TDict[str, str]) -> str:
        assert len(output_names) == 1
        return f'{output_names[0]} = {input_reprs[self._dict_name]}[{input_reprs[self._key_name]}]'
    
    @classmethod
    def get_relation(cls, input_uids:TDict[str, str], 
                     output_uids:TDict[str, str], 
                     metadata:TDict[str, TAny], call_uid:str) -> pd.DataFrame:
        assert not metadata
        data = {**input_uids, **output_uids}
        data[cls._call_name] = call_uid
        assert all(isinstance(v, str) for v in data.values())
        return pd.DataFrame({k: [v] for k, v in data.items()})
    
    def get_cols(self) -> TIter[str]:
        return [self._call_name, self._dict_name,
                self._key_name, self._value_name]
    
    def get_vref_cols(self) -> TIter[str]:
        return [self._dict_name, self._key_name, self._value_name]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return [[self._dict_name, self._key_name]]
    
    def get_primary_key(self) -> str:
        return self._call_name


class ConstructDict(Operation):
    _call_name = UID_COL
    _dict_name = 'dict'
    _key_name = 'key'
    _value_name = 'value'

    def __init__(self):
        self._sig = Signature(varkwarg=('elts', AnyType()), 
                              ord_outputs=[(self._dict_name, DictType())])
    
    @property
    def is_builtin(self) -> bool:
        return True
    
    @property
    def is_super(self) -> bool:
        return False
    
    @property
    def sig(self) -> BaseSignature:
        return self._sig
    
    @property
    def name(self) -> str:
        return OpIndex.get_impl_id(cls=type(self))
    
    @property
    def version(self) -> str:
        return '0'

    def detached(self) -> 'Operation':
        return ConstructDict()
    
    def compute(self, inputs: TDict[str, TAny]) -> TDict[str, TAny]:
        return {self._dict_name: inputs}

    def compute_wrapped(self, inputs:TDict[str, ValueRef],
                        output_types:TDict[str, Type],
                        ) -> TTuple[TDict[str, ValueRef], Call]:
        call_start = time.time()
        input_uids = {k: v.uid for k, v in inputs.items()}
        op = ConstructDict()
        if output_types:
            assert set(output_types) == {self._dict_name}
            res_tp = output_types[self._dict_name]
            assert isinstance(res_tp, DictType)
            metadata = {'output_type': json.dumps(res_tp.dump())}
        else:
            res_tp = DictType()
            metadata = {}
        raw_outputs = {self._dict_name: inputs}
        output_types = {self._dict_name: res_tp}
        call_uid, output_uids = self.get_call_and_output_uids(
            input_uids=input_uids,
            metadata=metadata,
            raw_outputs=raw_outputs,
            output_types=output_types
        )
        res_uid = output_uids[self._dict_name]
        res = DictRef(obj=inputs, in_memory=True, uid=res_uid, persistable=True)
        res.set_type(tp=res_tp)
        outputs = {self._dict_name: res}
        call = BaseCall.from_execution_data(op=op, uid=call_uid, inputs=inputs,
                                            outputs=outputs, metadata={})
        call_end = time.time()
        call.exec_interval = (call_start, call_end)
        return outputs, call
    
    def repr_call(self, call:'Call', output_names:TList[str], 
                  input_reprs:TDict[str, str]) -> str:
        assert len(output_names) == 1
        items_part = ', '.join([f'"{k}": {v}' for k, v in input_reprs.items()])
        return f'{output_names[0]} = {{{items_part}}}'

    @classmethod
    def get_relation(cls, input_uids: TDict[str, str], 
                     output_uids: TDict[str, str],
                     metadata: TDict[str, TAny], 
                     call_uid:str) -> pd.DataFrame:
        assert not metadata
        dict_uid = output_uids[cls._dict_name]
        element_keys = sorted(input_uids.keys())
        if CoreConfig.include_constructive_indexing_in_rels:
            logging.warning('This feature is not fully implemented')
            key_tp = BuiltinTypes.get(py_type=str)
            key_uids = [get_content_hash_with_type(raw_obj=k, tp=key_tp)
                        for k in element_keys]
        else:
            key_uids = element_keys 
        element_uids = [input_uids[k] for k in element_keys]
        return pd.DataFrame({
            cls._dict_name: dict_uid,
            cls._value_name: element_uids,
            cls._key_name: key_uids,
            cls._call_name: call_uid
        })
    
    def get_cols(self) -> TIter[str]:
        return [self._call_name, self._dict_name,
                self._key_name, self._value_name]
    
    def get_vref_cols(self) -> TIter[str]:
        return [self._dict_name, self._value_name]
    
    def get_multicol_indices(self) -> TList[TList[str]]:
        return [[self._dict_name, self._key_name]]
    
    def get_primary_key(self) -> TOption[str]:
        return None


### register operation implementations
OpIndex.register(SimpleFunc)
OpIndex.register(ConstructList)
OpIndex.register(GetItemList)
OpIndex.register(ConstructDict)
OpIndex.register(DeconstructList)
OpIndex.register(GetKeyDict)

BuiltinOpClasses.append(ConstructList)
BuiltinOpClasses.append(ConstructDict)
BuiltinOpClasses.append(GetItemList)
BuiltinOpClasses.append(DeconstructList)
BuiltinOpClasses.append(GetKeyDict)