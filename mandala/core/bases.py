from abc import abstractmethod, ABC

from .tps import Type, is_subtype, is_member
from .idx import OpIndex
from .sig import BaseSignature, BaseSigMap

from ..common_imports import *
from ..util.common_ut import (
    ContentHashing, CausalHashing, get_collection_atoms,
    concat_lists, extract_uniques
)
from ..util.common_ut import group_like
from ..storages.objects import BaseObjLocation

################################################################################
### vrefs
################################################################################
TVRefCollection = TUnion[
    TList['TVRefCollection'],
    TDict[str, 'TVRefCollection'],
    'ValueRef'
    ]

class VRefCollecUtils(object):

    @staticmethod
    def get_empty() -> TVRefCollection:
        return []
    

class ValueRef(ABC):
    """
    The core object representing values that are passed between operations.
    """
    @property
    @abstractmethod
    def uid(self) -> str:
        # should raise when requested but no UID is present
        raise NotImplementedError()
    
    @abstractmethod
    def _set_uid(self, uid:str):
        raise NotImplementedError()
    
    @abstractmethod
    def obj(self) -> TAny:
        """
        Return this vref's object (including constituent vrefs where
        applicable), raise error if not in memory
        """
        raise NotImplementedError()

    @staticmethod # b/c you can't create a static property in Python?
    @abstractmethod
    def is_compound() -> bool:
        raise NotImplementedError()
    
    ############################################################################ 
    ### storage properties
    ############################################################################ 
    @property
    @abstractmethod
    def in_memory(self) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def set_in_memory(self, value:bool):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def is_persistable(self) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def set_is_persistable(self, value:bool):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def delayed_storage(self) -> bool:
        raise NotImplementedError()
     
    ############################################################################ 
    ### attachment/detachment
    ############################################################################ 
    @abstractmethod
    def detached(self) -> 'ValueRef':
        raise NotImplementedError()
    
    @abstractmethod
    def attached(self, obj:TAny) -> 'ValueRef':
        raise NotImplementedError()
    
    @abstractmethod
    def attach(self, obj:TAny):
        """
        In-place attachment of an object. Use only when `self` is NOT in memory.
        """
        raise NotImplementedError()
    
    ### dark magics
    def _auto_attach(self, shallow:bool=False):
        """
        Used to implicitly attach so that relevant control flow can
        work
        """
        if shallow:
            if not self.in_memory:
                c = GlobalContext.get()
                c.storage.attach(self, shallow=True)
        else:
            if not is_deeply_in_memory(self):
                c = GlobalContext.get()
                c.storage.attach(self, shallow=False)

    ############################################################################ 
    ### storage interface
    ############################################################################ 
    @abstractmethod
    def get_residue(self) -> TTuple[TAny, TVRefCollection]:
        """
        Return (object residue, constitutent vrefs) for this vref. 
        
        The vref must be in memory for this to be applicable, however, it is not
        necessary for constituents to be in memory (they may be detached).
        """
        # return object residue + constituents
        return self.obj(), VRefCollecUtils.get_empty()
    
    @staticmethod
    @abstractmethod
    def obj_from_residue(obj:TAny, constituents:TVRefCollection) -> TAny:
        """
        Recover this vref's wrapped object in a purely structural way, meaning
        that this process should not rely on the *values* of the constituents
        (which may not be around when this is called), but only on how they fit
        into a data structure
        """
        raise NotImplementedError()
    
    ###
    @abstractmethod
    def get_type(self) -> Type:
        raise NotImplementedError()
    
    @abstractmethod
    def set_type(self, tp:Type):
        raise NotImplementedError()

    def unwrap(self) -> TAny:
        """
        Return *recursively* unwrapped value. Fails when value is not deeply in
        memory.
        """
        raise NotImplementedError()
        
    ############################################################################ 
    @abstractmethod
    def eq_detached(self, other:TAny) -> bool:
        """
        Check equality ignoring _obj
        """
        raise NotImplementedError()
    
    def __eq__(self, other:TAny) -> bool:
        raise NotImplementedError()

    def __repr__(self) -> str:
        data = {
            'type': self.get_type().annotation,
            'uid': self.uid,
            'transient': not self.is_persistable,
            'in_memory': self.in_memory,
            'obj': self._obj
        }
        parts = ', '.join([f'{k}={v}' for k, v in data.items()])
        return f'ValueRef({parts})'
     
################################################################################
### ops
################################################################################
class Operation(ABC):

    ############################################################################ 
    ### default implementations
    ############################################################################ 
    @classmethod
    def get_impl_id(cls) -> str:
        return OpIndex.get_impl_id(cls=cls)
    
    def get_call_uid(self, input_uids:TDict[str, str],
                     metadata:TDict[str, TAny]) -> str:
        op_id = self.name # importantly, we use the name connected to storage
        meta = copy.deepcopy(metadata)
        meta['op_id'] = op_id
        causal_hash = CausalHashing.hash_computation(input_hashes=input_uids, 
                                                     metadata=meta)
        return causal_hash
    
    @staticmethod
    def get_output_uids(call_uid:str, raw_outputs:TDict[str, TAny],
                        output_types:TDict[str, Type]) -> TDict[str, str]:
        res = {}
        for k, v in output_types.items():
            if v.hash_method == 'causal':
                res[k] = CausalHashing.hashforward_dict(cur_hash=call_uid, key=k)
            elif v.hash_method == 'content':
                res[k] = get_content_hash_with_type(raw_obj=raw_outputs[k], tp=v)
            else:
                raise ValueError()
        return res
    
    def get_call_and_output_uids(self, input_uids:TDict[str, str], 
                                 metadata:TDict[str, TAny],
                                 raw_outputs:TDict[str, TAny],
                                 output_types:TDict[str, Type]) -> TTuple[str, TDict[str, str]]:
        call_uid = self.get_call_uid(input_uids=input_uids, metadata=metadata)
        output_uids = self.get_output_uids(call_uid=call_uid, 
                                           raw_outputs=raw_outputs,
                                           output_types=output_types)
        return call_uid, output_uids

    @property
    def ui_name(self) -> str:
        return self.name
    
    def set_name(self, name:str):
        return
    
    @property
    def qualified_name(self) -> str:
        return f'{self.name}_{self.version}'
    
    @staticmethod
    def qualified_name_from_data(internal_name:str, version:str) -> str:
        return f'{internal_name}_{version}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operation):
            return False
        return (self.sig == other.sig 
                and self.name == other.name
                and self.ui_name == other.ui_name)

    def __repr__(self) -> str:
        data = {
            'name': self.ui_name,
            'version': self.version
        }
        return f'Operation(name={data["name"]}, version={data["version"]})'
    
    ############################################################################ 
    ### inteface
    ############################################################################ 
    @property
    @abstractmethod
    def is_super(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def compute(self, inputs:TDict[str, TAny]) -> TDict[str, TAny]:
        """
        Compute this operation on *raw*, unwrapped values.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_wrapped(self, inputs:TDict[str, ValueRef], 
                     input_types:TDict[str, Type], 
                     output_types:TDict[str, Type], 
                     type_dict:TDict[str, Type]) -> TTuple[TDict[str, ValueRef], 'Call']:
        raise NotImplementedError()
    
    @abstractmethod
    def detached(self) -> 'Operation':
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def sig(self) -> BaseSignature:
        raise NotImplementedError()
    
    @property
    def is_builtin(self) -> bool:
        raise NotImplementedError()
    
    def set_sig(self, sigmap:BaseSigMap):
        return
    
    def __hash__(self) -> int:
        return hash((self.name, self.version))

    @abstractmethod
    def repr_call(self, call:'Call', output_names:TList[str], 
                  input_reprs:TDict[str, str]) -> str:
        """
        For code gen
        """
        raise NotImplementedError()

    ###################################### 
    ### relation interface 
    ###################################### 
    @classmethod
    @abstractmethod
    def get_relation(cls, input_uids:TDict[str, str], 
                     output_uids:TDict[str, str],
                     metadata:TDict[str, TAny], call_uid:str) -> pd.DataFrame:
        """
        Construct relation table from call data
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_primary_key(self) -> TOption[str]:
        """
        Return the primary key for this operation, if any
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_multicol_indices(self) -> TList[TList[str]]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_cols(self) -> TIter[str]:
        """
        Return *all* columns in the table for this relation.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_vref_cols(self) -> TIter[str]:
        """
        Return the columns containing references to values. 
        """
        raise NotImplementedError()


################################################################################
### calls
################################################################################
class Call(ABC):

    @property
    @abstractmethod
    def op(self) -> Operation:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def inputs(self) -> TDict[str, ValueRef]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def outputs(self) -> TDict[str, ValueRef]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def metadata(self) -> TDict[str, TAny]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def uid(self) -> str:
        # should raise when requested but no UID is present
        raise NotImplementedError()
    
    ### 
    @staticmethod
    @abstractmethod
    def from_execution_data(op:Operation, 
                            inputs:TDict[str, ValueRef],
                            outputs:TDict[str, ValueRef],
                            metadata:TAny) -> 'Call':
        raise NotImplementedError()

    def get_relation(self) -> pd.DataFrame:
        input_uids = {k: v.uid for k, v in self.inputs.items()}
        output_uids = {k: v.uid for k, v in self.outputs.items()}
        df = self.op.get_relation(call_uid=self.uid, input_uids=input_uids,
                                    output_uids=output_uids, metadata=self.metadata)
        return df
        
    @abstractmethod
    def detached(self) -> 'Call':
        raise NotImplementedError()
    
    ############################################################################ 
    ### experimental interfaces
    ############################################################################ 
    @property
    @abstractmethod
    def exec_interval(self) -> TOption[TTuple[float, float]]:
        raise NotImplementedError()
    
    @exec_interval.setter
    @abstractmethod
    def exec_interval(self, interval:TTuple[float, float]):
        raise NotImplementedError()

    ############################################################################ 
    ### 
    ############################################################################ 
    def __eq__(self, other) -> bool:
        if not isinstance(other, Call):
            return False
        return (self.op == other.op
                and self.uid == other.uid
                and self.metadata == other.metadata
                and self.inputs == other.inputs
                and self.outputs == other.outputs)

    def __repr__(self) -> str:
        data = {
            'op': self.op,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'metadata': self.metadata
        }
        data_str = ', '.join([f'{k}={v}' for k, v in data.items()])
        return f'Call({data_str})'


################################################################################
### context
################################################################################
class BaseContext(ABC):
    # todo - should eventually get rid of this class
    @property
    @abstractmethod
    def storage(self) -> TAny:
        raise NotImplementedError()
    
    @abstractmethod
    def _descend(self, **updates):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def attach_query(self, query_obj:TAny):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def disable_ray(self) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class GlobalContext(object):
    current:TOption[BaseContext] = None

    @staticmethod
    def exists() -> bool:
        return GlobalContext.current is not None
    
    @staticmethod
    def get(fallback:BaseContext=None) -> BaseContext:
        if (not GlobalContext.exists()) and (fallback is None):
            raise ValueError()
        if GlobalContext.current is None:
            assert fallback is not None
            return fallback
        else:
            assert GlobalContext.current is not None
            return GlobalContext.current
    
    @staticmethod
    def set(context:BaseContext):
        GlobalContext.current = context

    @staticmethod
    def reset():
        GlobalContext.current = None


################################################################################
### helpers
################################################################################
def contains_any_vref(obj:TAny) -> bool:
    if isinstance(obj, ValueRef):
        return True
    if isinstance(obj, (list, tuple)):
        return any(contains_any_vref(elt) for elt in obj)
    elif isinstance(obj, dict):
        return any(contains_any_vref(obj=value) for value in obj.values())
    elif isinstance(obj, np.ndarray):
        return any(contains_any_vref(elt) for elt in obj)
    elif isinstance(obj, pd.Series):
        return any(contains_any_vref(elt) for elt in obj.values)
    elif isinstance(obj, pd.DataFrame):
        return any(contains_any_vref(x) for x in obj.values)
    else:
        return False
            
def is_deeply_persistable(vref:ValueRef) -> bool:
    if vref.is_compound():
        _, constituents = vref.get_residue()
        return all(is_deeply_persistable(x) for x in get_collection_atoms(collection=constituents))
    else:
        return vref.is_persistable
    
def is_deeply_in_memory(vref:ValueRef) -> bool:
    if vref.is_compound():
        if not vref.in_memory:
            return False
        _, constituents = vref.get_residue()
        return all(is_deeply_in_memory(x) for x in get_collection_atoms(collection=constituents))
    else:
        return vref.in_memory
    
def get_vrefs_from_calls(calls:TList[Call], 
                         drop_duplicates:bool=True) -> TList[ValueRef]:
    all_vrefs = []
    for call in calls:
        all_vrefs += list(itertools.chain(call.inputs.values(), call.outputs.values()))
    # drop duplicates based on UID
    if drop_duplicates:
        uniques_dict = {}
        for vref in all_vrefs:
            vref:ValueRef
            if vref.uid not in uniques_dict:
                uniques_dict[vref.uid] = vref
        return list(uniques_dict.values())
    else:
        return all_vrefs
    
def is_instance(vref:ValueRef, tp:Type) -> bool:
    return is_subtype(s=vref.get_type(), t=tp)
    
def is_member_of_tp(vref:ValueRef, tp:Type) -> bool:
    """
    Check whether the type of this vref is an *exact* member of the given
    reference type.
    """
    return is_member(s=vref.get_type(), t=tp)

def unwrap(obj:TAny, recursive:bool=True) -> TAny:
    if isinstance(obj, ValueRef):
        return obj.unwrap()
    else:
        if not recursive:
            return obj
        else:
            if isinstance(obj, list):
                return [unwrap(elt, recursive=recursive) for elt in obj]
            elif isinstance(obj, tuple):
                return tuple([unwrap(elt, recursive=recursive) for elt in obj])
            elif isinstance(obj, dict):
                return {k: unwrap(v, recursive=recursive) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                if obj.dtype == np.dtype('object'):
                    vfunc = np.vectorize(lambda x: unwrap(x, recursive=recursive))
                    return vfunc(obj)
                else:
                    return obj
            elif isinstance(obj, pd.Series):
                index = obj.index.map(lambda x: unwrap(x, recursive=recursive))
                data = unwrap(obj.values)
                return pd.Series(data=data, index=index)
            elif isinstance(obj, pd.DataFrame):
                index = obj.index.map(lambda x: unwrap(x, recursive=recursive))
                columns = obj.columns.map(lambda x: unwrap(x, recursive=recursive))
                data = unwrap(obj.values, recursive=recursive)
                return pd.DataFrame(data=data, columns=columns, index=index)
            else:
                return obj

def detached(obj:TAny, recursive:bool=True) -> TAny:
    if isinstance(obj, ValueRef):
        return obj.detached()
    else:
        if not recursive:
            return obj
        else:
            if isinstance(obj, list):
                return [detached(elt, recursive=recursive) for elt in obj]
            elif isinstance(obj, tuple):
                return tuple([detached(elt, recursive=recursive) for elt in obj])
            elif isinstance(obj, dict):
                return {k: detached(v, recursive=recursive) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                vfunc = np.vectorize(lambda x: detached(x, recursive=recursive))
                return vfunc(obj)
            elif isinstance(obj, pd.Series):
                index = obj.index.map(lambda x: detached(x, recursive=recursive))
                data = detached(obj.values)
                return pd.Series(data=data, index=index)
            elif isinstance(obj, pd.DataFrame):
                index = obj.index.map(lambda x: detached(x, recursive=recursive))
                columns = obj.columns.map(lambda x: detached(x, recursive=recursive))
                data = detached(obj.values, recursive=recursive)
                return pd.DataFrame(data=data, columns=columns, index=index)
            else:
                return obj

def overwrite_vref(vref:ValueRef, overwrite_from:ValueRef, by_reference:bool=True):
    """
    In-place update the state of a vref using the data of another passed by
    reference. 
    """
    if not by_reference:
        raise NotImplementedError()
    assert vref.get_type() == overwrite_from.get_type()
    assert type(vref) is type(overwrite_from)
    vref._set_uid(uid=overwrite_from.uid)
    vref.set_in_memory(value=overwrite_from.in_memory)
    vref.set_is_persistable(value=overwrite_from.is_persistable)
    # note: works regardless of whether they are in memory or not 
    vref._obj = overwrite_from._obj
    
def set_uid_pure(vref:ValueRef, new_uid:str) -> ValueRef:
    res = vref.detached()
    if vref.in_memory:
        res.attach(obj=vref.obj())
    res._set_uid(uid=new_uid)
    return res

def get_content_hash_with_type(raw_obj:TAny, tp:Type) -> str:
    """
    Combine content hash with object type to prevent UID collisions
    """
    content_hash = ContentHashing.DEFAULT(obj=raw_obj)
    type_hash = ContentHashing.DEFAULT(obj=json.dumps(tp.dump()))
    obj_identity = [content_hash, type_hash]
    uid = CausalHashing.combine_list(elt_hashes=obj_identity)
    return uid

def group_calls_by_op(calls:TList[Call], by:str='op') -> TDict[TAny, TList[Call]]:
    """
    Group calls in a dictionary keyed by their ops. 
    
    NOTE: Ops are hashed by name and version
    """
    if by == 'ui_name':
        labels = [call.op.ui_name for call in calls]
    elif by == 'name':
        #! to be backward compatible - remove at some point
        labels = [call.op.name if not call.op.is_builtin else call.op.ui_name for call in calls]
    elif by == 'op':
        labels = [call.op for call in calls]
    else:
        raise NotImplementedError()
    groups = group_like(objs=calls, labels=labels)
    return groups

def summarize_calls(calls:TList[Call], internal_to_ui:TDict[str, str]) -> pd.DataFrame:
    """
    Return a table with cols (op, num calls)
    """
    groups = group_calls_by_op(calls=calls, by='name')
    counts_by_op_name = {internal_to_ui.get(k, k): len(v) for k, v in groups.items()}
    opsort = sorted(counts_by_op_name)
    return pd.DataFrame({
        'Operation': opsort,
        'Number of calls': [counts_by_op_name[k] for k in sorted(opsort)]
    })
    
def summarize_vref_locs(locs:TList[BaseObjLocation]) -> pd.DataFrame:
    """
    Return a table with cols (partition, num vrefs)
    """
    vrefs_by_partition = group_like(objs=locs, labels=[loc.partition for loc in locs])
    partition_sort = sorted(vrefs_by_partition)
    return pd.DataFrame({
        'Partition': partition_sort,
        'Number of vrefs': [len(vrefs_by_partition[k]) for k in partition_sort]
    })
    