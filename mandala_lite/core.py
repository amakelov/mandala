from typing import Any, Dict, List, Callable, Tuple
import copy
import hashlib
import io
import os
import joblib
import inspect
import binascii
from pathlib import Path
import pandas as pd

def get_uid() -> str:
    return '{}'.format(binascii.hexlify(os.urandom(16)).decode('utf-8'))

################################################################################
class ValueRef:
    def __init__(self, uid:str, obj:Any):
        self.uid = uid
        self.obj = obj
    

class Hashing:
    @staticmethod
    def content_hash(obj:Any) -> str:
        """
        A deterministic hash function for python objects, one would hope
        """
        stream = io.BytesIO()
        joblib.dump(value=obj, filename=stream)
        stream.seek(0)
        m = hashlib.md5()
        m.update(str((stream.read())).encode())
        return m.hexdigest()


def wrap(obj:Any) -> ValueRef:
    return obj if isinstance(obj, ValueRef) else ValueRef(uid=Hashing.content_hash(obj), obj=obj)

################################################################################
class Call:
    def __init__(self, uid:str, inputs:Dict[str, ValueRef], 
                 outputs:List[ValueRef], op:'FuncOp'):
        self.uid = uid
        self.inputs = inputs
        self.outputs = outputs
        self.op = op
    

################################################################################
class FuncOp:
    def __init__(self, func:Callable, version:int=0):
        self.func = func 
        self.sig = Signature.from_py(sig=inspect.signature(func), 
                                           name=func.__name__, version=version)
        self.is_synchronized = False
    
    def compute(self, inputs:Dict[str, Any]) -> List[Any]:
        raise NotImplementedError()


class Signature:
    def __init__(self, name:str, input_names:List[str], n_outputs:int, 
                 defaults:Dict[str, Any], version:int):
        self.name = name
        self.input_names = input_names
        self.defaults = defaults
        self.n_outputs = n_outputs
        self.version = version
        self.internal_name = ''
        # internal name -> external name
        self.input_mapping = {}
    
    def set_internal(self, internal_name:str, input_mapping:dict) -> 'Signature':
        res = copy.deepcopy(self)
        res.internal_name, res.input_mapping = internal_name, input_mapping
        return res

    def generate_internal(self) -> 'Signature':
        res = copy.deepcopy(self)
        res.internal_name, res.input_mapping = get_uid(), {k: get_uid() for k in self.input_names}
        return res

    def get_diff(self, other:'Signature'): 
        if ((other.input_names == self.input_names) and
            (other.defaults == self.defaults) and
            (other.n_outputs == self.n_outputs)):
            return None
        else:
            raise NotImplementedError()

    def create_input(self, name:str) -> 'Signature':
        raise NotImplementedError()

    def rename(self, new_name:str) -> 'Signature':
        raise NotImplementedError()

    def rename_input(self, name:str, new_name:str) -> 'Signature':
        raise NotImplementedError()

    @staticmethod
    def from_py(name:str, version:int, sig:inspect.Signature) -> 'Signature':
        input_names = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        return_annotation = sig.return_annotation
        if hasattr(return_annotation, '__origin__') and return_annotation.__origin__ is tuple:
            n_outputs = len(return_annotation.__args__)
        elif return_annotation is inspect._empty:
            n_outputs = 0
        else:
            n_outputs = 1
        defaults = {param.name:param.default for param in sig.parameters.values() if param.default is not inspect.Parameter.empty}
        return Signature(name=name, input_names=input_names, n_outputs=n_outputs, defaults=defaults, version=version) 

################################################################################
class KVStorage:
    def __init__(self, root:Path):
        self.root = root

    def get_obj_path(self, k:str) -> Path:
        return self.root / f'{k}.joblib'

    def exists(self, k:str) -> bool:
        return self.get_obj_path(k).exists() 

    def set(self, k:str, v:Any):
        joblib.dump(v, self.get_obj_path(k))
    
    def get(self, k:str) -> Any:
        return joblib.load(self.get_obj_path(k))
    
    def delete(self, k:str):
        os.remove(path=self.get_obj_path(k=k))


################################################################################
class RelStorage:
    def create_relation(self, name:str, columns:List[str]):
        raise NotImplementedError()
    
    def delete_relation(self, name:str):
        raise NotImplementedError()

    def create_column(self, relation:str, name:str, default_value:str):
        raise NotImplementedError()

    def select(self, query):
        raise NotImplementedError()
    
    def insert(self, name:str, df:pd.DataFrame):
        raise NotImplementedError()

    def delete(self, name:str, index:List[str]):
        raise NotImplementedError()


class RelAdapter:
    def __init__(self, rel_storage:RelStorage):
        self.rel_storage = rel_storage 

    @staticmethod
    def tabulate_calls(calls:List[Call]) -> pd.DataFrame:
        raise NotImplementedError()


class SigStorage:
    def __init__(self):
        # (external name, version) -> signature
        self.sigs:Dict[Tuple[str, int], Signature] = {}
    
    def update_sig(self, sig:Signature) -> Signature:
        current = self.sigs[sig.name, sig.version]
        current.get_diff(other=sig)
        return current
    
    def synchronize(self, res:Signature) -> Signature:
        if (res.name, res.version) not in self.sigs:
            res = res.generate_internal()
            self.sigs[(res.name, res.version)] = res 
        else:
            res = self.update_sig(sig=res)
        return res
    
    def create_input(self, op_name:str, input_name:str) -> Signature:
        raise NotImplementedError()

################################################################################
class Storage:
    def __init__(self, root:Path):
        self.root = root
        self.call_storage = KVStorage(root=self.root / 'calls') 
        self.obj_storage = KVStorage(root=self.root / 'objs') 
        self.rel_storage = RelStorage()
        self.rel_adapter = RelAdapter(rel_storage=self.rel_storage) 
        self.sig_storage = SigStorage()


################################################################################
class FuncInterface:
    def __init__(self, op:FuncOp, storage:Storage):
        self.op = op
        self.storage = storage

    def wrap_inputs(self, inputs:Dict[str, Any]) -> Dict[str, ValueRef]:
        return {k: wrap(v) for k, v in inputs.items()}

    def wrap_outputs(self, outputs:List[Any]) -> Dict[str, ValueRef]:
        raise NotImplementedError()

    def bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        bound_args = self.op.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs_dict = dict(bound_args.arguments)
        return inputs_dict

    def call_run(self, inputs:Dict[str, Any]) -> Tuple[List[ValueRef], Call]:
        # wrap inputs
        wrapped_inputs = self.wrap_inputs(inputs) 
        # get call UID
        call_uid = Hashing.content_hash(obj=[{k: v.uid for k, v in wrapped_inputs.items()},
                                             self.op.sig.internal_name])
        # check if call UID exists in call storage
        if self.storage.call_storage.exists(call_uid):
            # get call from call storage
            call = self.storage.call_storage.get(call_uid)
            # get outputs from obj storage
            outputs = {k: self.storage.obj_storage.get(v.uid) for k, v in call.outputs.items()}
            # return outputs and call
            return outputs, call 
        else:
            # compute op
            outputs = self.op.compute(inputs)
            # wrap outputs
            wrapped_outputs = self.wrap_outputs(outputs)
            # create call
            call = Call(uid=call_uid, inputs=wrapped_inputs, outputs=wrapped_outputs, op=self.op)
            # set call in call storage
            self.storage.call_storage.set(k=call_uid, v=call)
            # set outputs in obj storage
            for v in outputs:
                self.storage.obj_storage.set(v.uid, v)
            # return outputs and call
            return outputs, call 

    def call_query(self, inputs:Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> List[ValueRef]:
        # run unction
        inputs = self.bind_inputs(args=args, kwargs=kwargs)
        result, calls = self.call_run(inputs=inputs)
        return result

        
class FuncDecorator:
    def __init__(self, storage:Storage):
        self.storage = storage
         
    def __call__(self, func) -> 'FuncInterface':
        op = FuncOp(func=func) 
        op.sig = self.storage.sig_storage.synchronize(res=op.sig)
        return FuncInterface(op=op, storage=self.storage)
    
    
op = FuncDecorator