from ..common_imports import *
from ..core.config import Config
from ..core.model import FuncOp, ValueRef, Call, wrap
from ..core.utils import Hashing
from ..storages.main import Storage
from .context import GlobalContext, MODES


class FuncInterface:
    """
    Wrapper around a memoized function. 
    
    This is the object the `@op` decorator converts functions into.
    """
    def __init__(self, op:FuncOp, storage:Storage):
        self.op = op
        self.storage = storage

    def wrap_inputs(self, inputs:Dict[str, Any]) -> Dict[str, ValueRef]:
        # "explicit" superops must always receive wrapped inputs
        if self.op.sig.is_super:
            assert all(isinstance(v, ValueRef) for v in inputs.values())
            return inputs
        # check if we allow implicit wrapping
        if Config.autowrap_inputs:
            return {k: wrap(v) for k, v in inputs.items()}
        else:
            assert all(isinstance(v, ValueRef) for v in inputs.values())
            return inputs

    def wrap_outputs(self, outputs:List[Any], call_uid:str) -> List[ValueRef]:
        output_uids = [Hashing.get_content_hash((call_uid, i)) for i in range(len(outputs))]
        wrapped_outputs = [wrap(v, uid=u) for v, u in zip(outputs, output_uids)]
        return wrapped_outputs

    def bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        bound_args = self.op.py_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs_dict = dict(bound_args.arguments)
        # rename to internal names
        inputs_dict = {self.op.sig.input_mapping[k]: v 
                       for k, v in inputs_dict.items()}
        return inputs_dict

    def format_as_outputs(self, outputs:List[ValueRef]) -> Union[None, Any, Tuple[Any]]:
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def call_run(self, inputs:Dict[str, Union[Any, ValueRef]]) -> Tuple[List[ValueRef], Call]:
        # wrap inputs
        wrapped_inputs = self.wrap_inputs(inputs) 
        # get call UID
        call_uid = Hashing.get_content_hash(obj=[{k: v.uid for k, v in wrapped_inputs.items()},
                                             self.op.sig.internal_name])
        # check if call UID exists in call storage
        if self.storage.calls.exists(call_uid):
            # get call from call storage
            call = self.storage.calls.get(call_uid)
            # get outputs from obj storage
            wrapped_outputs = [self.storage.objs.get(v.uid) for v in call.outputs]
            # return outputs and call
            return wrapped_outputs, call 
        else:
            # compute op
            if Config.autounwrap_inputs:
                raw_inputs = {k: v.obj for k, v in wrapped_inputs.items()}
            else:
                raw_inputs = wrapped_inputs
            outputs = self.op.compute(raw_inputs)
            # wrap outputs
            wrapped_outputs = self.wrap_outputs(outputs, call_uid=call_uid)
            # create call
            call = Call(uid=call_uid, inputs=wrapped_inputs, outputs=wrapped_outputs, op=self.op)
            # set call in call storage
            self.storage.calls.temp.set(k=call_uid, v=call)
            # set outputs in obj storage
            for v in wrapped_outputs:
                self.storage.objs.set(k=v.uid, v=v)
            # return outputs and call
            return wrapped_outputs, call 

    def call_query(self, inputs:Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> List[ValueRef]:
        context = GlobalContext.current
        if context is None:
            raise RuntimeError('No context to call from')
        mode = context.mode
        if mode == MODES.run:
            inputs = self.bind_inputs(args, kwargs)
            outputs, call = self.call_run(inputs)
            return self.format_as_outputs(outputs=outputs)
        elif mode == MODES.query:
            raise NotImplementedError()
        else:
            raise ValueError()

        
class FuncDecorator:
    """
    This is the `@op` decorator internally
    """
    def __init__(self, storage:Storage, is_super:bool=False):
        self.storage = storage
        self.super = is_super
         
    def __call__(self, func) -> 'FuncInterface':
        op = FuncOp(func=func, is_super=self.super) 
        op.sig = self.storage.synchronize(sig=op.sig)
        return FuncInterface(op=op, storage=self.storage)
    

op = FuncDecorator