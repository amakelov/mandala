from .context import GlobalContext
from .context import Context
from .storage import Storage, FuncUIBase
from .execution import CONTEXT_KW, OpCaller

from ..common_imports import *
from ..core.bases import Operation
from ..core.config import EnvConfig, MODES
if EnvConfig.has_ray:
    import ray
from ..core.impl import SimpleFunc
from ..core.exceptions import SynchronizationError
from ..storages.calls import CallLocation

RETURNS_KW = '__returns__'
RAY_OP_KW = '__rayop__'
RAY_CONTEXT_KW = '__raycontext__'

################################################################################
### function interface 
################################################################################
class FuncOpUI(FuncUIBase):
    InvalidatedOpSentinel = object()
    RETURNS_KW = '__returns__'
    
    def __init__(self, func:TCallable, n_out:int=None, 
                 version:TUnion[str, int]=None,
                 output_names:TList[str]=None, is_super:bool=False,
                 unwrap_inputs:bool=True, storage:Storage=None, 
                 ray_kwargs:TDict[str, TAny]=None, 
                 var_outputs:bool=False, mutations:TDict[str, int]=None):
        if n_out is not None and output_names is None:
            output_names = [f'output_{i}' for i in range(n_out)]
        version = str(version) if version is not None else version
        self._op = SimpleFunc(func=func, output_names=output_names, 
                              version=version,
                              is_super=is_super, 
                              unwrap_inputs=unwrap_inputs,
                              var_outputs=var_outputs, mutations=mutations)
        self._storage = None

        if GlobalContext.exists():
            global_storage = GlobalContext.get().storage
        else:
            global_storage = None
        storage = global_storage if storage is None else storage
        if storage is not None:
            if GlobalContext.exists():
                c = GlobalContext.get()
                if c.mode == MODES.define:
                    c._funcops.append(self)
                else:
                    self.synchronize(storage=storage)
            else:
                self.synchronize(storage=storage)
        
        if EnvConfig.has_ray:
            self._snapshot = self.get_callable_snapshot()
            ray_kwargs = {} if ray_kwargs is None else ray_kwargs
            if not ray_kwargs:
                self._remote_snapshot = ray.remote(self._snapshot)
            else:
                self._remote_snapshot = ray.remote(**ray_kwargs)(self._snapshot)
    
    @property
    def op(self) -> SimpleFunc:
        if self._op is self.InvalidatedOpSentinel:
            raise SynchronizationError(f'This operation has been invalidated, '
                                       f'reason:\n{self._invalidation_reason}')
        assert isinstance(self._op, SimpleFunc)
        return self._op
    
    def set_op(self, op:SimpleFunc):
        self._op = op

    @property
    def storage(self) -> TOption[Storage]:
        return self._storage
    
    def set_storage(self, storage: Storage):
        self._storage = storage

    def synchronize(self, storage:Storage):
        storage.synchronize(funcop=self)
    
    def invalidate(self, reason:str):
        self._invalidation_ui_name = self.op.ui_name
        self._op = self.InvalidatedOpSentinel
        self._invalidation_reason = reason
    
    @property
    def is_invalidated(self) -> bool:
        return self._op is self.InvalidatedOpSentinel
    
    ############################################################################ 
    def get_table(self, as_vrefs:bool=False) -> pd.DataFrame:
        if as_vrefs:
            return self.storage.rel_adapter.get_op_vrefs(
                op=self.op,
                rename=True,
                include_builtin_cols=False
            )
        else:
            return self.storage.rel_adapter.get_op_values(
                op=self.op,
                rename=True,
                include_builtin_cols=False
            )
    
    ############################################################################ 
    ### call-like interfaces
    ############################################################################ 
    def __call__(self, *args, **kwargs):
        """
        Run this op in any possible mode
        """
        if GlobalContext.exists():
            c = GlobalContext.get()
            if c.storage is not None:
                storage:Storage = c.storage
                if not storage.synchronizer.is_connected(f=self.op):
                    raise SynchronizationError(f'The operation {repr(self)} '
                        f'is not connected to the storage {c.storage}')
        else:
            c = None
        op_caller = OpCaller(op=self.op, c=c)
        return op_caller.run(args=args, kwargs=kwargs)
    
    def is_recoverable(self, *args, **kwargs) -> bool:
        if OpCaller.CONTEXT_KW in kwargs:
            c = kwargs.pop(OpCaller.CONTEXT_KW)
        else:
            c = GlobalContext.get()
        op_caller = OpCaller(op=self.op, c=c)
        return op_caller.run_is_recoverable(args=args, kwargs=kwargs)

    def mrecoverable(self, *marg_cols, **mkwarg_cols) -> TList[bool]:
        if OpCaller.CONTEXT_KW in mkwarg_cols:
            c = mkwarg_cols.pop(OpCaller.CONTEXT_KW)
        else:
            c = GlobalContext.get()
        op_caller = OpCaller(op=self.op, c=c)
        return op_caller.mrun_is_recoverable(marg_cols=marg_cols, 
                                             mkwarg_cols=mkwarg_cols, c=c)
    
    def locate(self, *args, **kwargs) -> CallLocation:
        c = GlobalContext.get(fallback=None)
        op_caller = OpCaller(op=self.op, c=c)
        return op_caller.locate(args=args, kwargs=kwargs)
    
    def mcall(self, *marg_cols, **mkwarg_cols):
        if OpCaller.CONTEXT_KW in mkwarg_cols:
            c = mkwarg_cols.pop(OpCaller.CONTEXT_KW)
        else:
            c = GlobalContext.get()
        op_caller = OpCaller(op=self.op, c=c)
        # unzip
        return op_caller.mrun(marg_cols=marg_cols, mkwarg_cols=mkwarg_cols, c=c)
    
    ############################################################################ 
    ### parallelization: ray
    ############################################################################ 
    def set_ray_kwargs(self, kwargs:TDict[str, TAny]=None):
        assert EnvConfig.has_ray
        kwargs = {} if kwargs is None else kwargs
        if not kwargs:
            self._remote_snapshot = ray.remote(self._snapshot)
        else:
            self._remote_snapshot = ray.remote(**kwargs)(self._snapshot)
            
    def get_callable_snapshot(self) -> TCallable:
        def f(*args, **kwargs):
            op = kwargs.pop(RAY_OP_KW)
            c:Context = kwargs.pop(RAY_CONTEXT_KW)
            if CONTEXT_KW in kwargs:
                # override
                c = None
            op_caller = OpCaller(op=op, c=c)
            return op_caller.run(args=args, kwargs=kwargs)
        return f
    
    def remote(self, *args, **kwargs):
        if GlobalContext.exists() and GlobalContext.get().disable_ray:
            return self.__call__(*args, **kwargs)
        else:
            assert RAY_OP_KW not in kwargs
            kwargs[RAY_OP_KW] = self.op
            assert RAY_CONTEXT_KW not in kwargs
            kwargs[RAY_CONTEXT_KW] = GlobalContext.get()
            return self._remote_snapshot.remote(*args, **kwargs)
    
    def __repr__(self) -> str:
        if self.is_invalidated:
            lines = [
                f'INVALIDATED copy of operation {self._invalidation_ui_name}',
                f'Reason:',
                self._invalidation_reason
            ]
            return '\n'.join(lines)
        else:
            return f'FuncOp(name={self.op.ui_name}, \
                version={self.op.version}, \
                    signature={repr(inspect.signature(self.op.func))})'
    

class FuncOpDecorator(object):
    IS_SUPER_DEFAULT = False
    
    def __init__(self, storage:Storage=None, version:TUnion[str, int]=None, 
                 output_names:TList[str]=None, n_out:int=None,
                 is_super:bool=None, unwrap_inputs:bool=True,
                 ray_kwargs:TDict[str, TAny]=None, 
                 var_outputs:bool=False, mutations:TDict[str, int]=None):
        if is_super is None:
            is_super = self.IS_SUPER_DEFAULT
        self.version = version
        self.output_names = output_names
        self.n_out = n_out
        self.storage = storage
        self.is_super = is_super
        self.unwrap_inputs = unwrap_inputs
        self.ray_kwargs =ray_kwargs
        self.var_outputs = var_outputs
        if mutations is not None and self.is_super:
            raise ValueError()
        self.mutations = mutations
    
    def __call__(self, func:TCallable) -> 'func':
        if self.var_outputs:
            logging.warning('Using functions with varying number of outputs '
                'is discouraged')
        return FuncOpUI(func=func, version=self.version, n_out=self.n_out,
                        is_super=self.is_super,
                        unwrap_inputs=self.unwrap_inputs,
                        output_names=self.output_names,
                        storage=self.storage, 
                        ray_kwargs=self.ray_kwargs, 
                        var_outputs=self.var_outputs, 
                        mutations=self.mutations)
        

class FuncSuperOpDecorator(FuncOpDecorator):
    IS_SUPER_DEFAULT = True

    
op = FuncOpDecorator
superop = FuncSuperOpDecorator