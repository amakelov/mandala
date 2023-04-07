from functools import wraps
from ..common_imports import *
from ..core.model import FuncOp, TransientObj, Call
from ..core.utils import unwrap_decorators
from ..core.sig import Signature
from ..core.tps import AnyType
from ..queries.weaver import ValQuery, qwrap, call_query
from ..deps.tracers.dec_impl import DecTracer

from . import contexts
from .utils import wrap_inputs, bind_inputs, format_as_outputs, MODES


def Q(pattern: Optional[Any] = None) -> "pattern":
    """
    Create a `ValQuery` instance to be used as a placeholder in a query
    """
    if pattern is None:
        # return ValQuery(creator=None, created_as=None)
        return ValQuery(creators=[], created_as=[], constraint=None, tp=AnyType())
    else:
        return qwrap(obj=pattern)


T = TypeVar("T")


def Transient(obj: T, unhashable: bool = False) -> T:
    if contexts.GlobalContext.current is not None:
        return TransientObj(obj=obj, unhashable=unhashable)
    else:
        return obj


class FuncInterface:
    """
    Wrapper around a memoized function.

    This is the object the `@op` decorator converts functions into.
    """

    def __init__(
        self,
        func_op: FuncOp,
        executor: str = "python",
    ):
        self.func_op = func_op
        self.__name__ = self.func_op.sig.ui_name
        self._is_synchronized = False
        self._is_invalidated = False
        self._storage_id = None
        self.executor = executor
        if (
            contexts.GlobalContext.current is not None
            and contexts.GlobalContext.current.mode == MODES.define
        ):
            contexts.GlobalContext.current._defined_funcs.append(self)

    @property
    def sig(self) -> Signature:
        return self.func_op.sig

    def __repr__(self) -> str:
        sig = self.func_op.sig
        if self._is_invalidated:
            # clearly distinguish stale functions
            return f"FuncInterface(func_name={sig.ui_name}, is_invalidated=True)"
        else:
            from rich.text import Text

            return f"FuncInterface(signature={sig})"

    def invalidate(self):
        self._is_invalidated = True
        self._is_synchronized = False

    @property
    def is_invalidated(self) -> bool:
        return self._is_invalidated

    def _preprocess_call(
        self, *args, **kwargs
    ) -> Tuple[Dict[str, Any], str, "storage.Storage", contexts.Context]:
        context = contexts.GlobalContext.current
        storage = context.storage
        if self._is_invalidated:
            raise RuntimeError(
                "This function has been invalidated due to a change in the signature, and cannot be called"
            )
        # if not self.func_op.sig.has_internal_data:
        # synchronize if necessary
        storage.synchronize(self)
        # synchronize(func=self, storage=context.storage)
        inputs = bind_inputs(args, kwargs, mode=context.mode, func_op=self.func_op)
        mode = context.mode
        return inputs, mode, storage, context

    def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        logger.debug(f"Calling {self.func_op.sig.ui_name}")
        context = contexts.GlobalContext.current
        if context is None:
            # mandala is completely disabled when not in a context
            return self.func_op.func(*args, **kwargs)
        inputs, mode, storage, context = self._preprocess_call(*args, **kwargs)
        if mode == MODES.run:
            if self.executor == "python":
                context._call_depth += 1
                outputs, call, wrapped_inputs = storage.call_run(
                    func_op=self.func_op,
                    inputs=inputs,
                    allow_calls=context.allow_calls,
                    debug_calls=context.debug_calls,
                    recompute_transient=context.recompute_transient,
                    lazy=context.lazy,
                    _call_depth=context._call_depth,
                    _code_state=context._code_state,
                    _versioner=context._cached_versioner,
                )
                sig = self.func_op.sig
                context._call_uids[(sig.internal_name, sig.version)].append(call.uid)
                context._call_depth -= 1
                if context._attach_call_to_outputs:
                    for output in outputs:
                        output._call = call.detached()

                return format_as_outputs(outputs=outputs)
            # elif self.executor == 'asyncio' or inspect.iscoroutinefunction(self.func_op.func):
            elif self.executor == "dask":
                assert (
                    not storage.rel_storage.in_memory
                ), "Dask executor only works with a persistent storage"

                def daskop_f(*args, __data__, **kwargs):
                    call_cache, obj_cache, db_path = __data__
                    temp_storage = Storage(db_path=db_path, _read_only=True)
                    temp_storage.call_cache = call_cache
                    temp_storage.obj_cache = obj_cache
                    inputs = bind_inputs(
                        func_op=self.func_op, args=args, kwargs=kwargs, mode=MODES.run
                    )
                    outputs, _ = temp_storage.call_run(
                        func_op=self.func_op, inputs=inputs, mode=mode
                    )
                    return format_as_outputs(outputs=outputs)

                __data__ = (
                    storage.call_cache_by_causal,
                    storage.obj_cache,
                    storage.db_path,
                )
                nout = self.func_op.sig.n_outputs
                return delayed(daskop_f, nout=nout)(*args, __data__=__data__, **kwargs)
            else:
                raise NotImplementedError()
        elif mode == MODES.query:
            return format_as_outputs(
                outputs=call_query(func_op=self.func_op, inputs=inputs)
            )
        elif mode == MODES.batch:
            assert self.executor == "python"
            wrapped_inputs = wrap_inputs(inputs)
            outputs, call_struct = storage.call_batch(
                func_op=self.func_op, inputs=wrapped_inputs
            )
            context._call_structs.append(call_struct)
            return format_as_outputs(outputs=outputs)
        else:
            raise ValueError()


class AsyncioFuncInterface(FuncInterface):
    async def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        context = contexts.GlobalContext.current
        if context is None:
            # mandala is completely disabled when not in a context
            return self.func_op.func(*args, **kwargs)
        inputs, mode, storage, context = self._preprocess_call(*args, **kwargs)
        if mode == MODES.run:

            async def async_f(*args, __data__, **kwargs):
                call_cache, obj_cache, db_path = __data__
                temp_storage = Storage(db_path=db_path, _read_only=True)
                temp_storage.call_cache = call_cache
                temp_storage.obj_cache = obj_cache
                inputs = bind_inputs(
                    func_op=self.func_op, args=args, kwargs=kwargs, mode=MODES.run
                )
                outputs, _ = await temp_storage.call_run_async(
                    func_op=self.func_op, inputs=inputs
                )
                return format_as_outputs(outputs=outputs)

            __data__ = (
                storage.call_cache_by_causal,
                storage.obj_cache,
                storage.db_path,
            )
            return await async_f(*args, __data__=__data__, **kwargs)
        else:
            return super().__call__(*args, **kwargs)


class FuncDecorator:
    # parametrized version of `@op` decorator
    def __init__(
        self,
        **kwargs,
    ):
        self.kwargs = kwargs

    def __call__(self, func: Callable) -> "func":
        # func = unwrap_decorators(func, strict=True)
        func_op = FuncOp(
            func=func,
            n_outputs_override=self.kwargs.get("n_outputs"),
            version=self.kwargs.get("version"),
            ui_name=self.kwargs.get("ui_name"),
            is_super=self.kwargs.get("is_super", False),
        )
        if inspect.iscoroutinefunction(func):
            InterfaceCls = AsyncioFuncInterface
        else:
            InterfaceCls = FuncInterface
        return wraps(func)(
            InterfaceCls(
                func_op=func_op,
                executor=self.kwargs.get("executor", "python"),
            )
        )


def op(
    version: Union[Callable, Optional[int]] = None,
    nout: Optional[int] = None,
    ui_name: Optional[str] = None,
    executor: str = "python",
) -> "version":  # a hack to make mypy/autocomplete happy
    if callable(version):
        # a hack to handle the @op case
        func = version
        # func = unwrap_decorators(func, strict=True)
        func_op = FuncOp(func=func, n_outputs_override=nout)
        if inspect.iscoroutinefunction(func):
            return wraps(func)(AsyncioFuncInterface(func_op=func_op))
        else:
            return wraps(func)(FuncInterface(func_op=func_op))
    else:
        # @op(...) case
        return FuncDecorator(
            version=version,
            n_outputs=nout,
            ui_name=ui_name,
            executor=executor,
        )


def superop(
    version: Union[Callable, Optional[int]] = None,
    ui_name: Optional[str] = None,
    executor: str = "python",
) -> "version":
    if callable(version):
        # a hack to handle the @op case
        func = version
        # func_op = FuncOp(func=unwrap_decorators(func, strict=True),
        # is_super=True)
        func_op = FuncOp(func=func, is_super=True)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        return FuncDecorator(
            version=version, ui_name=ui_name, executor=executor, is_super=True
        )


from . import storage
