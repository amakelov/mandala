from functools import wraps
from ..common_imports import *
from ..core.model import FuncOp, TransientObj, Call
from ..core.utils import unwrap_decorators
from ..core.sig import Signature, _postprocess_outputs
from ..core.tps import AnyType
from ..queries.weaver import ValNode, qwrap, call_query
from ..deps.tracers.dec_impl import DecTracer

from . import contexts
from .utils import bind_inputs, format_as_outputs, MODES, wrap_atom


def Q(pattern: Optional[Any] = None) -> "pattern":
    """
    Create a `ValQuery` instance to be used as a placeholder in a query
    """
    if pattern is None:
        # return ValQuery(creator=None, created_as=None)
        return ValNode(creators=[], created_as=[], constraint=None, tp=AnyType())
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
        # synchronize if necessary
        storage.synchronize(self)
        # synchronize(func=self, storage=context.storage)
        inputs = bind_inputs(args, kwargs, mode=context.mode, func_op=self.func_op)
        mode = context.mode
        return inputs, mode, storage, context

    def call(self, args, kwargs) -> Tuple[Union[None, Any, Tuple[Any]], Optional[Call]]:
        # low-level API for more control over internal machinery
        r = runner.Runner(context=contexts.GlobalContext.current, func_op=self.func_op)
        # inputs, mode, storage, context = self._preprocess_call(*args,
        # **kwargs)
        if r.storage is not None:
            r.storage.synchronize(self)
        if r.mode == MODES.run:
            func_op = r.func_op
            r.preprocess(args, kwargs)
            if r.must_execute:
                r.pre_execute(conn=None)
                if r.tracer_option is not None:
                    tracer = r.tracer_option
                    with tracer:
                        if isinstance(tracer, DecTracer):
                            node = tracer.register_call(func=func_op.func)
                        result = func_op.func(**r.func_inputs)
                        if isinstance(tracer, DecTracer):
                            tracer.register_return(node=node)
                else:
                    result = func_op.func(**r.func_inputs)
                outputs = _postprocess_outputs(sig=func_op.sig, result=result)
                call = r.post_execute(outputs=outputs)
            else:
                call = r.load_call(conn=None)
            return r.postprocess(call=call), call
        return r.process_other_modes(args, kwargs), None

    def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        return self.call(args, kwargs)[0]


class AsyncFuncInterface(FuncInterface):
    async def call(
        self, args, kwargs
    ) -> Tuple[Union[None, Any, Tuple[Any]], Optional[Call]]:
        # low-level API for more control over internal machinery
        r = runner.Runner(context=contexts.GlobalContext.current, func_op=self.func_op)
        # inputs, mode, storage, context = self._preprocess_call(*args,
        # **kwargs)
        if r.storage is not None:
            r.storage.synchronize(self)
        if r.mode == MODES.run:
            func_op = r.func_op
            r.preprocess(args, kwargs)
            if r.must_execute:
                r.pre_execute(conn=None)
                if r.tracer_option is not None:
                    tracer = r.tracer_option
                    with tracer:
                        if isinstance(tracer, DecTracer):
                            node = tracer.register_call(func=func_op.func)
                        result = await func_op.func(**r.func_inputs)
                        if isinstance(tracer, DecTracer):
                            tracer.register_return(node=node)
                else:
                    result = await func_op.func(**r.func_inputs)
                outputs = _postprocess_outputs(sig=func_op.sig, result=result)
                call = r.post_execute(outputs=outputs)
            else:
                call = r.load_call(conn=None)
            return r.postprocess(call=call), call
        return r.process_other_modes(args, kwargs), None

    async def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        return (await self.call(args, kwargs))[0]


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
            InterfaceCls = AsyncFuncInterface
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
            return wraps(func)(AsyncFuncInterface(func_op=func_op))
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
            return AsyncFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        return FuncDecorator(
            version=version, ui_name=ui_name, executor=executor, is_super=True
        )


from . import storage
from . import runner
