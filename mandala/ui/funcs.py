from ..common_imports import *
from ..core.model import FuncOp
from ..core.weaver import ValQuery, BuiltinQueries, qwrap
from .main import FuncInterface, AsyncioFuncInterface, OnChange


def Q(pattern: Optional[Any] = None) -> "pattern":
    """
    Create a `ValQuery` instance to be used as a placeholder in a query
    """
    if pattern is None:
        return ValQuery(creator=None, created_as=None)
    else:
        return qwrap(obj=pattern)


class FuncDecorator:
    # parametrized version of `@op` decorator
    def __init__(
        self,
        **kwargs,
    ):
        self.kwargs = kwargs

    def __call__(self, func: Callable) -> "func":
        func_op = FuncOp(
            func=func,
            n_outputs=self.kwargs.get("n_outputs"),
            version=self.kwargs.get("version"),
            ui_name=self.kwargs.get("ui_name"),
            is_super=self.kwargs.get("is_super", False),
        )
        if inspect.iscoroutinefunction(func):
            InterfaceCls = AsyncioFuncInterface
        else:
            InterfaceCls = FuncInterface
        return InterfaceCls(
            func_op=func_op,
            executor=self.kwargs.get("executor", "python"),
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
        func_op = FuncOp(func=func, n_outputs=nout)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
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
