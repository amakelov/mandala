from ..common_imports import *
from ..core.model import FuncOp
from ..core.weaver import ValQuery
from .main import FuncInterface, AsyncioFuncInterface, OnChange


def Q() -> ValQuery:
    """
    Create a `ValQuery` instance to be used as a placeholder in a query
    """
    return ValQuery(creator=None, created_as=None)


class FuncDecorator:
    # parametrized version of `@op` decorator
    def __init__(
        self,
        version: int = 0,
        ui_name: Optional[str] = None,
        executor: str = "python",
        on_change: Optional[str] = None,
        autoversion: bool = False,
    ):
        self.version = version
        self.ui_name = ui_name
        self.executor = executor
        self.on_change = on_change
        # to avoid confusion,
        # autoversion is always True when dependency changes trigger a new version
        self.autoversion = (
            True if self.on_change == OnChange.new_version else autoversion
        )

    def __call__(self, func: Callable) -> "func":
        func_op = FuncOp(func=func, version=self.version, ui_name=self.ui_name)
        if inspect.iscoroutinefunction(func):
            InterfaceCls = AsyncioFuncInterface
        else:
            InterfaceCls = FuncInterface
        return InterfaceCls(
            func_op=func_op,
            executor=self.executor,
            on_change=self.on_change,
            autoversion=self.autoversion,
        )


def op(
    version_or_func: Union[Callable, Optional[int]] = None,
    ui_name: Optional[str] = None,
    on_change: Optional[str] = None,
    executor: str = "python",
    autoversion: bool = False,
) -> Callable:
    if callable(version_or_func):
        # a hack to handle the @op case
        func = version_or_func
        func_op = FuncOp(func=func)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        if autoversion and version_or_func is not None:
            raise ValueError(
                "Setting the version explicitly to a function decorated with `autoversion=True` is not allowed."
            )
        version_or_func = 0 if version_or_func is None else version_or_func
        return FuncDecorator(
            version=version_or_func,
            ui_name=ui_name,
            executor=executor,
            on_change=on_change,
            autoversion=autoversion,
        )
