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
        version: Optional[int] = None,
        ui_name: Optional[str] = None,
        executor: str = "python",
        on_change: Optional[str] = None,
    ):
        self.version = version
        self.ui_name = ui_name
        self.executor = executor
        self.on_change = on_change
        if on_change is not None and version is not None:
            # on_change manages the version through a separate mechanism
            raise ValueError(
                "Setting the version explicitly to a function decorated with `on_change` is not allowed."
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
        )


def op(
    version: Union[Callable, Optional[int]] = None,
    ui_name: Optional[str] = None,
    on_change: Optional[str] = None,
    executor: str = "python",
) -> Callable:
    if callable(version):
        # a hack to handle the @op case
        func = version
        func_op = FuncOp(func=func)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        return FuncDecorator(
            version=version,
            ui_name=ui_name,
            executor=executor,
            on_change=on_change,
        )
