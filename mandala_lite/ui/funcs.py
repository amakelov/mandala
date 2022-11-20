from ..common_imports import *
from ..core.model import FuncOp
from ..core.weaver import ValQuery
from .main import FuncInterface, AsyncioFuncInterface


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
    ):
        self.version = version
        self.ui_name = ui_name
        self.executor = executor

    def __call__(self, func: Callable) -> "func":
        func_op = FuncOp(func=func, version=self.version, ui_name=self.ui_name)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op, executor=self.executor)
        else:
            return FuncInterface(func_op=func_op, executor=self.executor)


def op(*args, **kwargs) -> Callable:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # @op case
        func = args[0]
        func_op = FuncOp(func=func)
        if inspect.iscoroutinefunction(func):
            return AsyncioFuncInterface(func_op=func_op)
        else:
            return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        version = kwargs.get("version", 0)
        ui_name = kwargs.get("ui_name", None)
        executor = kwargs.get("executor", "python")
        return FuncDecorator(version=version, ui_name=ui_name, executor=executor)
