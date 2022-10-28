from ..common_imports import *
from ..core.model import FuncOp
from ..core.weaver import ValQuery
from .main import FuncInterface


def Q() -> ValQuery:
    """
    Create a `ValQuery` instance to be used as a placeholder in a query
    """
    return ValQuery(creator=None, created_as=None)


class FuncDecorator:
    # parametrized version of `@op` decorator
    def __init__(self, version: int = 0, ui_name: Optional[str] = None):
        self.version = version
        self.ui_name = ui_name

    def __call__(self, func: Callable) -> "func":
        func_op = FuncOp(func=func, version=self.version, ui_name=self.ui_name)
        return FuncInterface(func_op=func_op)


def op(*args, **kwargs) -> Callable:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # @op case
        func_op = FuncOp(func=args[0])
        return FuncInterface(func_op=func_op)
    else:
        # @op(...) case
        version = kwargs.get("version", 0)
        ui_name = kwargs.get("ui_name", None)
        return FuncDecorator(version=version, ui_name=ui_name)
