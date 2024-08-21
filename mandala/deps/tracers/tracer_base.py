from ...common_imports import *
from ...config import Config
import importlib
from ..model import DependencyGraph, CallableNode
from abc import ABC, abstractmethod


class TracerABC(ABC):
    def __init__(
        self,
        paths: List[Path],
        strict: bool = True,
        allow_methods: bool = False,
        track_globals: bool = True,
    ):
        self.call_stack: List[Optional[CallableNode]] = []
        self.graph = DependencyGraph()
        self.paths = paths
        self.strict = strict
        self.allow_methods = allow_methods
        self.track_globals = track_globals

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_active_trace_obj() -> Optional[Any]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_active_trace_obj(trace_obj: Any):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def register_leaf_event(trace_obj: Any, data: Any):
        raise NotImplementedError


BREAK = "break"  # stop tracing (currently doesn't really work b/c python)
CONTINUE = "continue"  # continue tracing, but don't add call to dependencies
KEEP = "keep"  # continue tracing and add call to dependencies
MAIN = "__main__"


def get_closure_names(code_obj: types.CodeType, func_qualname: str) -> Tuple[str]:
    closure_vars = code_obj.co_freevars
    if "." in func_qualname and "__class__" in closure_vars:
        closure_vars = tuple([var for var in closure_vars if var != "__class__"])
    return closure_vars


def get_module_flow(module_name: Optional[str], paths: List[Path]) -> str:
    if module_name is None:
        return BREAK
    if module_name == MAIN:
        return KEEP
    try:
        module = importlib.import_module(module_name)
        is_importable = True
    except ModuleNotFoundError:
        is_importable = False
    if not is_importable:
        return BREAK
    try:
        module_path = Path(inspect.getfile(module))
    except TypeError:
        # this happens when the module is a built-in module
        return BREAK
    if (
        not any(root in module_path.parents for root in paths)
        and module_path not in paths
    ):
        # module is not in the paths we're inspecting; stop tracing
        logger.debug(f"    Module {module_name} not in paths, BREAK")
        return BREAK
    elif module_name.startswith(Config.module_name) and not module_name.startswith(
        Config.tests_module_name
    ):
        # this function is part of `mandala` functionality. Continue tracing
        # but don't add it to the dependency state
        logger.debug(f"    Module {module_name} is mandala, CONTINUE")
        return CONTINUE
    else:
        logger.debug(f"    Module {module_name} is not mandala but in paths, KEEP")
        return KEEP
