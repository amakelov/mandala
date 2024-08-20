import types
import dis
import importlib
import gc
from typing import Literal

from ..common_imports import *
from ..model import Ref
from ..utils import get_content_hash, unwrap_decorators
from ..config import Config

DepKey = Tuple[str, str]  # (module name, object address in module)


class GlobalClassifier:
    """
    Try to bucket Python objects into categories for the sake of tracking
    global state.
    """
    SCALARS = "scalars"
    DATA = "data"
    ALL = "all"

    @staticmethod
    def is_excluded(obj: Any) -> bool:
        return (
            inspect.ismodule(obj)  # exclude modules
            or isinstance(obj, type)  # exclude classes
            or inspect.isfunction(obj)  # exclude functions
            # or callable(obj)  # exclude callables... this is very questionable
            or type(obj).__name__ == Config.func_interface_cls_name  #! a hack to exclude memoized functions
        )

    @staticmethod
    def is_scalar(obj: Any) -> bool:
        result = isinstance(obj, (int, float, str, bool, type(None)))
        return result

    @staticmethod
    def is_data(obj: Any) -> bool:
        if GlobalClassifier.is_scalar(obj):
            result = True
        elif type(obj) in (tuple, list):
            result = all(GlobalClassifier.is_data(x) for x in obj)
        elif type(obj) is dict:
            result = all(GlobalClassifier.is_data((x, y)) for (x, y) in obj.items())
        elif type(obj) in (np.ndarray, pd.DataFrame, pd.Series, pd.Index):
            result = True
        else:
            result = False
        # if not result and not GlobalsStrictness.is_callable(obj):
        #     logger.warning(f'Access to global variable "{obj}" is not tracked because it is not a scalar or a data structure')
        return result


def is_global_val(obj: Any, allow_only: str = "all") -> bool:
    """
    Determine whether the given Python object should be treated as a global
    variable whose *content* should be tracked.

    The alternative is that this is a callable object whose *dependencies*
    should be tracked.

    However, the distinction is not always clear, making this method somewhat
    heuristic. For example, a callable object could be either a function or a
    global variable we want to track.
    """
    if isinstance(obj, Ref): # easy case; we always track globals that we explicitly wrapped
        return True
    if allow_only == GlobalClassifier.SCALARS:
        return GlobalClassifier.is_scalar(obj=obj)
    elif allow_only == GlobalClassifier.DATA:
        return GlobalClassifier.is_data(obj=obj)
    elif allow_only == GlobalClassifier.ALL:
        return not (
            inspect.ismodule(obj)  # exclude modules
            or isinstance(obj, type)  # exclude classes
            or inspect.isfunction(obj)  # exclude functions
            # or callable(obj)  # exclude callables ### this is very questionable
            or type(obj).__name__
            == Config.func_interface_cls_name  #! a hack to exclude memoized functions
        )
    else:
        raise ValueError(
            f"Unknown strictness level for tracking global variables: {allow_only}"
        )


def is_callable_obj(obj: Any, strict: bool) -> bool:
    if type(obj).__name__ == Config.func_interface_cls_name:
        return True
    if isinstance(obj, types.FunctionType):
        return True
    if not strict and callable(obj):  # quite permissive
        return True
    return False


def extract_func_obj(obj: Any, strict: bool) -> types.FunctionType:
    if type(obj).__name__ == Config.func_interface_cls_name:
        return obj.f
    obj = unwrap_decorators(obj, strict=strict)
    if isinstance(obj, types.BuiltinFunctionType):
        raise ValueError(f"Expected a non-built-in function, but got {obj}")
    if not isinstance(obj, types.FunctionType):
        if not strict:
            if (
                isinstance(obj, type)
                and hasattr(obj, "__init__")
                and isinstance(obj.__init__, types.FunctionType)
            ):
                return obj.__init__
            else:
                return unknown_function
        else:
            raise ValueError(f"Expected a function, but got {obj} of type {type(obj)}")
    return obj


def extract_code(obj: Callable) -> types.CodeType:
    if type(obj).__name__ == Config.func_interface_cls_name:
        obj = obj.f
    if isinstance(obj, property):
        obj = obj.fget
    obj = unwrap_decorators(obj, strict=True)
    if not isinstance(obj, (types.FunctionType, types.MethodType)):
        logger.debug(f"Expected a function or method, but got {type(obj)}")
        # raise ValueError(f"Expected a function or method, but got {obj}")
    return obj.__code__


def get_runtime_description(code: types.CodeType) -> Any:
    assert isinstance(code, types.CodeType)
    return get_sanitized_bytecode_representation(code=code)


def get_global_names_candidates(code: types.CodeType) -> Set[str]:
    result = set()
    instructions = list(dis.get_instructions(code))
    for instr in instructions:
        if instr.opname == "LOAD_GLOBAL":
            result.add(instr.argval)
        if isinstance(instr.argval, types.CodeType):
            result.update(get_global_names_candidates(instr.argval))
    return result


def get_sanitized_bytecode_representation(
    code: types.CodeType,
) -> List[dis.Instruction]:
    instructions = list(dis.get_instructions(code))
    result = []
    for instr in instructions:
        if isinstance(instr.argval, types.CodeType):
            result.append(
                dis.Instruction(
                    instr.opname,
                    instr.opcode,
                    instr.arg,
                    get_sanitized_bytecode_representation(instr.argval),
                    "",
                    instr.offset,
                    instr.starts_line,
                    is_jump_target=instr.is_jump_target,
                )
            )
        else:
            result.append(instr)
    return result


def unknown_function():
    # this is a placeholder function that we use to get the source of
    # functions that we can't get the source of
    pass


UNKNOWN_GLOBAL_VAR = "UNKNOWN_GLOBAL_VAR"


def get_bytecode(f: Union[types.FunctionType, types.CodeType, str]) -> str:
    if isinstance(f, str):
        f = compile(f, "<string>", "exec")
    instructions = dis.get_instructions(f)
    return "\n".join([str(i) for i in instructions])


def hash_dict(d: dict) -> str:
    return get_content_hash(obj=[(k, d[k]) for k in sorted(d.keys())])


def load_obj(module_name: str, obj_name: str) -> Tuple[Any, bool]:
    module = importlib.import_module(module_name)
    parts = obj_name.split(".")
    current = module
    found = True
    for part in parts:
        if not hasattr(current, part):
            found = False
            break
        else:
            current = getattr(current, part)
    return current, found


def get_dep_key_from_func(func: types.FunctionType) -> DepKey:
    module_name = func.__module__
    qualname = func.__qualname__
    return module_name, qualname


def get_func_qualname(
    func_name: str,
    code: types.CodeType,
    frame: types.FrameType,
) -> str:
    # this is evil
    referrers = gc.get_referrers(code)
    func_referrers = [r for r in referrers if isinstance(r, types.FunctionType)]
    matching_name = [r for r in func_referrers if r.__name__ == func_name]
    if len(matching_name) != 1:
        return get_func_qualname_fallback(func_name=func_name, code=code, frame=frame)
    else:
        return matching_name[0].__qualname__


def get_func_qualname_fallback(
    func_name: str, code: types.CodeType, frame: types.FrameType
) -> str:
    # get the argument names to *try* to tell if the function is a method
    arg_names = code.co_varnames[: code.co_argcount]
    # a necessary but not sufficient condition for this to
    # be a method
    is_probably_method = (
        len(arg_names) > 0
        and arg_names[0] == "self"
        and hasattr(frame.f_locals["self"].__class__, func_name)
    )
    if is_probably_method:
        # handle nested classes via __qualname__
        cls_qualname = frame.f_locals["self"].__class__.__qualname__
        func_qualname = f"{cls_qualname}.{func_name}"
    else:
        func_qualname = func_name
    return func_qualname
