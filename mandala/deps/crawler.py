import types
from ..common_imports import *
from ..utils import unwrap_decorators
import importlib
from .model import (
    DepKey,
    CallableNode,
)
from .utils import (
    is_callable_obj,
    extract_func_obj,
    unknown_function,
)


def crawl_obj(
    obj: Any,
    module_name: str,
    include_methods: bool,
    result: Dict[DepKey, CallableNode],
    strict: bool,
    objs_result: Dict[DepKey, Callable],
):
    """
    Find functions and optionally methods native to the module of this object.
    """
    if is_callable_obj(obj=obj, strict=strict):
        if isinstance(unwrap_decorators(obj, strict=False), types.BuiltinFunctionType):
            return
        v = extract_func_obj(obj=obj, strict=strict)
        if v is not unknown_function and v.__module__ != module_name:
            # exclude non-local functions
            return
        dep_key = (module_name, v.__qualname__)
        node = CallableNode.from_obj(obj=v, dep_key=dep_key)
        result[dep_key] = node
        objs_result[dep_key] = obj
    if isinstance(obj, type):
        if include_methods:
            if obj.__module__ != module_name:
                return
            for k in obj.__dict__.keys():
                v = obj.__dict__[k]
                crawl_obj(
                    obj=v,
                    module_name=module_name,
                    include_methods=include_methods,
                    result=result,
                    strict=strict,
                    objs_result=objs_result,
                )


def crawl_static(
    root: Optional[Path],
    strict: bool,
    package_name: Optional[str] = None,
    include_methods: bool = False,
) -> Tuple[Dict[DepKey, CallableNode], Dict[DepKey, Callable]]:
    """
    Find all python files in the root directory, and use importlib to import
    them, look for callable objects, and create callable nodes from them.
    """
    result: Dict[DepKey, CallableNode] = {}
    objs_result: Dict[DepKey, Callable] = {}
    paths = []
    if root is not None:
        if root.is_file():
            assert package_name is not None  # needs this to be able to import
            paths = [root]
        else:
            paths.extend(list(root.rglob("*.py")))
    paths.append("__main__")
    for path in paths:
        filename = path.name if path != "__main__" else "__main__"
        if filename in ("setup.py", "console.py"):
            continue
        if path != "__main__" and root is not None:
            if root.is_file():
                module_name = root.stem
            else:
                module_name = (
                    path.with_suffix("").relative_to(root).as_posix().replace("/", ".")
                )
            if package_name is not None:
                module_name = ".".join([package_name, module_name])
        else:
            module_name = "__main__"
        try:
            module = importlib.import_module(module_name)
        except:
            msg = f"Failed to import {module_name}:"
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                continue
        keys = list(module.__dict__.keys())
        for k in keys:
            v = module.__dict__[k]
            crawl_obj(
                obj=v,
                module_name=module_name,
                strict=strict,
                include_methods=include_methods,
                result=result,
                objs_result=objs_result,
            )
    return result, objs_result
