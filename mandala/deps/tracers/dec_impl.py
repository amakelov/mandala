import types
from functools import wraps, update_wrapper
from ...common_imports import *
from ...utils import unwrap_decorators
from ...config import Config
from ..model import (
    DependencyGraph,
    CallableNode,
    GlobalVarNode,
    TerminalData,
    TerminalNode,
)
from ..utils import (
    get_global_names_candidates,
    is_global_val,
    extract_code,
    extract_func_obj,
    DepKey,
    GlobalClassifier,
)
from .tracer_base import TracerABC, get_module_flow, KEEP, get_closure_names


class DecTracerConfig:
    allow_class_tracking = True
    restrict_global_accesses = False
    allow_owned_class_accesses = False
    allow_nonfunc_attributes = False


class TracerState:
    tracer: Optional["DecTracer"] = None
    registry: Dict[DepKey, Any] = {}

    @staticmethod
    def is_tracked(f: Union[types.FunctionType, type]) -> bool:
        assert isinstance(f, (types.FunctionType, type))
        dep_key = (f.__module__, f.__qualname__)
        return dep_key in TracerState.registry and TracerState.registry[dep_key] is f


class TrackedDict(dict):
    """
    A dictionary that tracks global variable accesses.
    """
    def __init__(self, original: dict, verbose: bool = False):
        self.__original__ = original
        self.verbose = verbose
    
    def __repr__(self) -> str:
        return f"TrackedDict({self.__original__})"

    def __getitem__(self, __key: str) -> Any:
        result = self.__original__.__getitem__(__key)
        if TracerState.tracer is not None:
            tracer = TracerState.tracer
            unwrapped_result = unwrap_decorators(result, strict=False)
            if isinstance(unwrapped_result, (types.FunctionType, type)):
                is_owned = tracer.is_owned_obj(obj=unwrapped_result)
                is_cls_access = isinstance(result, type)
                if (
                    is_owned
                    and is_cls_access
                    and not DecTracerConfig.allow_owned_class_accesses
                ):
                    raise ValueError(
                        f"Attempting to access class {result} from module {unwrapped_result.__module__}."
                    )
                is_tracked = TracerState.is_tracked(unwrapped_result)
                if is_owned and not is_tracked:
                    raise ValueError(
                        f"Function/class {result} from module {unwrapped_result.__module__} is accessed but not tracked"
                    )
            elif is_global_val(result):
                tracer.register_global_access(key=__key, value=result)
            else:
                if (
                    DecTracerConfig.restrict_global_accesses
                    and not GlobalClassifier.is_excluded(result)
                ):
                    raise ValueError(
                        f"Accessing global value {result} of type {type(result)} is not allowed"
                    )
                else:
                    # we failed to classify this value
                    msg = f"Accessing global value {result} of type {type(result)} is not tracked, because it couldn't be classified."
                    if self.verbose:
                        logger.warning(msg)
                    else:
                        logger.debug(msg)
        return result


def make_tracked_copy(f: types.FunctionType) -> types.FunctionType:
    result = types.FunctionType(
        code=f.__code__,
        globals=TrackedDict(f.__globals__),
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    result = update_wrapper(result, f)
    result.__module__ = f.__module__
    result.__kwdefaults__ = copy.deepcopy(f.__kwdefaults__)
    result.__annotations__ = copy.deepcopy(f.__annotations__)
    return result


def get_nonfunc_attributes(cls: type) -> Dict[str, Any]:
    result = {}
    for k, v in cls.__dict__.items():
        if not k.startswith("__") and not isinstance(
            unwrap_decorators(v, strict=False), (types.FunctionType, type)
        ):
            result[k] = v
    return result


def track(obj: Union[types.FunctionType, type]) -> "obj":
    if isinstance(obj, type):
        if not DecTracerConfig.allow_class_tracking:
            raise ValueError("Class tracking is not allowed")
        if not DecTracerConfig.allow_nonfunc_attributes:
            nonfunc_attributes = get_nonfunc_attributes(obj)
            if len(nonfunc_attributes) > 0:
                raise ValueError(
                    f"Class tracking for {obj} is not allowed: found non-function attributes {nonfunc_attributes}"
                )
        # decorate all methods/classes in the class
        for k, v in obj.__dict__.items():
            if isinstance(v, (types.FunctionType, type)):
                setattr(obj, k, track(v))
        TracerState.registry[(obj.__module__, obj.__qualname__)] = obj
        return obj
    elif isinstance(obj, types.FunctionType):
        #! NOTE: this was done with weird decorators in mind, but might actually
        # be a bad idea...
        # obj = make_tracked_copy(unwrap_decorators(obj, strict=True))
        obj = make_tracked_copy(obj)

        @wraps(obj)
        def wrapper(*args, **kwargs) -> Any:
            tracer = DecTracer.get_active_trace_obj()
            if tracer is not None:
                node = tracer.register_call(func=obj)
                outcome = obj(*args, **kwargs)
                tracer.register_return(node)
                return outcome
            else:
                return obj(*args, **kwargs)

        TracerState.registry[(obj.__module__, obj.__qualname__)] = unwrap_decorators(
            obj, strict=True
        )
        return wrapper
    elif type(obj).__name__ == Config.func_interface_cls_name:
        obj.f = make_tracked_copy(f=obj.f)
        TracerState.registry[
            (obj.f.__module__, obj.f.__qualname__)
        ] = obj.f
        return obj
    else:
        raise TypeError("Can only track callable objects")


class DecTracer(TracerABC):
    """
    A decorator-based tracer that tracks function calls and global variable accesses.
    """
    def __init__(
        self,
        paths: List[Path],
        graph: Optional[DependencyGraph] = None,
        strict: bool = True,
        track_globals: bool = True,
        allow_methods: bool = False,
        skip_unhashable_globals: bool = True,
        skip_globals_silently: bool = False,
    ):
        self.call_stack: List[CallableNode] = []
        self.graph = DependencyGraph() if graph is None else graph
        self.paths = paths
        self.strict = strict

        self.track_globals = track_globals
        self.skip_unhashable_globals = skip_unhashable_globals
        self.skip_globals_silently = skip_globals_silently

        self.allow_methods = allow_methods

        self._traced = {}
        self._traced_funcs = {}

    def is_owned_obj(self, obj: Union[types.FunctionType, type]) -> bool:
        module_name = obj.__module__
        return get_module_flow(module_name=module_name, paths=self.paths) == KEEP

    @staticmethod
    def get_active_trace_obj() -> Optional["DecTracer"]:
        return TracerState.tracer

    @staticmethod
    def set_active_trace_obj(trace_obj: Optional["DecTracer"]):
        if trace_obj is not None and TracerState.tracer is not None:
            raise ValueError("Tracer already active")
        TracerState.tracer = trace_obj

    def get_globals(self, func: Callable) -> List[GlobalVarNode]:
        """
        Get the global variables available to the function as a list of
        GlobalVarNode objects. 

        Currently, this is not used, because it doesn't really track accesses
        to globals, and can thus over-estimate the dependencies of a function.
        """
        # result = []
        # code_obj = extract_code(obj=func)
        # global_scope = extract_func_obj(obj=func, strict=self.strict).__globals__
        # for name in get_global_names_candidates(code=code_obj):
        #     # names used by the function; not all of them are global variables
        #     if name in global_scope.keys():
        #         global_val = global_scope[name]
        #         if not is_global_val(global_val):
        #             continue
        #         node = GlobalVarNode.from_obj(
        #             obj=global_val, dep_key=(func.__module__, name)
        #         )
        #         result.append(node)
        # return result
        return []

    def register_call(self, func: Callable) -> CallableNode:
        module_name = func.__module__
        qualname = func.__qualname__
        # check for closure variables
        closure_names = get_closure_names(
            code_obj=func.__code__, func_qualname=qualname
        )
        if len(closure_names) > 0:
            msg = f"Found closure variables accessed by function {module_name}.{qualname}:\n{closure_names}"
            self._process_failure(msg, level='debug')
        ### get call node
        node = CallableNode.from_runtime(
            module_name=module_name, obj_name=qualname, code_obj=extract_code(obj=func)
        )
        self.call_stack.append(node)
        self.graph.add_node(node)
        if len(self.call_stack) > 1:
            parent = self.call_stack[-2]
            assert parent is not None
            self.graph.add_edge(parent, node)
        ### get globals
        # global_nodes = self.get_globals(func=func)
        # for global_node in global_nodes:
        #     self.graph.add_edge(node, global_node)
        if len(self.call_stack) == 1:
            # this is the root of the graph
            self.graph.roots.add(node.key)
        return node

    def register_global_access(self, key: str, value: Any):
        if not self.track_globals:
            return
        assert len(self.call_stack) > 0
        calling_node = self.call_stack[-1]
        node = GlobalVarNode.from_obj(
            obj=value, dep_key=(calling_node.module_name, key), 
            skip_unhashable=self.skip_unhashable_globals,
            skip_silently=self.skip_globals_silently
        )
        self.graph.add_edge(calling_node, node)

    def register_return(self, node: CallableNode):
        assert self.call_stack[-1] == node
        self.call_stack.pop()

    @staticmethod
    def register_leaf_event(trace_obj: "DecTracer", data: TerminalData):
        unique_id = "_".join(
            [
                data.op_internal_name,
                str(data.op_version),
                data.call_content_version,
                data.call_semantic_version,
            ]
        )
        module_name = data.dep_key[0]
        node = TerminalNode(
            module_name=module_name, obj_name=unique_id, representation=data
        )
        if len(trace_obj.call_stack) > 0:
            trace_obj.graph.add_edge(trace_obj.call_stack[-1], node)
        return

    @staticmethod
    def leaf_signal(data):
        # a way to detect the end of a trace
        raise NotImplementedError

    def __enter__(self):
        DecTracer.set_active_trace_obj(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DecTracer.set_active_trace_obj(None)

    def _process_failure(self, msg: str, level: str = 'warning'):
        if self.strict:
            raise RuntimeError(msg)
        else:
            if level == 'warning':
                logger.warning(msg)
            elif level == 'debug':
                logger.debug(msg)