import types
from ...common_imports import *
from ..utils import (
    get_func_qualname,
    is_global_val,
    get_global_names_candidates,
)
from ..model import (
    DependencyGraph,
    CallableNode,
    TerminalData,
    TerminalNode,
    GlobalVarNode,
)
import sys
import importlib
from .tracer_base import TracerABC, get_closure_names

################################################################################
### tracer
################################################################################
# control flow constants
from .tracer_base import BREAK, CONTINUE, KEEP, MAIN, get_module_flow

LEAF_SIGNAL = "leaf_signal"

# constants for special Python function names
LAMBDA = "<lambda>"
COMPREHENSIONS = ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")
SKIP_FRAMES = tuple(list(COMPREHENSIONS) + [LAMBDA])


class SysTracer(TracerABC):
    def __init__(
        self,
        paths: List[Path],
        graph: Optional[DependencyGraph] = None,
        strict: bool = True,
        allow_methods: bool = False,
    ):
        self.call_stack: List[Optional[CallableNode]] = []
        self.graph = DependencyGraph() if graph is None else graph
        self.paths = paths
        self.path_strs = [str(path) for path in paths]
        self.strict = strict
        self.allow_methods = allow_methods

    @staticmethod
    def leaf_signal(data):
        # a way to detect the end of a trace
        pass

    @staticmethod
    def register_leaf_event(trace_obj: types.FunctionType, data: Any):
        SysTracer.leaf_signal(data)

    @staticmethod
    def get_active_trace_obj() -> Optional[Any]:
        return sys.gettrace()

    @staticmethod
    def set_active_trace_obj(trace_obj: Any):
        sys.settrace(trace_obj)

    def _process_failure(self, msg: str):
        if self.strict:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)

    def find_most_recent_call(self) -> Optional[CallableNode]:
        if len(self.call_stack) == 0:
            return None
        else:
            # return the most recent non-None obj on the stack
            for i in range(len(self.call_stack) - 1, -1, -1):
                call = self.call_stack[i]
                if isinstance(call, CallableNode):
                    return call
            return None

    def __enter__(self):
        if sys.gettrace() is not None:
            # pre-check this is used correctly
            raise RuntimeError("Another tracer is already active")

        def tracer(frame: types.FrameType, event: str, arg: Any):
            if event not in ("call", "return"):
                return
            module_name = frame.f_globals.get("__name__")
            # fast check to rule out non-user code
            if event == "call":
                try:
                    module = importlib.import_module(module_name)
                    if not any(
                        [
                            module.__file__.startswith(path_str)
                            for path_str in self.path_strs
                        ]
                    ):
                        return
                except:
                    if module_name != MAIN:
                        return
            code_obj = frame.f_code
            func_name = code_obj.co_name
            if event == "return":
                logging.debug(f"Returning from {func_name}")
                if len(self.call_stack) > 0:
                    popped = self.call_stack.pop()
                    logging.debug(f"Popped {popped} from call stack")
                    # some sanity checks
                    if func_name in SKIP_FRAMES:
                        if popped != func_name:
                            self._process_failure(
                                f"Expected to pop {func_name} from call stack, but popped {popped}"
                            )
                    else:
                        if popped.obj_name.split(".")[-1] != func_name:
                            self._process_failure(
                                f"Expected to pop {func_name} from call stack, but popped {popped.obj_name}"
                            )
                else:
                    # something went wrong
                    raise RuntimeError("Call stack is empty")
                return

            if func_name == LEAF_SIGNAL:
                data: TerminalData = frame.f_locals["data"]
                unique_id = "_".join(
                    [
                        data.op_internal_name,
                        str(data.op_version),
                        data.call_content_version,
                        data.call_semantic_version,
                    ]
                )
                node = TerminalNode(
                    module_name=module_name, obj_name=unique_id, representation=data
                )
                most_recent_option = self.find_most_recent_call()
                if most_recent_option is not None:
                    self.graph.add_edge(source=most_recent_option, target=node)
                # self.call_stack.append(None)
                return

            module_control_flow = get_module_flow(
                paths=self.paths, module_name=module_name
            )
            if module_control_flow in (BREAK, CONTINUE):
                frame.f_trace = None
                return

            logger.debug(f"Tracing call to {module_name}.{func_name}")

            ### get the qualified name of the function/method
            func_qualname = get_func_qualname(
                func_name=func_name, code=code_obj, frame=frame
            )
            if "." in func_qualname:
                if not self.allow_methods:
                    raise RuntimeError(
                        f"Methods are currently not supported: {func_qualname} from {module_name}"
                    )

            ### detect use of closure variables
            closure_names = get_closure_names(
                code_obj=code_obj, func_qualname=func_qualname
            )
            if len(closure_names) > 0 and func_name not in SKIP_FRAMES:
                closure_values = {
                    var: frame.f_locals.get(var, frame.f_globals.get(var, None))
                    for var in closure_names
                }
                msg = f"Found closure variables accessed by function {module_name}.{func_name}:\n{closure_values}"
                self._process_failure(msg=msg)

            ### get the global variables used by the function
            globals_nodes = []
            for name in get_global_names_candidates(code=code_obj):
                # names used by the function; not all of them are global variables
                if name in frame.f_globals:
                    global_val = frame.f_globals[name]
                    if not is_global_val(global_val):
                        continue
                    node = GlobalVarNode.from_obj(
                        obj=global_val, dep_key=(module_name, name)
                    )
                    globals_nodes.append(node)

            ### if this is a comprehension call, add the globals to the most
            ### recent tracked call
            if func_name in SKIP_FRAMES:
                most_recent_tracked_call = self.find_most_recent_call()
                assert most_recent_tracked_call is not None
                for global_node in globals_nodes:
                    self.graph.add_edge(
                        source=most_recent_tracked_call, target=global_node
                    )
                self.call_stack.append(func_name)
                return tracer

            ### manage the call stack
            call_node = CallableNode.from_runtime(
                module_name=module_name, obj_name=func_qualname, code_obj=code_obj
            )
            self.graph.add_node(node=call_node)
            ### global variable edges from this function always exist
            for global_node in globals_nodes:
                self.graph.add_edge(source=call_node, target=global_node)
            ### call edges exist only if there is a caller on the stack
            if len(self.call_stack) > 0:
                # find the most recent tracked call
                most_recent_tracked_call = self.find_most_recent_call()
                if most_recent_tracked_call is not None:
                    self.graph.add_edge(
                        source=most_recent_tracked_call, target=call_node
                    )
            self.call_stack.append(call_node)
            if len(self.call_stack) == 1:
                self.graph.roots.add(call_node.key)
            return tracer

        sys.settrace(tracer)

    def __exit__(self, *exc_info):
        sys.settrace(None)  # Stop tracing


class SuspendSysTraceContext:
    def __init__(self):
        self.suspended_trace = None

    def __enter__(self) -> "SuspendSysTraceContext":
        if sys.gettrace() is not None:
            self.suspended_trace = sys.gettrace()
            sys.settrace(None)
        return self

    def __exit__(self, *exc_info):
        if self.suspended_trace is not None:
            sys.settrace(self.suspended_trace)
            self.suspended_trace = None
