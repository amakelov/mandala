"""
Function-level dependency tracking tools for Python. Briefly:
    - **dependency state**: a function `f`'s dependencies **for a given
    execution** `f(args)` are `f`'s source code, the global variables `f(args)`
    accesses, and recursively the dependencies of all functions called by
    `f(args)`.
    - this module provides tools to:
        - **find functional dependencies**: find the sources, modules and
        classes (if applicable) of all the functions and methods called by `f(args)`;
        - **find global variable dependencies**: find the values of the global
        variables contained in their source code (note this is an OVERESTIMATE
        of the true dependency state!)
        - **recover and compare**: given such an overestimate, try to
        recover the current state (potentially after changes to the source code)
        and compare it to the previous state.
    - there are various assumptions, limitations, and safeguards, SEE BELOW.
    This tracker is a best-effort attempt that leans towards (but does not
    always achieve!) completeness (catching all dependencies), and dynamically
    traces dependencies for each separate call to a function. It is not
    guaranteed to be sound (i.e. there may be false-positive dependencies), but
    should still be a helpful guide in a wide range of practical cases. Use at
    your own risk.

Assumptions:
    - a dependency to a subroutine is only tracked if the subroutine is actually
      called by the function. This means that different executions of the same
      function may find different dependencies. 
    - changes in functions are decided by comparing source code 
        - (could also use ast)
    - changes in global variables are decided by comparing content hashes
    - Only dependencies whose code is in a user-provided directory (or in the
      `__main__` module) are tracked.
    - furthermore, once the call stack reaches a function whose code is not in
    the directory, the tracker stops tracing to reduce overhead. This means that
    if a call to an external function then calls back some other function under
    the directory, this chain of dependency will not be tracked.
        - (this could be configurable)
    - Only functions whose source is available using `inspect.getsource` have
      their source code tracked (note that if `getsource` doesn't work for `f`,
      functions called by `f` will still be tracked).
    - Only global variables that can be content-hashed are tracked.

Guarantees:
    - The tracker is *complete (all dependencies are accounted for) when it
    works*, thanks to the use of dynamic tracing as opposed to static analysis.
    Every call to a function will be caught; when the tracker detects things it
    can't reliably track, it will (by default) raise an error.
    - It disambiguates between different functions with the same name in
    different modules.
    - It can track calls to methods to the proper subclass. 

Limitations:
    - **the tracker is possibly unsound**: 
        - it may overestimate dependencies on global variables. 
        - when a global variable has some structure (e.g. a dict), it may
        overestimate the part of the structure relevant for a given call of the
        function (the entire structure is tracked, even if only a small part of
        it is relevant for a given call).
        - it may overestimate the part of a function's source code relevant for
        a given call of the function (the entire function's source code is
        tracked, even if only a small part of it is relevant for a given call).
    - **the recovery process can't recover closure variables** because they
    don't exist at import time. This means that if a function uses a closure
    variable defined outside of any of the functions whose source code is
    tracked, the tracker will not be able to detect this dependency change.
    - **it's (currently) impossible to figure out when something is not a
    dependency any more**: false-positive dependencies may remain from old
    executions of a function.
    - **it does not track lambdas**, but tracks functions called by lambdas.
    Realistically, this is not a big limitation, since lambdas are usually
    defined in the same scope as the function that calls them, so will be
    tracked as part of the function's source code.
    

Safeguards: 
    - the tracker can be configured to raise an error when it detects conditions
    under which it can fail to guarantee completeness. These are:
        - a change in the source code of a function while tracing;
        - a change to a global variable's content hash while tracing;
        - closure variables used in a function while tracing.
        - a function whose source code is not available.
        - a global variable that cannot be content-hashed.
    - the tracker can also be configured to raise an error when recovering a
      stored description of dependencies fails:
        - if a function that was a dependency cannot be found in the current
          namespace, cannot be inspected, or is replaced by a non-function object;
        - if a global variable that was a dependency cannot be found in the
          current namespace, or cannot be content-hashed.

"""
from ..common_imports import *
from .utils import Hashing, remove_func_signature_and_comments, load_obj
from .config import Config
import sys
import inspect
import importlib
import types
import textwrap

from ..ui.viz import (
    Node as DotNode,
    Edge as DotEdge,
    Group as DotGroup,
    to_dot_string,
    SOLARIZED_LIGHT,
    write_output,
)


################################################################################
### dependency graph model
################################################################################
DepKey = Tuple[str, str]  # (module name, object address in module)


class Node:
    def __init__(
        self, module_name: str, obj_addr: str, representation: Optional[Any] = None
    ):
        self.module_name = module_name
        self.obj_addr = obj_addr
        self.representation = representation

    @property
    def key(self) -> DepKey:
        return (self.module_name, self.obj_addr)

    @staticmethod
    def represent(obj: Any) -> Any:
        raise NotImplementedError()

    def comparable_representation(self) -> Any:
        return self.representation

    def diff_representation(self) -> str:
        raise NotImplementedError()


class CallableNode(Node):
    @property
    def is_method(self) -> bool:
        return "." in self.obj_addr

    @property
    def class_name(self) -> str:
        assert self.is_method
        return ".".join(self.obj_addr.split(".")[:-1])

    @staticmethod
    def represent(obj: types.FunctionType) -> str:
        if type(obj).__name__ == "FuncInterface":
            obj = obj.func_op.func
        try:
            source = inspect.getsource(obj)
        except OSError:
            raise RuntimeError(
                f"Failed to get source code for {obj} in module {obj.__module__}"
            )
        return inspect.getsource(obj)

    def comparable_representation(self) -> str:
        return remove_func_signature_and_comments(self.representation)

    def diff_representation(self) -> str:
        return self.representation


class GlobalVarNode(Node):
    def __init__(
        self,
        module_name: str,
        obj_addr: str,
        # (content hash, truncated repr)
        representation: Optional[Tuple[str, str]] = None,
    ):
        super().__init__(module_name, obj_addr, representation)

    @staticmethod
    def represent(obj: Any) -> Tuple[str, str]:
        try:
            content_hash = Hashing.get_content_hash(obj=obj)
        except Exception as e:
            raise RuntimeError(
                f"Failed to hash global variable {obj} of type {type(obj)}"
            )
        return content_hash, textwrap.shorten(text=repr(obj), width=80)

    def comparable_representation(self) -> str:
        return self.representation[0]

    def diff_representation(self) -> str:
        return self.representation[1]


class TerminalData:
    def __init__(
        self, internal_name: str, version: int, module_name: str, func_name: str
    ):
        self.internal_name = internal_name
        self.version = version
        self.module_name = module_name
        self.func_name = func_name


class TerminalNode(Node):
    def __init__(self, module_name: str, obj_addr: str, representation: TerminalData):
        super().__init__(module_name, obj_addr, representation)

    @property
    def key(self) -> DepKey:
        return (self.representation.module_name, self.representation.func_name)


class DependencyGraph:
    def __init__(self):
        self.nodes: Dict[DepKey, Node] = {}
        self.roots: Set[DepKey] = set()
        self.edges: Set[Tuple[DepKey, DepKey]] = set()

    def show(self, path: Optional[Path] = None, how: str = "none"):
        dot = to_dot(self)
        output_ext = "svg" if how in ["browser"] else "png"
        return write_output(
            dot_string=dot, output_path=path, output_ext=output_ext, show_how=how
        )

    def __repr__(self) -> str:
        if len(self.nodes) == 0:
            return "DependencyGraph()"
        return to_string(self)

    def add_node(self, node: Node):
        self.nodes[node.key] = node

    def add_edge(self, source: Node, target: Node):
        if source.key not in self.nodes:
            self.nodes[source.key] = source
        if target.key not in self.nodes:
            self.nodes[target.key] = target
        self.edges.add((source.key, target.key))

    def remove_node(self, node_key: DepKey):
        if node_key in self.nodes:
            del self.nodes[node_key]
        self.roots.discard(node_key)
        self.edges = {
            (source, target)
            for source, target in self.edges
            if source != node_key and target != node_key
        }

    def load_current_state(self, keys: List[DepKey]) -> Dict[DepKey, Node]:
        result = {}
        for key in keys:
            node = copy.deepcopy(self.nodes[key])
            obj, found = load_obj(node.module_name, node.obj_addr)
            if found:
                node.representation = node.represent(obj)
                result[key] = node
        return result

    def update(self, other: "DependencyGraph"):
        self.nodes.update(other.nodes)
        self.roots.update(other.roots)
        self.edges.update(other.edges)

    def update_nodes(self, nodes: Dict[DepKey, Node]):
        for key, node in nodes.items():
            if key not in self.nodes:
                self.nodes[key] = node

    def update_representations(self, nodes: Dict[DepKey, Node]):
        for key, node in nodes.items():
            if key in self.nodes:
                self.nodes[key].representation = node.representation


################################################################################
### tracer
################################################################################
class Tracer:
    BREAK = "break"
    CONTINUE = "continue"
    KEEP = "keep"
    BREAK_SIGNAL = "break_signal"

    @staticmethod
    def break_signal(data):
        pass

    LAMBDA = "<lambda>"
    COMPREHENSIONS = ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")

    def __init__(self, graph: DependencyGraph, paths: List[Path], strict: bool = True):
        self.call_stack: List[Optional[CallableNode]] = []
        self.graph = graph
        self.paths = paths
        self.strict = strict

    def _process_failure(self, msg: str):
        if self.strict:
            raise RuntimeError(msg)
        else:
            logging.warning(msg)

    def find_most_recent_call(self) -> Optional[CallableNode]:
        if len(self.call_stack) == 0:
            return None
        else:
            # return the most recent non-None obj on the stack
            for i in range(len(self.call_stack) - 1, -1, -1):
                call = self.call_stack[i]
                if call is not None:
                    return call
            return None

    @staticmethod
    def get_func_key(func: Callable) -> DepKey:
        # get the module name and function name from the function
        module_name = func.__module__
        func_name = func.__qualname__
        return module_name, func_name

    @staticmethod
    def generate_terminal_data(
        func: Callable, internal_name: str, version: int
    ) -> TerminalData:
        module_name, func_name = Tracer.get_func_key(func=func)
        data = TerminalData(
            internal_name=internal_name,
            version=version,
            module_name=module_name,
            func_name=func_name,
        )
        return data

    @staticmethod
    def get_func_qualname(
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

    def control_flow(self, module_name: str, func_name: str) -> str:
        if func_name == self.LAMBDA:
            return self.CONTINUE
        try:
            # case 1: module is a file
            module = importlib.import_module(module_name)
            module_path = Path(inspect.getfile(module))
            assert module_path.is_absolute()
            if len(self.paths) != 0:
                if not any(root in module_path.parents for root in self.paths):
                    # module is not in the paths we're inspecting; stop tracing
                    return self.BREAK
                elif module_name.startswith(
                    Config.module_name
                ) and not module_name.startswith(Config.tests_module_name):
                    # this function is part of `mandala` functionality. Continue tracing
                    # but don't add it to the dependency state
                    return self.CONTINUE
                else:
                    return self.KEEP
            else:
                return self.KEEP
        except:
            # case 2: module is not a file, e.g. it is a jupyter notebook
            logging.debug(f"Module {module_name} is not a file")
            if module_name != "__main__":
                raise NotImplementedError(f"Cannot handle module {module_name}")
            return self.KEEP

    def __enter__(self):
        if sys.gettrace() is not None:
            # ensure this is used correctly
            raise RuntimeError("Another tracer is already active")

        def tracer(frame, event, arg):
            if event == "return":
                if len(self.call_stack) > 0:
                    self.call_stack.pop()
                else:
                    # something went wrong
                    raise RuntimeError("Call stack is empty")
            if event != "call":
                return

            # qualified name of the module where the function/method is defined.
            module_name = frame.f_globals.get("__name__")
            # code object of function being called
            code = frame.f_code
            # function's name
            func_name = code.co_name

            if func_name == self.BREAK_SIGNAL:
                data = frame.f_locals["data"]
                node = TerminalNode(
                    module_name=module_name, obj_addr=func_name, representation=data
                )
                most_recent_option = self.find_most_recent_call()
                if most_recent_option is not None:
                    self.graph.add_edge(source=most_recent_option, target=node)
                self.call_stack.append(None)
                return

            control = self.control_flow(module_name=module_name, func_name=func_name)
            if control == self.BREAK:
                return
            elif control == self.CONTINUE:
                self.call_stack.append(None)
                return tracer
            elif control == self.KEEP:
                pass
            else:
                raise ValueError(f"Invalid control value {control}")

            ### detect use of closure variables
            closure_vars = code.co_freevars
            if len(closure_vars) > 0 and func_name not in self.COMPREHENSIONS:
                closure_values = {
                    var: frame.f_locals.get(var, frame.f_globals.get(var, None))
                    for var in closure_vars
                }
                msg = f"Found closure variables accessed by function {module_name}.{func_name}:\n{closure_values}"
                self._process_failure(msg=msg)

            ### get the global variables used by the function
            globals_nodes = []
            for name in code.co_names:
                # names used by the function; not all of them are global variables
                if name in frame.f_globals:
                    global_val = frame.f_globals[name]
                    if (
                        inspect.ismodule(global_val)
                        or isinstance(global_val, type)
                        or inspect.isfunction(global_val)
                        or type(global_val).__name__ == "FuncInterface"  #! a hack
                    ):
                        # ignore modules, classes and functions
                        continue
                    node = GlobalVarNode(module_name=module_name, obj_addr=name)
                    globals_nodes.append(node)

            ### if this is a comprehension call, add the globals to the most
            ### recent tracked call
            if func_name in self.COMPREHENSIONS:
                most_recent_tracked_call = self.find_most_recent_call()
                assert most_recent_tracked_call is not None
                for global_node in globals_nodes:
                    self.graph.add_edge(
                        source=most_recent_tracked_call, target=global_node
                    )
                self.call_stack.append(None)
                return tracer

            ### get the qualified name of the function/method
            func_qualname = self.get_func_qualname(
                func_name=func_name, frame=frame, code=code
            )

            ### manage the call stack
            call = CallableNode(module_name=module_name, obj_addr=func_qualname)
            self.graph.add_node(node=call)
            ### global variable edges from this function always exist
            for global_node in globals_nodes:
                self.graph.add_edge(source=call, target=global_node)
            ### call edges exist only if there is a caller on the stack
            if len(self.call_stack) > 0:
                # find the most recent tracked call
                most_recent_tracked_call = self.find_most_recent_call()
                if most_recent_tracked_call is not None:
                    self.graph.add_edge(source=most_recent_tracked_call, target=call)
            self.call_stack.append(call)
            if len(self.call_stack) == 1:
                self.graph.roots.add(call.key)
            return tracer

        sys.settrace(tracer)

    def __exit__(self, *exc_info):
        sys.settrace(None)  # Stop tracing


################################################################################
### mandala-specific things
################################################################################
OpKey = Tuple[str, int]


class MandalaDependencies:
    def __init__(self):
        self.global_graph: DependencyGraph = DependencyGraph()
        # (internal name, version) -> graph for this op
        self.op_graphs: Dict[OpKey, DependencyGraph] = {}

    @staticmethod
    def expand_terminal(
        graph: DependencyGraph,
        terminal: TerminalNode,
        subgraph: DependencyGraph,
        replacement: Node,
    ):
        del graph.nodes[terminal.key]
        for key, node in subgraph.nodes.items():
            graph.nodes[key] = node
        # graph.roots.update(subgraph.roots)
        graph.edges.update(subgraph.edges)

    def get_expanded(self, op_key: OpKey) -> DependencyGraph:
        graph = copy.deepcopy(self.op_graphs.get(op_key, DependencyGraph()))
        terminals = [
            node for node in graph.nodes.values() if isinstance(node, TerminalNode)
        ]
        for terminal in terminals:
            internal_name, version = (
                terminal.representation.internal_name,
                terminal.representation.version,
            )
            sub_key: OpKey = (internal_name, version)
            subgraph = self.get_expanded(op_key=sub_key)
            assert len(subgraph.roots) == 1
            replacement = subgraph.nodes[list(subgraph.roots)[0]]
            self.expand_terminal(
                graph=graph,
                terminal=terminal,
                subgraph=subgraph,
                replacement=replacement,
            )
        for key, node in graph.nodes.items():
            node.representation = self.global_graph.nodes[key].representation
        return graph

    def get_deps_to_ops(self) -> Dict[DepKey, List[OpKey]]:
        """
        Importantly, return only latest versions of each op.
        """
        res: Dict[DepKey, List[OpKey]] = {}
        latest_versions = {k: 0 for k, _ in self.op_graphs.keys()}
        for k, version in self.op_graphs.keys():
            if version > latest_versions[k]:
                latest_versions[k] = version
        latest_keys = {
            (k, v) for k, v in self.op_graphs.keys() if v == latest_versions[k]
        }
        for op_key in latest_keys:
            graph = self.get_expanded(op_key=op_key)
            for dep_key in graph.nodes:
                res.setdefault(dep_key, []).append(op_key)
        return res

    def update_op(self, op_key: OpKey, graph: DependencyGraph):
        ### unify the graph with this op's graph
        if op_key not in self.op_graphs:
            self.op_graphs[op_key] = DependencyGraph()
        self.op_graphs[op_key].update(other=graph)
        ### unify the global graph with this op's graph
        # add the non-terminals
        nodes = {
            key: node
            for key, node in graph.nodes.items()
            if not isinstance(node, TerminalNode)
        }
        self.global_graph.update_nodes(nodes=nodes)
        # load the representations of these new objects
        current_state = self.global_graph.load_current_state(keys=nodes.keys())
        self.global_graph.update_representations(nodes=current_state)


################################################################################
### visualizations
################################################################################
def to_string(graph: DependencyGraph) -> str:
    """
    Get a string for pretty-printing.
    """
    # group the nodes by module
    module_groups: Dict[str, List[Node]] = {}
    for key, node in graph.nodes.items():
        module_name, _ = key
        module_groups.setdefault(module_name, []).append(node)
    # for each module, include the representations of the global variables first,
    # then the functions.
    lines = []
    for module_name, nodes in module_groups.items():
        global_nodes = [node for node in nodes if isinstance(node, GlobalVarNode)]
        callable_nodes = [node for node in nodes if isinstance(node, CallableNode)]
        module_desc = f"MODULE: {module_name}"
        lines.append(module_desc)
        lines.append("-" * len(module_desc))
        lines.append("===Global Variables===")
        for node in global_nodes:
            desc = f"{node.obj_addr} = {node.diff_representation()}"
            lines.append(textwrap.indent(desc, 4 * " "))
            # lines.append(f"  {node.diff_representation()}")
        lines.append("")
        lines.append("===Functions===")
        # group the methods by class
        method_nodes = [node for node in callable_nodes if node.is_method]
        func_nodes = [node for node in callable_nodes if not node.is_method]
        methods_by_class: Dict[str, List[CallableNode]] = {}
        for method_node in method_nodes:
            methods_by_class.setdefault(method_node.class_name, []).append(method_node)
        for class_name, method_nodes in methods_by_class.items():
            lines.append(textwrap.indent(f"class {class_name}:", 4 * " "))
            for node in method_nodes:
                desc = node.diff_representation()
                lines.append(textwrap.indent(textwrap.dedent(desc), 8 * " "))
            lines.append("")
        for node in func_nodes:
            desc = node.diff_representation()
            lines.append(textwrap.indent(textwrap.dedent(desc), 4 * " "))
        lines.append("")
    return "\n".join(lines)


def to_dot(graph: DependencyGraph) -> str:
    nodes: Dict[DepKey, DotNode] = {}
    module_groups: Dict[str, DotGroup] = {}  # module name -> Group
    class_groups: Dict[str, DotGroup] = {}  # class name -> Group
    for key, node in graph.nodes.items():
        module_name, obj_addr = key
        if module_name not in module_groups:
            module_groups[module_name] = DotGroup(
                label=module_name, nodes=[], parent=None
            )
        if isinstance(node, GlobalVarNode):
            color = SOLARIZED_LIGHT["red"]
        elif isinstance(node, CallableNode):
            color = (
                SOLARIZED_LIGHT["blue"]
                if not node.is_method
                else SOLARIZED_LIGHT["violet"]
            )
        else:
            color = SOLARIZED_LIGHT["base03"]
            # raise NotImplementedError()
        dot_node = DotNode(
            internal_name=".".join(key), label=node.obj_addr, color=color
        )
        nodes[key] = dot_node
        module_groups[module_name].nodes.append(dot_node)
        if isinstance(node, CallableNode) and node.is_method:
            class_name = node.class_name
            class_groups.setdefault(
                class_name,
                DotGroup(
                    label=class_name,
                    nodes=[],
                    parent=module_groups[module_name],
                ),
            ).nodes.append(dot_node)
    edges: Dict[Tuple[DotNode, DotNode], DotEdge] = {}
    for source, target in graph.edges:
        source_node = nodes[source]
        target_node = nodes[target]
        edge = DotEdge(source_node=source_node, target_node=target_node)
        edges[(source_node, target_node)] = edge
    dot_string = to_dot_string(
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        groups=list(module_groups.values()) + list(class_groups.values()),
        rankdir="BT",
    )
    return dot_string
