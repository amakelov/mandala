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
      their source code tracked (note that calls they make to other functions
      may still be tracked).
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
        - it can overestimate dependencies on global variables. 
        - when a global variable has some structure (e.g. a dict), it can
        overestimate the part of the structure relevant for a given call of the
        function (the entire structure is tracked, even if only a small part of
        it is relevant for a given call).
        - it can overestimate the part of a function's source code relevant for
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
from ..core.utils import Hashing
from ..core.config import Config
import sys
import inspect
import importlib
import textwrap
from textwrap import indent
import difflib
import webbrowser

from ..ui.viz import (
    Node,
    Edge,
    Group,
    to_dot_string,
    write_output,
    SOLARIZED_LIGHT,
    write_output,
)

################################################################################
### utils
################################################################################
ValType = TypeVar("ValType")
DiffType = TypeVar("DiffType")
NestedDictType = Dict[str, Dict[str, ValType]]


def _nested_dict_subtract(
    x: NestedDictType[ValType], y: NestedDictType[ValType]
) -> NestedDictType[ValType]:
    """
    Given two nested string-keyed dictionaries, return a nested dict that has
    all the "depth-2" keys in `x` that are not found in `y`
    """
    result: NestedDictType = {}
    for k, v in x.items():
        if k not in y:
            result[k] = v
            continue
        for k2, v2 in v.items():
            if k2 not in y[k]:
                result.setdefault(k, {})[k2] = v2
    return result


def _nested_dict_diff(
    new: NestedDictType[ValType],
    old: NestedDictType[ValType],
    differ: Callable[[ValType, ValType], DiffType],
) -> NestedDictType[DiffType]:
    """
    Given two nested string-keyed dictionaries, return a nested dict that has
    the result of `differ` applied to all the "depth-2" keys in `new` that are
    also found in `old` but have different values.
    """
    changed = {}
    for new_outer_key, new_inner_dict in new.items():
        if new_outer_key not in old:
            continue
        for new_inner_key, new_val in new_inner_dict.items():
            if new_inner_key not in old[new_outer_key]:
                continue
            old_val = old[new_outer_key][new_inner_key]
            if new_val != old_val:  # note that values can use custom __eq__ here
                changed.setdefault(new_outer_key, {})[new_inner_key] = differ(
                    new_val, old_val
                )
    return changed


def _pprint_structure(
    globals_text: Dict[str, Dict[str, str]], sources_text: Dict[str, Dict[str, str]]
) -> str:
    """
    Pretty-print text about global variables and functions, organized by module
    name.

    Does not handle truncation.
    """
    all_modules = set(globals_text.keys()).union(sources_text.keys())
    lines = []
    for module in all_modules:
        # add the module name to the lines, with some decoration around
        module_line = f"MODULE {module}:"
        module_below = "=" * len(module_line)
        lines += [module_line, module_below]
        ### show globals data
        if module in globals_text and len(globals_text[module]) > 0:
            lines.append(indent("===GLOBALS===:", " " * 4))
            for global_name in globals_text[module].keys():
                s = globals_text[module][global_name]
                lines.append(
                    indent(
                        f"{global_name}: {s}",
                        " " * 8,
                    )
                )
        ### show sources
        if module in sources_text and len(sources_text[module]) > 0:
            lines.append(indent("===FUNCTIONS===:", " " * 4))
            for func_name, source in sources_text[module].items():
                lines.append(indent(f"<<<{func_name}>>>:", " " * 8))
                source = textwrap.dedent(source)
                source_lines = source.splitlines()
                lines += [indent(f"{line}", " " * 12) for line in source_lines]
        # add a blank line between modules
        lines.append("")
    return "\n".join(lines)


def _colorize(text: str, color: str) -> str:
    """
    Return `text` with ANSI color codes for `color` added.
    """
    colors = {
        "red": 31,
        "green": 32,
        "blue": 34,
        "yellow": 33,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
    }
    return f"\033[{colors[color]}m{text}\033[0m"


def _get_colorized_diff(current: str, new: str) -> str:
    """
    Return a line-by-line colorized diff of the changes between `current` and
    `new`. each line removed from `current` is colored red, and each line added
    to `new` is colored green.
    """
    lines = []
    for line in difflib.unified_diff(
        current.splitlines(),
        new.splitlines(),
        n=2,  # number of lines of context around changes to show
        # fromfile="current", tofile="new"
        lineterm="",
    ):
        if line.startswith("@@") or line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("-"):
            lines.append(_colorize(line, "red"))
        elif line.startswith("+"):
            lines.append(_colorize(line, "green"))
        else:
            lines.append(line)
    return "\n".join(lines)


################################################################################
### a model of dependencies
################################################################################
class GlobalVar:
    def __init__(self, name: str, content_hash: Optional[str], repr_: Optional[str]):
        self.content_hash = content_hash
        self.repr_ = repr_
        self.name = name

    def __eq__(self, other: object) -> bool:
        """
        Equality is decided by content hash alone
        """
        if not isinstance(other, GlobalVar):
            return NotImplemented
        return self.content_hash == other.content_hash

    def to_string(self) -> str:
        return (
            "<repr() not available>"
            if self.repr_ is None
            else textwrap.shorten(self.repr_, width=80)
        )

    def __repr__(self) -> str:
        return f"GlobalVar({self.name}, {self.content_hash}, {self.repr_})"


class FuncSource:
    def __init__(
        self, source: Optional[str], qualified_name: str, is_op: Optional[bool] = None
    ):
        self.source = source
        self.is_op = is_op
        self.qualified_name = qualified_name

    @property
    def is_method(self) -> bool:
        return "." in self.qualified_name

    @property
    def class_name(self) -> str:
        return self.qualified_name.split(".")[0]

    def to_string(self) -> str:
        return "<source not available>" if self.source is None else self.source

    def __eq__(self, other: object) -> bool:
        """
        Importantly, @ops will compare equal when their bodies agree, even if
        the signatures are different. This is to enable refactoring of @op
        signatures without triggering the dependency tracking logic.

        This decision should remain hidden in this class.
        """
        if not isinstance(other, FuncSource):
            return NotImplemented
        if self.is_op:
            if not other.is_op:
                return False
            if self.source is None or other.source is None:
                return self.source == other.source
            return FuncSource.remove_function_signature(
                self.source
            ) == FuncSource.remove_function_signature(other.source)
        return self.source == other.source and self.is_op == other.is_op

    @staticmethod
    def remove_function_signature(source: str) -> str:
        """
        Given the source code of a function, remove the part that contains the
        function signature.

        This is used to prevent changes to the signatures of `@op` functions from
        triggering the dependency tracking logic.

        NOTE: Has the extra effect of removing comments and docstrings by going
        through an ast parse->unparse cycle. This means that it should be
        applied consistently when recovering dependency state.
        """
        # using dedent is necessary here to handle decorators
        tree = ast.parse(textwrap.dedent(source))
        assert isinstance(tree, ast.Module)
        body = tree.body
        assert len(body) == 1
        assert isinstance(body[0], ast.FunctionDef)
        func_body = body[0].body
        return ast.unparse(func_body)


class Call:
    def __init__(
        self,
        is_tracked: bool,
        module_name: str = "",
        qualified_func_name: str = "",
    ):
        self.is_tracked = is_tracked
        if self.is_tracked:
            assert all([module_name is not None, qualified_func_name is not None])
        self.module_name = module_name
        self.qualified_func_name = qualified_func_name

    def __repr__(self) -> str:
        return (
            f"Call({self.is_tracked}, {self.module_name}, {self.qualified_func_name})"
        )


class CallEdge:
    def __init__(self, caller: Call, callee: Call):
        assert caller.is_tracked and callee.is_tracked
        self.caller = caller
        self.callee = callee

    def __repr__(self) -> str:
        return f"CallEdge({self.caller}, {self.callee})"


class GlobalEdge:
    def __init__(self, caller: Call, global_var: GlobalVar):
        assert caller.is_tracked
        self.caller = caller
        self.global_var = global_var

    def __repr__(self) -> str:
        return f"GlobalEdge({self.caller}, {self.global_var})"


class DependencyState:
    def __init__(
        self,
        roots: List[Path],
        sources: Optional[Dict[str, Dict[str, FuncSource]]] = None,
        globals_: Optional[Dict[str, Dict[str, GlobalVar]]] = None,
        call_edges: Optional[List[CallEdge]] = None,
        globals_edges: Optional[List[GlobalEdge]] = None,
    ):
        """
        - roots=[] means unrestricted tracing. Otherwise, you only look
          inside the paths in `roots`.
        """
        self.roots = [root.absolute().resolve() for root in roots]
        # module name -> func address in module -> `FuncSource`
        self.sources = {} if sources is None else sources
        # qualified module name -> global variable name -> `GlobalVar`
        self.globals_ = {} if globals_ is None else globals_
        # tracked by the call stack
        self.call_edges = [] if call_edges is None else call_edges
        self.globals_edges = [] if globals_edges is None else globals_edges

    def show(self, path: Optional[Path] = None, how: str = "none"):
        dot = self.to_dot()
        output_ext = "svg" if how in ["browser"] else "png"
        return write_output(
            dot_string=dot, output_path=path, output_ext=output_ext, show_how=how
        )

    @staticmethod
    def get_readable_description(
        sources: Dict[str, Dict[str, FuncSource]],
        globals_: Dict[str, Dict[str, GlobalVar]],
    ) -> str:
        sources_text = {
            k: {k2: v2.to_string() for k2, v2 in v.items()} for k, v in sources.items()
        }
        globals_text = {
            k: {k2: v2.to_string() for k2, v2 in v.items()} for k, v in globals_.items()
        }
        return _pprint_structure(globals_text=globals_text, sources_text=sources_text)

    def __repr__(self) -> str:
        if self.size == 0:
            return "Empty DependencyState object"
        return self.get_readable_description(self.sources, self.globals_)

    @property
    def num_sources(self) -> int:
        return sum(len(v) for v in self.sources.values())

    @property
    def num_globals(self) -> int:
        return sum(len(v) for v in self.globals_.values())

    @property
    def size(self) -> int:
        return self.num_sources + self.num_globals

    def merge(self, new: "DependencyState") -> "DependencyState":
        # todo: more careful implementation may be better. For now, just
        # accumulate state
        diff = self.diff(new=new)
        result = copy.deepcopy(self)
        for module, func_sources in diff.changed_sources.items():
            for k, v in func_sources.items():
                result.sources[module][k] = v.new
        for module, func_sources in diff.new_sources.items():
            for k, v in func_sources.items():
                result.sources.setdefault(module, {})[k] = v
        for module, globals_ in diff.changed_globals.items():
            for k, v in globals_.items():
                result.globals_[module][k] = v.new
        for module, globals_ in diff.new_globals.items():
            for k, v in globals_.items():
                result.globals_.setdefault(module, {})[k] = v
        for edge in new.call_edges:
            if edge not in result.call_edges:
                result.call_edges.append(edge)
        for edge in new.globals_edges:
            if edge not in result.globals_edges:
                result.globals_edges.append(edge)
        return result

    @staticmethod
    def recover(
        old: "DependencyState", ignore_errors: bool = False
    ) -> Tuple["DependencyState", "DepsDiff"]:
        """
        Given an old dependency state, recover what's possible, and return a new
        dependency state and a diff from the old to the new.

        Note that the diff will not contain any "new" items by construction.
        """
        sources, globals_ = {}, {}
        ### recover sources
        for module_name, func_sources in old.sources.items():
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            sources[module_name] = {}
            for qualified_func_name, old_source in func_sources.items():
                parts = qualified_func_name.split(".")
                current = module
                found = True
                for part in parts:
                    if not hasattr(current, part):
                        found = False
                        break
                    else:
                        current = getattr(current, part)
                if not found:
                    msg = f"Could not find function {qualified_func_name} from module {module_name}."
                    if ignore_errors:
                        logging.warning(msg)
                    else:
                        raise RuntimeError(msg)
                if inspect.isfunction(current):
                    if old_source.is_op:
                        logging.warning(
                            f"Function {qualified_func_name} from module {module_name} is not an op, but was marked as such in the old dependency state."
                        )
                    is_op, func_obj = False, current
                elif type(current).__name__ == "FuncInterface":  #! a hack
                    if not old_source.is_op:
                        logging.warning(
                            f"Function {qualified_func_name} from module {module_name} is an op, but was not marked as such in the old dependency state."
                        )
                    is_op, func_obj = True, current.func_op.func
                else:
                    msg = f"{qualified_func_name} from module {module_name} used to be a function, but is now neither a function nor an op."
                    if ignore_errors:
                        logging.warning(msg)
                        continue
                    else:
                        raise RuntimeError(msg)
                try:
                    source = inspect.getsource(func_obj)
                except Exception as e:
                    msg = f"Unable to recover source for {qualified_func_name} from module {module_name}."
                    if ignore_errors:
                        logging.warning(msg)
                        continue
                    else:
                        raise RuntimeError(msg) from e
                source_representation = FuncSource(
                    source=source, is_op=is_op, qualified_name=qualified_func_name
                )
                sources[module_name][qualified_func_name] = source_representation
        ### recover globals
        for module_name, old_globals in old.globals_.items():
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                msg = f"Could not find module {module_name} from old dependency state."
                if ignore_errors:
                    logging.warning(msg)
                    continue
                else:
                    raise RuntimeError(msg)
            globals_[module_name] = {}
            for global_name, _ in old_globals.items():
                if not hasattr(module, global_name):
                    msg = f"Could not find global variable {global_name} from module {module_name} from old dependency state."
                    if ignore_errors:
                        logging.warning(msg)
                        continue
                    else:
                        raise RuntimeError(msg)
                try:
                    content_hash = Hashing.get_content_hash(
                        getattr(module, global_name)
                    )
                except:
                    msg = f"Unable to recover content hash for global variable {global_name} from module {module_name}."
                    if ignore_errors:
                        logging.warning(msg)
                        continue
                    else:
                        raise RuntimeError(msg)
                try:
                    repr_ = repr(getattr(module, global_name))
                except Exception as e:
                    repr_ = None
                globals_[module_name][global_name] = GlobalVar(
                    name=global_name,
                    content_hash=content_hash,
                    repr_=repr_,
                )
        recovered = DependencyState(roots=old.roots, sources=sources, globals_=globals_)
        diff = old.diff(new=recovered)
        return recovered, diff

    def diff(self, new: "DependencyState") -> "DepsDiff":
        new_sources = _nested_dict_subtract(new.sources, self.sources)
        removed_sources = _nested_dict_subtract(self.sources, new.sources)
        changed_sources = _nested_dict_diff(
            new=new.sources, old=self.sources, differ=FuncSourceDiff
        )
        new_globals = _nested_dict_subtract(new.globals_, self.globals_)
        removed_globals = _nested_dict_subtract(self.globals_, new.globals_)
        changed_globals = _nested_dict_diff(
            new=new.globals_, old=self.globals_, differ=GlobalVarDiff
        )
        return DepsDiff(
            new_sources=new_sources,
            removed_sources=removed_sources,
            changed_sources=changed_sources,
            new_globals=new_globals,
            removed_globals=removed_globals,
            changed_globals=changed_globals,
        )

    def to_dot(self) -> str:
        nodes: Dict[
            Tuple[str, str], Node
        ] = {}  # (module name, object name) -> Node object
        edges: Dict[Tuple[Node, Node], Edge] = {}  # (Node, Node) -> Edge object
        module_groups: Dict[str, Group] = {}  # module name -> Group
        class_groups: Dict[str, Group] = {}  # class name -> Group
        for module_name, module_globals in self.globals_.items():
            if module_name not in module_groups:
                module_groups[module_name] = Group(
                    label=module_name, nodes=[], parent=None
                )
            for global_name, global_var in module_globals.items():
                node_name = f"{module_name}.{global_name}"
                node = Node(
                    internal_name=node_name,
                    label=global_name,
                    color=SOLARIZED_LIGHT["red"],
                )
                nodes[(module_name, global_name)] = node
                module_groups[module_name].nodes.append(node)
        for module_name, module_sources in self.sources.items():
            if module_name not in module_groups:
                module_groups[module_name] = Group(
                    label=module_name, nodes=[], parent=None
                )
            for func_name, func_source in module_sources.items():
                node_name = f"{module_name}.{func_name}"
                if func_source.is_method:
                    color = SOLARIZED_LIGHT["violet"]
                else:
                    color = SOLARIZED_LIGHT["blue"]
                node = Node(internal_name=node_name, label=func_name, color=color)
                nodes[(module_name, func_name)] = node
                module_groups[module_name].nodes.append(node)
                if func_source.is_method:
                    class_name = func_source.class_name
                    class_groups.setdefault(
                        class_name,
                        Group(
                            label=class_name,
                            nodes=[],
                            parent=module_groups[module_name],
                        ),
                    ).nodes.append(node)
        for call_edge in self.call_edges:
            caller_node = nodes[
                (call_edge.caller.module_name, call_edge.caller.qualified_func_name)
            ]
            callee_node = nodes[
                (call_edge.callee.module_name, call_edge.callee.qualified_func_name)
            ]
            edge = Edge(source_node=caller_node, target_node=callee_node)
            edges[(caller_node, callee_node)] = edge
        for global_edge in self.globals_edges:
            caller_node = nodes[
                (global_edge.caller.module_name, global_edge.caller.qualified_func_name)
            ]
            global_node = nodes[
                (global_edge.caller.module_name, global_edge.global_var.name)
            ]
            edge = Edge(source_node=caller_node, target_node=global_node)
            edges[(caller_node, global_node)] = edge
        dot_string = to_dot_string(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            groups=list(module_groups.values()) + list(class_groups.values()),
            rankdir="BT",
        )
        return dot_string


################################################################################
### a model of dependency diffs
################################################################################
class FuncSourceDiff:
    def __init__(self, new: FuncSource, old: FuncSource):
        self.new = new
        self.old = old

    def to_string(self) -> str:
        return _get_colorized_diff(
            current=self.old.to_string(), new=self.new.to_string()
        )


class GlobalVarDiff:
    def __init__(self, new: GlobalVar, old: GlobalVar):
        self.new = new
        self.old = old

    def to_string(self) -> str:
        return "\n" + _get_colorized_diff(
            current=self.old.to_string(), new=self.new.to_string()
        )


class DepsDiff:
    def __init__(
        self,
        new_sources: Dict[str, Dict[str, FuncSource]],
        removed_sources: Dict[str, Dict[str, FuncSource]],
        changed_sources: Dict[str, Dict[str, FuncSourceDiff]],
        new_globals: Dict[str, Dict[str, GlobalVar]],
        removed_globals: Dict[str, Dict[str, GlobalVar]],
        changed_globals: Dict[str, Dict[str, GlobalVarDiff]],
    ):
        self.new_sources = new_sources
        self.removed_sources = removed_sources
        self.changed_sources = changed_sources
        self.new_globals = new_globals
        self.removed_globals = removed_globals
        self.changed_globals = changed_globals

    @property
    def is_empty(self) -> bool:
        return (
            len(self.new_sources) == 0
            and len(self.removed_sources) == 0
            and len(self.changed_sources) == 0
            and len(self.new_globals) == 0
            and len(self.removed_globals) == 0
            and len(self.changed_globals) == 0
        )

    @staticmethod
    def get_readable_description(diff: "DepsDiff") -> str:
        result_parts = []
        ### process new dependencies
        if len(diff.new_globals) > 0 or len(diff.new_sources) > 0:
            raise NotImplementedError()
        ### process missing dependencies
        if len(diff.removed_globals) > 0 or len(diff.removed_sources) > 0:
            preamble_missing = "===THE FOLLOWING DEPENDENCIES COULD NOT BE RECOVERED==="
            missing_sources_text = {
                k: {k2: "" for k2, v2 in v.items()}
                for k, v in diff.removed_sources.items()
            }
            missing_globals_text = {
                k: {k2: "" for k2, v2 in v.items()}
                for k, v in diff.removed_globals.items()
            }
            missing_desc = _pprint_structure(
                globals_text=missing_globals_text, sources_text=missing_sources_text
            )
            result_parts += [preamble_missing, missing_desc]
        if len(diff.changed_globals) > 0 or len(diff.changed_sources) > 0:
            ### process changes
            preamble_changed = "===THE FOLLOWING DEPENDENCIES HAVE CHANGED==="
            changed_sources_text = {
                k: {k2: v2.to_string() for k2, v2 in v.items()}
                for k, v in diff.changed_sources.items()
            }
            changed_globals_text = {
                k: {k2: v2.to_string() for k2, v2 in v.items()}
                for k, v in diff.changed_globals.items()
            }
            changed_desc = _pprint_structure(
                globals_text=changed_globals_text, sources_text=changed_sources_text
            )
            result_parts += [preamble_changed, changed_desc]
        return "\n".join(result_parts)

    def __repr__(self) -> str:
        return self.get_readable_description(self)


################################################################################
### tracer
################################################################################
class DependencyTracer:
    """
    Attempt to collect the source code of all functions called within this
    context, as well as the content hashes and `repr()` of global variables
    *contained in the source code* of these functions.
    """

    def __init__(self, ds: DependencyState = None, ignore_errors: bool = False):
        self.ds = (
            ds if ds is not None else DependencyState(roots=[], sources={}, globals_={})
        )
        self.ignore_errors = ignore_errors
        self.call_stack: List[Call] = []

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError("Another tracer is already active")
        # Create a new tracer function that records the calls
        def tracer(frame, event, arg):
            if event == "return":
                if len(self.call_stack) > 0:
                    self.call_stack.pop()
            if event != "call":
                return

            # qualified name of the module where the function/method is defined.
            def_module_name = frame.f_globals.get("__name__")
            code = frame.f_code  # code object of function being called
            func_name = code.co_name  # function's name
            if func_name == "<lambda>":
                logging.warning("Cannot handle lambdas")
                call = Call(is_tracked=False)
                self.call_stack.append(call)
                return tracer  #! continue tracing in the lambda
            # ? if func_name == '<module>':
            # ?     self.calls.append(f'{code.co_filename}.{func_name}')

            ### figure out the module we're importing from
            try:
                # case 1: module is a file
                module = importlib.import_module(def_module_name)
                module_path = Path(inspect.getfile(module))
                assert module_path.is_absolute()
                if len(self.ds.roots) != 0:
                    if not any(root in module_path.parents for root in self.ds.roots):
                        # module is not in the paths we're inspecting; stop tracing
                        return
                    elif def_module_name.startswith(
                        Config.module_name
                    ) and not def_module_name.startswith(Config.tests_module_name):
                        # this function is part of `mandala` functionality. Continue tracing
                        # but don't add it to the dependency state
                        collect_dependency = False
                    else:
                        collect_dependency = True
                else:
                    collect_dependency = True
            except:
                # case 2: module is not a file, e.g. it is a jupyter notebook
                logging.debug(f"Module {def_module_name} is not a file")
                if def_module_name != "__main__":
                    raise NotImplementedError()
                collect_dependency = True

            if not collect_dependency:
                call = Call(is_tracked=False)
                self.call_stack.append(call)
                return tracer

            ####################################################################
            ### detect use of closure variables
            ####################################################################
            closure = frame.f_code.co_freevars
            if closure:
                closure_values = {
                    var: frame.f_locals.get(var, frame.f_globals.get(var, None))
                    for var in closure
                }
                msg = f"Found closure variables accessed by function {def_module_name}.{func_name}:\n{closure_values}"
                if self.ignore_errors:
                    logging.warning(msg)
                else:
                    raise RuntimeError(msg)

            ####################################################################
            ### get the global variables used by the function
            ####################################################################
            this_call_globals = []
            if def_module_name not in self.ds.globals_:
                self.ds.globals_[def_module_name] = {}
            for (
                name
            ) in (
                code.co_names
            ):  # names used by the function; not all of them are global variables
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
                    try:
                        content_hash = Hashing.get_content_hash(global_val)
                    except Exception as e:
                        msg = f"Found global variable '{name}' accessed from module '{def_module_name}' of type {type(global_val)} that cannot be pickled"
                        if self.ignore_errors:
                            logging.warning(msg)
                            content_hash = None
                        else:
                            raise RuntimeError(msg)
                    global_var = GlobalVar(
                        name=name, content_hash=content_hash, repr_=repr(global_val)
                    )
                    if name in self.ds.globals_[def_module_name]:
                        if global_var != self.ds.globals_[def_module_name][name]:
                            msg = f"Global variable {name} accessed from module '{def_module_name}' has changed its value. This is likely due to a global variable being modified by a function call."
                            if self.ignore_errors:
                                logging.warning(msg)
                            else:
                                raise RuntimeError(msg)
                    self.ds.globals_[def_module_name][name] = global_var
                    this_call_globals.append(global_var)

            ####################################################################
            ### get the qualified name of the function/method
            ####################################################################
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
            if def_module_name not in self.ds.sources:
                self.ds.sources[def_module_name] = {}

            ####################################################################
            ### get the function source
            ####################################################################
            try:
                # source code is not always available
                func_source = inspect.getsource(code)
                if self.ds.num_sources == 0:
                    #! a hack
                    # this is the first function we're tracing, i.e. an op under the
                    # current implementation.
                    source = FuncSource(
                        source=func_source,
                        is_op=True,
                        qualified_name=func_qualname,
                    )
                else:
                    source = FuncSource(
                        source=func_source, is_op=False, qualified_name=func_qualname
                    )
            except OSError:
                source = FuncSource(
                    source=None, is_op=None, qualified_name=func_qualname
                )
            if func_qualname in self.ds.sources[def_module_name]:
                if source != self.ds.sources[def_module_name][func_qualname]:
                    logging.warning(
                        f"Function {func_qualname} has changed its source code or op status. This may cause dependency tracking to fail."
                    )
            self.ds.sources[def_module_name][func_qualname] = source

            ####################################################################
            ### manage the call stack
            ####################################################################
            call = Call(
                is_tracked=True,
                module_name=def_module_name,
                qualified_func_name=func_qualname,
            )
            ### global variable edges from this function always exist
            globals_edges = [
                GlobalEdge(caller=call, global_var=global_var)
                for global_var in this_call_globals
            ]
            self.ds.globals_edges += globals_edges
            ### call edges exist only if there is a caller on the stack
            if len(self.call_stack) > 0:
                # find the most recent tracked call
                i = None
                for i in range(len(self.call_stack) - 1, -1, -1):
                    if self.call_stack[i].is_tracked:
                        break
                if i is not None:
                    caller = self.call_stack[i]
                    call_edge = CallEdge(caller=caller, callee=call)
                    self.ds.call_edges.append(call_edge)
            self.call_stack.append(call)
            return tracer

        sys.settrace(tracer)

    def __exit__(self, *exc_info):
        sys.settrace(None)  # Stop tracing
