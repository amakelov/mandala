from ..common_imports import *
from ..core.utils import Hashing
import sys
import inspect
import importlib
import textwrap
from pickle import PicklingError

NestedDictType = Dict[str, Dict[str, Any]]


def _nested_dict_subtract(x: NestedDictType, y: NestedDictType) -> NestedDictType:
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


class NestedDictDiff:
    def __init__(
        self, new: NestedDictType, removed: NestedDictType, changed: NestedDictType
    ):
        self.new = new
        self.removed = removed
        self.changed = changed

    def __repr__(self) -> str:
        return f"NestedDictDiff(new={self.new}, removed={self.removed}, changed={self.changed})"

    @property
    def is_empty(self) -> bool:
        return not self.new and not self.removed and not self.changed


def _nested_dict_diff(x: NestedDictType, y: NestedDictType) -> NestedDictDiff:
    """
    Return three dicts:
        - one that contains the keys in `x` that are not in `y`,
        - one that contains the keys in `y` not in `x`
        - one with the keys in `x` and `y` that have different values.
    """
    x_only = _nested_dict_subtract(x=x, y=y)
    y_only = _nested_dict_subtract(x=y, y=x)
    different_vals: NestedDictType = {}
    for k, v in x.items():
        if k not in y:
            continue
        for k2, v2 in v.items():
            if k2 not in y[k]:
                continue
            if v2 != y[k][k2]:
                different_vals.setdefault(k, {})[k2] = v2
    return NestedDictDiff(new=x_only, removed=y_only, changed=different_vals)


class DepsDiff:
    def __init__(self, sources: NestedDictDiff, globals_: NestedDictDiff):
        self.sources = sources
        self.globals_ = globals_


class DependencyState:
    def __init__(
        self,
        root: Optional[Path] = None,
        sources: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        globals_: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    ):
        self.root = root if root is None else root.absolute().resolve()
        # module name -> func address in module -> func source, if available
        self.sources: Dict[str, Dict[str, Optional[str]]] = (
            {} if sources is None else sources
        )
        # qualified module name -> global variable name -> content hash
        self.globals_: Dict[str, Dict[str, Optional[str]]] = (
            {} if globals_ is None else globals_
        )

    def __repr__(self) -> str:
        all_modules = set(self.sources.keys()).union(self.globals_.keys())
        lines = []
        for module in all_modules:
            # add the module name to the lines, with some decoration around
            lines.append(f"Module {module}:")
            # add the global variable names, one per line (if any), indented 4
            # spaces
            if module in self.globals_:
                for global_name in self.globals_[module]:
                    lines.append(f"    {global_name}")
            # same for function sources
            if module in self.sources:
                for func_name, source in self.sources[module].items():
                    lines.append(f"    {func_name}")
                    if source is None:
                        to_show = "<source not available>"
                    else:
                        to_show = source
                    lines += [f"        {line}" for line in to_show.splitlines()]
            # add a blank line between modules
            lines.append("")
        return "\n".join(lines)

    def num_sources(self) -> int:
        return sum(len(v) for v in self.sources.values())

    def diff(self, new: "DependencyState") -> DepsDiff:
        """
        Return two NestedDictDiffs:
            - one for the sources
            - one for the globals
        """
        return DepsDiff(
            sources=_nested_dict_diff(self.sources, new.sources),
            globals_=_nested_dict_diff(self.globals_, new.globals_),
        )

    def update(self, new: "DependencyState") -> "DependencyState":
        # todo: more careful implementation may be better
        return new

    @staticmethod
    def remove_function_signature(source: str) -> str:
        """
        Given the source code of a function, remove the part that contains the
        function signature.

        Has the extra effect of removing comments and docstrings.
        """
        # using dedent is necessary here to handle decorators
        tree = ast.parse(textwrap.dedent(source))
        assert isinstance(tree, ast.Module)
        body = tree.body
        assert len(body) == 1
        assert isinstance(body[0], ast.FunctionDef)
        func_body = body[0].body
        return ast.unparse(func_body)

    @staticmethod
    def recover(old: "DependencyState") -> Tuple["DependencyState", DepsDiff]:
        """
        Given an old dependency state, recover what's possible, and return a new
        dependency state and a diff from the old to the new.

        Note that the diff will not contain any "new" items by construction.
        """
        calls, globals_ = {}, {}
        for module_name, func_sources in old.sources.items():
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            calls[module_name] = {}
            for qualified_func_name, _ in func_sources.items():
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
                    continue
                if inspect.isfunction(current):
                    source = inspect.getsource(current)
                elif type(current).__name__ == "FuncInterface":
                    #! a hack
                    source = inspect.getsource(current.func_op.func)
                calls[module_name][qualified_func_name] = source
        for module_name, globals in old.globals_.items():
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            globals_[module_name] = {}
            for global_name, _ in globals.items():
                if not hasattr(module, global_name):
                    continue
                globals_[module_name][global_name] = Hashing.get_content_hash(
                    getattr(module, global_name)
                )
        recovered = DependencyState(root=old.root, sources=calls, globals_=globals_)
        diff = old.diff(new=recovered)
        return recovered, diff


class DependencyTracer:
    """
    Collects the source code of all functions called within this context, as
    well as the content hashes of global variables *contained in the source code*
    of these functions.

    Limitations:
        - Does not handle lambdas
        - Does not handle nested functions
        - Does not detect which global variables are used by the functions, only
        which global variables are referred to by name in the functions' code.
        So it may overestimate the set of global variables accessed by a given
        call.
    """

    def __init__(self, ds: DependencyState = None):
        """
        Args:
            root (Path): the root directory of the user's codebase. Used to
            limit the scope of the tracer.
        """
        self.ds = (
            ds
            if ds is not None
            else DependencyState(root=None, sources={}, globals_={})
        )
        # self.root = root
        # self.calls: Dict[str, Optional[str]] = {} # qualified function/method name -> source code, if available
        # self.globals_ :Dict[str, Dict[str, str]] = {} # qualified module name -> global variable name -> content hash

    def __enter__(self):
        if self.ds.root is None:
            return
        if sys.gettrace() is not None:
            raise RuntimeError("Another tracer is already active")
        # Create a new tracer function that records the calls
        def tracer(frame, event, arg):
            if event != "call":
                return
            # qualified name of the module where the function/method is
            # defined.
            def_module_name = frame.f_globals.get("__name__")

            ### figure out the module we're importing from
            try:
                # case 1: module is a file
                module = importlib.import_module(def_module_name)
                module_path = Path(inspect.getfile(module))
                assert module_path.is_absolute()
                if self.ds.root is not None:
                    if not self.ds.root in module_path.parents:
                        # module is not in the user's codebase; stop tracing
                        return
            except:
                # case 2: module is not a file, e.g. it is a jupyter notebook
                logging.debug(f"Module {def_module_name} is not a file")
                if def_module_name != "__main__":
                    raise NotImplementedError()
            code = frame.f_code  # code object of function being called
            func_name = code.co_name  # function's name
            try:
                # source code is not always available
                func_source = inspect.getsource(code)
            except OSError:
                func_source = None

            if func_name == "<lambda>":
                logging.warning("Cannot handle lambdas")
                return tracer  #! continue tracing in the lambda
            # ? if func_name == '<module>':
            # ?     self.calls.append(f'{code.co_filename}.{func_name}')

            ####################################################################
            ### get the global variables used by the function
            ####################################################################
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
                    ):
                        # ignore modules, classes and functions
                        continue
                    try:
                        content_hash = Hashing.get_content_hash(global_val)
                    except PicklingError:
                        logging.warning(
                            f"Found global variable of type {type(global_val)} that cannot be pickled"
                        )
                        content_hash = None
                    except Exception as e:
                        logging.warning(
                            f"Found global variable of type {type(global_val)} that cannot be hashed"
                        )
                        content_hash = None
                    if name in self.ds.globals_[def_module_name]:
                        if content_hash != self.ds.globals_[def_module_name][name]:
                            logging.warning(
                                f"Global variable {name} has changed its value"
                            )
                    if name in self.ds.globals_[def_module_name]:
                        if content_hash != self.ds.globals_[def_module_name][name]:
                            logging.warning(
                                f"Global variable {name} has changed its value. This may cause dependency tracking to fail."
                            )
                    self.ds.globals_[def_module_name][name] = content_hash

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
            if func_qualname in self.ds.sources[def_module_name]:
                if func_source != self.ds.sources[def_module_name][func_qualname]:
                    logging.warning(
                        f"Function {func_qualname} has changed its source code! This may cause dependency tracking to fail."
                    )
            if self.ds.num_sources() == 0:
                #! a hack
                # this is the first function we're tracing, i.e. an op.
                if func_source is not None:
                    func_source = DependencyState.remove_function_signature(
                        source=func_source
                    )
            self.ds.sources[def_module_name][func_qualname] = func_source
            return tracer

        sys.settrace(tracer)

    def __exit__(self, *exc_info):
        if self.ds.root is None:
            return
        sys.settrace(None)  # Stop tracing
