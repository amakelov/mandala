import textwrap
from abc import abstractmethod, ABC
import types

from ..common_imports import *
from ..utils import get_content_hash
from ..viz import (
    write_output,
)
from ..model import Ref

from .utils import (
    DepKey,
    load_obj,
    get_runtime_description,
    extract_code,
    unknown_function,
    UNKNOWN_GLOBAL_VAR,
)


class Node(ABC):
    def __init__(self, module_name: str, obj_name: str, representation: Any):
        self.module_name = module_name
        self.obj_name = obj_name
        self.representation = representation

    @property
    def key(self) -> DepKey:
        return (self.module_name, self.obj_name)

    def present_key(self) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def represent(obj: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def content(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def readable_content(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def content_hash(self) -> str:
        raise NotImplementedError

    def load_obj(self, skip_missing: bool, skip_silently: bool) -> Any:
        obj, found = load_obj(module_name=self.module_name, obj_name=self.obj_name)
        if not found:
            msg = f"{self.present_key()} not found"
            if skip_missing:
                if skip_silently:
                    logger.debug(msg)
                else:
                    logger.warning(msg)
                if hasattr(self, "FALLBACK_OBJ"):
                    return self.FALLBACK_OBJ
                else:
                    raise ValueError(f"No fallback object defined for {self.__class__}")
            else:
                raise ValueError(msg)
        return obj


class CallableNode(Node):
    FALLBACK_OBJ = unknown_function

    def __init__(
        self,
        module_name: str,
        obj_name: str,
        representation: Optional[str],
        runtime_description: str,
    ):
        self.module_name = module_name
        self.obj_name = obj_name
        self.runtime_description = runtime_description
        if representation is not None:
            self._set_representation(value=representation)
        else:
            self._representation = None
            self._content_hash = None

    @staticmethod
    def from_obj(obj: Any, dep_key: DepKey) -> "CallableNode":
        representation = CallableNode.represent(obj=obj)
        code_obj = extract_code(obj)
        runtime_description = get_runtime_description(code=code_obj)
        return CallableNode(
            module_name=dep_key[0],
            obj_name=dep_key[1],
            representation=representation,
            runtime_description=runtime_description,
        )

    @staticmethod
    def from_runtime(
        module_name: str,
        obj_name: str,
        code_obj: types.CodeType,
    ) -> "CallableNode":
        return CallableNode(
            module_name=module_name,
            obj_name=obj_name,
            representation=None,
            runtime_description=get_runtime_description(code=code_obj),
        )

    @property
    def representation(self) -> str:
        return self._representation

    def _set_representation(self, value: str):
        assert isinstance(value, str)
        self._representation = value
        self._content_hash = get_content_hash(value)

    @representation.setter
    def representation(self, value: str):
        self._set_representation(value)

    @property
    def is_method(self) -> bool:
        return "." in self.obj_name

    def present_key(self) -> str:
        return f"function {self.obj_name} from module {self.module_name}"

    @property
    def class_name(self) -> str:
        assert self.is_method
        return ".".join(self.obj_name.split(".")[:-1])

    @staticmethod
    def represent(
        obj: Union[types.FunctionType, types.CodeType, Callable],
        allow_fallback: bool = False,
    ) -> str:
        if type(obj).__name__ == "Op":
            obj = obj.f
        if not isinstance(obj, (types.FunctionType, types.MethodType, types.CodeType)):
            logger.warning(f"Found {obj} of type {type(obj)}")
        try:
            source = inspect.getsource(obj)
        except Exception as e:
            msg = f"Could not get source for {obj} because {e}"
            if allow_fallback:
                source = inspect.getsource(CallableNode.FALLBACK_OBJ)
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        # strip whitespace to prevent different sources looking the same in the
        # ui
        lines = source.splitlines()
        lines = [line.rstrip() for line in lines]
        source = "\n".join(lines)
        return source

    def content(self) -> str:
        return self.representation

    def readable_content(self) -> str:
        return self.representation

    @property
    def content_hash(self) -> str:
        assert isinstance(self._content_hash, str)
        return self._content_hash


class GlobalVarNode(Node):
    FALLBACK_OBJ = UNKNOWN_GLOBAL_VAR

    def __init__(
        self,
        module_name: str,
        obj_name: str,
        # (content hash, truncated repr)
        representation: Tuple[str, str],
    ):
        self.module_name = module_name
        self.obj_name = obj_name
        self._representation = representation

    @staticmethod
    def from_obj(obj: Any, dep_key: DepKey, 
                 skip_unhashable: bool = False,
                 skip_silently: bool = False,) -> "GlobalVarNode":
        representation = GlobalVarNode.represent(obj=obj, skip_unhashable=skip_unhashable, skip_silently=skip_silently)
        return GlobalVarNode(
            module_name=dep_key[0],
            obj_name=dep_key[1],
            representation=representation,
        )

    @property
    def representation(self) -> Tuple[str, str]:
        return self._representation

    @staticmethod
    def represent(obj: Any, skip_unhashable: bool = True, 
                  skip_silently: bool = False,
                  ) -> Tuple[str, str]:
        """
        Return a hash of this global variable's value + a truncated
        representation useful for debugging/printing. 

        If `obj` is a `Ref`, the content hash is reused from the `Ref` object. 
        This is so that you can avoid repeatedly hashing the same (potentially
        large) object any time the code state needs to be synced. 
        """
        truncated_repr = textwrap.shorten(text=repr(obj), width=80)
        if isinstance(obj, Ref):
            content_hash = obj.cid
        else:
            try:
                content_hash = get_content_hash(obj=obj)
            except Exception as e:
                shortened_exception = textwrap.shorten(text=str(e), width=80)
                msg = f"Failed to hash global variable {truncated_repr} of type {type(obj)}, because {shortened_exception}"
                if skip_unhashable:
                    content_hash = UNKNOWN_GLOBAL_VAR
                    if skip_silently:
                        logger.debug(msg)
                    else:
                        logger.warning(msg)
                else:
                    raise RuntimeError(msg)
        return content_hash, truncated_repr

    def present_key(self) -> str:
        return f"global variable {self.obj_name} from module {self.module_name}"

    def content(self) -> str:
        return self.representation

    def readable_content(self) -> str:
        return self.representation[1]

    @property
    def content_hash(self) -> str:
        assert isinstance(self.representation, tuple)
        return self.representation[0]


class TerminalData:
    def __init__(
        self,
        op_internal_name: str,
        op_version: int,
        call_content_version: str,
        call_semantic_version: str,
        # data: Tuple[Tuple[str, int], Tuple[str, str]],
        dep_key: DepKey,
    ):
        # ((internal name, version), (content_version, semantic_version))
        self.op_internal_name = op_internal_name
        self.op_version = op_version
        self.call_content_version = call_content_version
        self.call_semantic_version = call_semantic_version
        self.dep_key = dep_key


class TerminalNode(Node):
    def __init__(self, module_name: str, obj_name: str, representation: TerminalData):
        self.module_name = module_name
        self.obj_name = obj_name
        self.representation = representation

    @property
    def key(self) -> DepKey:
        return self.module_name, self.obj_name

    def present_key(self) -> str:
        raise NotImplementedError

    @property
    def content_hash(self) -> str:
        raise NotImplementedError

    def content(self) -> Any:
        raise NotImplementedError

    def readable_content(self) -> str:
        raise NotImplementedError

    @staticmethod
    def represent(obj: Any) -> Any:
        raise NotImplementedError


class DependencyGraph:
    def __init__(self):
        self.nodes: Dict[DepKey, Node] = {}
        self.roots: Set[DepKey] = set()
        self.edges: Set[Tuple[DepKey, DepKey]] = set()

    def get_trace_state(self) -> Tuple[DepKey, Dict[DepKey, Node]]:
        if len(self.roots) != 1:
            raise ValueError(f"Expected exactly one root, got {len(self.roots)}")
        component = list(self.roots)[0]
        return component, self.nodes

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
            self.add_node(source)
        if target.key not in self.nodes:
            self.add_node(target)
        self.edges.add((source.key, target.key))


from .viz import to_dot, to_string
