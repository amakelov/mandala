from .engine import map_it, get_sources_and_language
from .model import Node, Edge, GROUP_TYPE
from ..common_imports import *


class DependencyGraph:
    def __init__(self, all_nodes: List[Node], edges: List[Edge]):
        self.all_nodes = all_nodes
        self.edges = edges
        self.nodes_by_qualname = {}
        # precompute the set of nodes with a given qualname
        for node in self.all_nodes:
            qual_name = DependencyGraph.get_node_qualified_name(node=node)
            if qual_name in self.nodes_by_qualname:
                filename, func_path_in_file = qual_name.split(".")
                logging.warning(
                    f"Duplicate function name found in file {filename}.py: {func_path_in_file}"
                )
            else:
                self.nodes_by_qualname[qual_name] = []
            self.nodes_by_qualname[qual_name].append(node)
        # precompute the set of edges from each node
        self.edges_by_tail = {}
        for edge in self.edges:
            if edge.node0 not in self.edges_by_tail:
                self.edges_by_tail[edge.node0] = []
            self.edges_by_tail[edge.node0].append(edge)

    @staticmethod
    def get_node_qualified_name(node: Node) -> str:
        result = []
        current = node
        while current is not None:
            result.append(current.token)
            current = current.parent
        return ".".join(reversed(result))

    @staticmethod
    def from_code2flow(root: Path):
        sources, _ = get_sources_and_language(
            raw_source_paths=[str(root.absolute())], language="py"
        )
        file_groups, all_nodes, edges = map_it(sources=sources, extension="py")
        return DependencyGraph(all_nodes=all_nodes, edges=edges)

    @staticmethod
    def get_node_globals_qualname(node: Node) -> str:
        """
        Try to get the name a function should have if it were defined in the
        global scope.
        """
        if node.parent is None or node.parent.group_type == GROUP_TYPE.FILE:
            return node.token
        elif node.parent.group_type == GROUP_TYPE.CLASS:
            if node.parent.parent is not None:
                if node.parent.parent.group_type != GROUP_TYPE.FILE:
                    raise NotImplementedError("Cannot handle nested classes")
            return ".".join([node.parent.token, node.token])
        else:
            raise NotImplementedError()

    @staticmethod
    def find_func_globals(f: Callable) -> Dict[str, Any]:
        globals_ = f.__globals__
        names = f.__code__.co_names
        result = {}
        for name in names:
            if name in globals_:
                obj = globals_[name]
                if isinstance(obj, type) or inspect.isfunction(obj):
                    # ignore classes and functions: they should be accounted for
                    # by the `get_dependencies` method
                    continue
                result[name] = obj
        return result

    def get_dependencies(
        self, f: Callable
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        Given a *function* object (not a method), attempt to provide a list of
        all the functions that it depends on, as well as find the objects
        corresponding to these functions/methods by following pointers in
        `__globals__`. Complain about duplicate names.
        """
        if inspect.ismethod(f):
            raise NotImplementedError("get_dependencies does not support methods")
        if inspect.isclass(f):
            raise NotImplementedError("get_dependencies does not support classes")
        if not inspect.isfunction(f):
            raise ValueError("get_dependencies only supports functions")
        # find the name of the file where this function is defined
        filename = Path(inspect.getfile(f)).stem
        qualified_name = ".".join([filename, f.__name__])
        if qualified_name not in self.nodes_by_qualname:
            raise ValueError(f"Could not find {qualified_name} in the dependency graph")
        starting_nodes = self.nodes_by_qualname[qualified_name]
        if len(starting_nodes) > 1:
            logging.warning(
                f"Found multiple nodes for {qualified_name} in the dependency graph"
            )

        nodes_found = set(starting_nodes)
        # dynamic objects corresponding to the functions/methods we find
        node_to_obj: Dict[str, Any] = {qualified_name: f}
        found_new = True
        while found_new:
            new_nodes = []
            for node in nodes_found:
                node_qualified_name = DependencyGraph.get_node_qualified_name(node=node)
                node_obj = node_to_obj.get(node_qualified_name, None)
                edges = self.edges_by_tail.get(node, [])
                new_nodes = []
                for edge in edges:
                    dependency = edge.node1
                    if node_obj is not None:
                        # find the object that this dependency refers to
                        dependency_global_qualname = (
                            DependencyGraph.get_node_globals_qualname(dependency)
                        )
                        parts = dependency_global_qualname.split(".")
                        globals_ = node_obj.__globals__
                        current = globals_
                        for part in parts:
                            if isinstance(current, dict):
                                current = current.get(part)
                            else:
                                current = getattr(current, part)
                        node_to_obj[
                            DependencyGraph.get_node_qualified_name(node=dependency)
                        ] = current
                    if dependency not in nodes_found:
                        new_nodes.append(dependency)
            if len(new_nodes) == 0:
                found_new = False
            else:
                nodes_found = nodes_found.union(new_nodes)
        result = {
            DependencyGraph.get_node_qualified_name(node=node): node.source_code
            for node in nodes_found
        }
        return result, node_to_obj
