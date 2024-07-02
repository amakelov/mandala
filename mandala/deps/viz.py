import textwrap
from ..common_imports import *
from ..viz import (
    Node as DotNode,
    Edge as DotEdge,
    Group as DotGroup,
    to_dot_string,
    SOLARIZED_LIGHT,
)

from .utils import (
    DepKey,
)


def to_string(graph: "model.DependencyGraph") -> str:
    """
    Get a string for pretty-printing.
    """
    # group the nodes by module
    module_groups: Dict[str, List["model.Node"]] = {}
    for key, node in graph.nodes.items():
        module_name, _ = key
        module_groups.setdefault(module_name, []).append(node)
    lines = []
    for module_name, nodes in module_groups.items():
        global_nodes = [node for node in nodes if isinstance(node, model.GlobalVarNode)]
        callable_nodes = [
            node for node in nodes if isinstance(node, model.CallableNode)
        ]
        module_desc = f"MODULE: {module_name}"
        lines.append(module_desc)
        lines.append("-" * len(module_desc))
        lines.append("===Global Variables===")
        for node in global_nodes:
            desc = f"{node.obj_name} = {node.readable_content()}"
            lines.append(textwrap.indent(desc, 4 * " "))
            # lines.append(f"  {node.diff_representation()}")
        lines.append("")
        lines.append("===Functions===")
        # group the methods by class
        method_nodes = [node for node in callable_nodes if node.is_method]
        func_nodes = [node for node in callable_nodes if not node.is_method]
        methods_by_class: Dict[str, List["model.CallableNode"]] = {}
        for method_node in method_nodes:
            methods_by_class.setdefault(method_node.class_name, []).append(method_node)
        for class_name, method_nodes in methods_by_class.items():
            lines.append(textwrap.indent(f"class {class_name}:", 4 * " "))
            for node in method_nodes:
                desc = node.readable_content()
                lines.append(textwrap.indent(textwrap.dedent(desc), 8 * " "))
            lines.append("")
        for node in func_nodes:
            desc = node.readable_content()
            lines.append(textwrap.indent(textwrap.dedent(desc), 4 * " "))
        lines.append("")
    return "\n".join(lines)


def to_dot(graph: "model.DependencyGraph") -> str:
    nodes: Dict[DepKey, DotNode] = {}
    module_groups: Dict[str, DotGroup] = {}  # module name -> Group
    class_groups: Dict[str, DotGroup] = {}  # class name -> Group
    for key, node in graph.nodes.items():
        module_name, obj_addr = key
        if module_name not in module_groups:
            module_groups[module_name] = DotGroup(
                label=module_name, nodes=[], parent=None
            )
        if isinstance(node, model.GlobalVarNode):
            color = SOLARIZED_LIGHT["red"]
        elif isinstance(node, model.CallableNode):
            color = (
                SOLARIZED_LIGHT["blue"]
                if not node.is_method
                else SOLARIZED_LIGHT["violet"]
            )
        else:
            color = SOLARIZED_LIGHT["base03"]
        dot_node = DotNode(
            internal_name=".".join(key), label=node.obj_name, color=color
        )
        nodes[key] = dot_node
        module_groups[module_name].nodes.append(dot_node)
        if isinstance(node, model.CallableNode) and node.is_method:
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


from . import model
