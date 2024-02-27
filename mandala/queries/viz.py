from abc import ABC, abstractmethod
from ..common_imports import *
from typing import Literal
from ..core.config import parse_output_idx, Config
from ..core.model import Ref
from ..core.wrapping import unwrap
from ..core.tps import ListType, StructType, DictType, SetType
from .weaver import (
    ValNode,
    CallNode,
    StructOrientations,
    get_items,
    get_elts,
    get_elt_and_struct,
    get_idx,
    is_key,
    is_idx,
    traverse_all,
    get_vq_orientation,
)
from .graphs import InducedSubgraph, get_canonical_order
from ..ui.viz import (
    Node,
    Edge,
    SOLARIZED_LIGHT,
    to_dot_string,
    write_output,
    HTMLBuilder,
    Cell,
)
import textwrap


class ValueLoaderABC(ABC):
    @abstractmethod
    def load_value(self, full_uid: str) -> Any:
        raise NotImplementedError


class GraphPrinter:
    def __init__(
        self,
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        value_loader: Optional[ValueLoaderABC] = None,
        names: Optional[Dict[ValNode, str]] = None,
    ):
        self.vqs = vqs
        self.fqs = fqs
        self.value_loader = value_loader
        self.g = InducedSubgraph(vqs=vqs, fqs=fqs)
        self.v_labels, self.f_labels, self.vq_to_node = self.g.canonicalize()
        self.full_topsort, self.sources, self.sinks = self.g.topsort(
            canonical_labels={**self.v_labels, **self.f_labels}
        )
        if names is None:
            names = get_names(
                hints={}, canonical_order=get_canonical_order(vqs=vqs, fqs=fqs)
            )
        self.names = names

    def get_struct_comment(
        self,
        vq: ValNode,
        elt_names: Tuple[str, ...],
        idx_names: Optional[Tuple[str, ...]] = None,
    ) -> str:
        id_to_name = {"__list__": "list", "__dict__": "dict", "__set__": "set"}
        if len(elt_names) == 1:
            s = f"{self.names[vq]} will match any {id_to_name[vq.tp.struct_id]} containing a match for {elt_names[0]}"
            if idx_names is not None:
                s += f" at index {idx_names[0]}"
        else:
            s = f'{self.names[vq]} will match any {id_to_name[vq.tp.struct_id]} containing matches for each of {", ".join(elt_names)}'
            if idx_names is not None:
                s += f" at indices {', '.join(idx_names)}"
        return s

    def get_source_comment(self, vq: ValNode) -> str:
        if is_idx(vq=vq):
            return "index into list"
        elif is_key(vq=vq):
            return "key into dict"
        else:
            return "input to computation; can match anything"

    def get_construct_computation_rhs(self, node: ValNode) -> str:
        """
        Given a constructive struct, return
            [name_0, ..., name_1] for lists
            {key_0: name_0, ..., key_n: name_n} for dicts
            {name_0, ..., name_n} for sets
        """
        if isinstance(node.tp, (ListType, DictType)):
            idxs_and_elts = list(get_items(vq=node).values())
            idxs = [x[0] for x in idxs_and_elts]
            idx_values = [
                unwrap(self.value_loader.load_value(full_uid=idx.constraint[0]))
                for idx in idxs
            ]
            elts = [x[1] for x in idxs_and_elts]
            elt_names = tuple([str(self.names[elt]) for elt in elts])
            idx_value_to_elt_name = {
                idx_value: elt_name
                for idx_value, elt_name in zip(idx_values, elt_names)
            }
            if isinstance(node.tp, ListType):
                assert sorted(idx_value_to_elt_name.keys()) == list(range(len(elts)))
                rhs = f'[{", ".join([idx_value_to_elt_name[i] for i in range(len(elts))])}]'
            elif isinstance(node.tp, DictType):
                elt_strings = [
                    f"'{idx_value}': {elt_name}"
                    for idx_value, elt_name in idx_value_to_elt_name.items()
                ]
                rhs = f'{", ".join(elt_strings)}'
                rhs = f"{{{rhs}}}"
            else:
                raise RuntimeError
        elif isinstance(node.tp, SetType):
            elts = list(get_elts(vq=node).values())
            elt_names = tuple([str(self.names[elt]) for elt in elts])
            rhs = f'{", ".join(elt_names)}'
            rhs = f"{{{rhs}}}"
        else:
            raise ValueError
        return rhs

    def get_construct_query_rhs(self, node: ValNode) -> str:
        if isinstance(node.tp, (ListType, DictType)):
            idxs_and_elts = list(get_items(vq=node).values())
            idxs = [x[0] for x in idxs_and_elts]
            elts = [x[1] for x in idxs_and_elts]
            elt_names = tuple([str(self.names[elt]) for elt in elts])
            idx_names = tuple(
                [str(self.names[idx]) if idx is not None else "?" for idx in idxs]
            )
            comment = self.get_struct_comment(
                vq=node, elt_names=elt_names, idx_names=idx_names
            )
            if isinstance(node.tp, ListType):
                rhs = f'ListQ(elts=[{", ".join(elt_names)}], idxs=[{", ".join(idx_names)}]) # {comment}'
            elif isinstance(node.tp, DictType):
                elt_strings = [
                    f"{idx_name}: {elt_name}"
                    for idx_name, elt_name in zip(idx_names, elt_names)
                ]
                rhs = f'DictQ(dct={{..., {", ".join(elt_strings)}, ...}}) # {comment}'
            else:
                raise RuntimeError
        elif isinstance(node.tp, SetType):
            elts = list(get_elts(vq=node).values())
            elt_names = tuple([str(self.names[elt]) for elt in elts])
            comment = self.get_struct_comment(vq=node, elt_names=elt_names)
            rhs = f'SetQ(elts=[{", ".join(elt_names)}]) # {comment}'
        else:
            raise ValueError
        return rhs

    def get_op_computation_line(self, node: CallNode) -> str:
        full_returns = node.returns_interp
        full_returns_names = [
            self.names[vq] if vq is not None else "_" for vq in full_returns
        ]
        lhs = ", ".join(full_returns_names)
        # rhs
        args_dict = {arg_name: self.names[vq] for arg_name, vq in node.inputs.items()}
        for inp_name in node.func_op.sig.input_names:
            if inp_name not in args_dict:
                raise RuntimeError
        args_string = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
        rhs = f"{node.func_op.sig.ui_name}({args_string})"
        if len(lhs) == 0:
            return rhs
        else:
            return f"{lhs} = {rhs}"

    def get_op_query_line(self, node: CallNode) -> str:
        # lhs
        full_returns = node.returns_interp
        full_returns_names = [
            self.names[vq] if vq is not None else "_" for vq in full_returns
        ]
        lhs = ", ".join(full_returns_names)
        # rhs
        args_dict = {arg_name: self.names[vq] for arg_name, vq in node.inputs.items()}
        for inp_name in node.func_op.sig.input_names:
            if inp_name not in args_dict:
                args_dict[inp_name] = "Q()"
        args_string = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
        rhs = f"{node.func_op.sig.ui_name}({args_string})"
        if len(lhs) == 0:
            return rhs
        else:
            return f"{lhs} = {rhs}"

    def get_destruct_computation_line(self, node: CallNode) -> str:
        elt, struct = get_elt_and_struct(fq=node)
        idx = get_idx(fq=node)
        lhs = f"{self.names[elt]}"
        if idx is None:
            raise NotImplementedError
        # inline the index value if the index is a source in the graph
        if idx in self.sources:
            if idx.constraint is None:
                rhs = f"{self.names[struct]}[{self.names[idx]}]"
            else:
                assert len(idx.constraint) == 1
                idx_ref = self.value_loader.load_value(full_uid=idx.constraint[0])
                idx_value = unwrap(idx_ref)
                if isinstance(idx_value, int):
                    rhs = f"{self.names[struct]}[{idx_value}]"
                elif isinstance(idx_value, str):
                    rhs = f"{self.names[struct]}['{idx_value}']"
                else:
                    raise RuntimeError
        else:
            rhs = f"{self.names[struct]}[{self.names[idx]}]"
        return f"{lhs} = {rhs}"

    def get_destruct_query_line(self, node: CallNode) -> str:
        elt, struct = get_elt_and_struct(fq=node)
        idx = get_idx(fq=node)
        lhs = f"{self.names[elt]}"
        if idx is None:
            idx_label = "?"
        else:
            idx_label = self.names[idx]
        rhs = f"{self.names[struct]}[{idx_label}] # {self.names[elt]} will match any element of a match for {self.names[struct]} at index matching {idx_label}"
        return f"{lhs} = {rhs}"

    def print_computational_graph(
        self, show_sources_as: Literal["values", "uids", "omit", "name_only"] = "values"
    ) -> str:
        res = []
        for node in self.full_topsort:
            if isinstance(node, ValNode):
                if node in self.sources:
                    if is_idx(node) or is_key(node):
                        # exclude indices/keys if they are sources (we will inline them)
                        continue
                    if show_sources_as == "omit":
                        continue
                    elif show_sources_as == "name_only":
                        res.append(self.names[node])
                        continue
                    assert node.constraint is not None
                    assert len(node.constraint) == 1
                    if show_sources_as == "values":
                        ref = self.value_loader.load_value(full_uid=node.constraint[0])
                        value = unwrap(ref)
                        rep = textwrap.shorten(repr(value), 25)
                        res.append(f"{self.names[node]} = {rep}")
                    elif show_sources_as == "uids":
                        uid, causal_uid = Ref.parse_full_uid(
                            full_uid=node.constraint[0]
                        )
                        res.append(
                            f"{self.names[node]} = Ref(uid={uid}, causal_uid={causal_uid})"
                        )
                    else:
                        raise ValueError
                elif isinstance(node.tp, StructType):
                    if get_vq_orientation(node) != StructOrientations.construct:
                        continue
                    rhs = self.get_construct_computation_rhs(node=node)
                    lhs = f"{self.names[node]}"
                    line = f"{lhs} = {rhs}"
                    res.append(line)
            elif isinstance(node, CallNode):
                if not node.func_op.is_builtin:
                    res.append(self.get_op_computation_line(node=node))
                else:
                    if node.orientation == StructOrientations.destruct:
                        res.append(self.get_destruct_computation_line(node=node))
        return "\n".join(res)

    def print_query_graph(
        self,
        selection: Optional[List[ValNode]],
        pprint: bool = False,
    ):
        res = []
        for node in self.full_topsort:
            if isinstance(node, ValNode):
                if node in self.sources:
                    comment = self.get_source_comment(vq=node)
                    res.append(f"{self.names[node]} = Q() # {comment}")
                elif isinstance(node.tp, StructType):
                    if get_vq_orientation(node) != StructOrientations.construct:
                        continue
                    rhs = self.get_construct_query_rhs(node=node)
                    lhs = f"{self.names[node]}"
                    line = f"{lhs} = {rhs}"
                    res.append(line)
            elif isinstance(node, CallNode):
                if not node.func_op.is_builtin:
                    res.append(self.get_op_query_line(node=node))
                else:
                    if node.orientation == StructOrientations.destruct:
                        res.append(self.get_destruct_query_line(node=node))
        if selection is not None:
            res.append(
                f"result = storage.df({', '.join([self.names[vq] for vq in selection])})"
            )
        res = [textwrap.indent(line, "    ") for line in res]
        if Config.has_rich and pprint:
            from rich.syntax import Syntax

            highlighted = Syntax(
                "\n".join(res),
                "python",
                theme="solarized-light",
                line_numbers=False,
            )
            # return Panel.fit(highlighted, title="Computational Graph")
            return highlighted
        else:
            return "\n".join(res)


def print_graph(
    vqs: Set[ValNode],
    fqs: Set[CallNode],
    names: Dict[ValNode, str],
    selection: Optional[List[ValNode]],
    pprint: bool = False,
):
    printer = GraphPrinter(vqs=vqs, fqs=fqs, names=names)
    s = printer.print_query_graph(selection=selection)
    if Config.has_rich and pprint:
        rich.print(s)
    else:
        print(s)


def graph_to_dot(
    vqs: List[ValNode],
    fqs: List[CallNode],
    names: Dict[ValNode, str],
    layout: Literal["computational", "bipartite"] = "computational",
    memoization_tables: Optional[Dict[CallNode, pd.DataFrame]] = None,
) -> str:
    # should work for subgraphs
    assert set(vqs) <= set(names.keys())
    nodes = {}  # val/op query -> Node obj
    edges = []
    col_names = []
    counter = 0
    for vq in vqs:
        if names[vq] is not None:
            col_names.append(names[vq])
        else:
            col_names.append(f"unnamed_{counter}")
            counter += 1
    for vq, col_name in zip(vqs, col_names):
        html_label = HTMLBuilder()
        if hasattr(vq, "_hidden_message"):
            msg = vq._hidden_message
            text = f"{col_name} ({msg})"
        else:
            text = str(col_name)
        html_label.add_row(
            cells=[
                Cell(
                    text=text,
                    port=None,
                    bold=True,
                    bgcolor=SOLARIZED_LIGHT["orange"],
                    font_color=SOLARIZED_LIGHT["base3"],
                )
            ]
        )
        node = Node(
            internal_name=str(id(vq)),
            # label=str(val_query.column_name),
            label=html_label.to_html_like_label(),
            color=SOLARIZED_LIGHT["blue"],
            shape="plain",
        )
        nodes[vq] = node
    for func_query in fqs:
        html_label = HTMLBuilder()
        func_preview = (
            # f'{func_query.displayname}({", ".join(func_query.inputs.keys())})'
            func_query.displayname
        )
        if hasattr(func_query, "_hidden_message"):
            msg = func_query._hidden_message
            func_preview = f"{func_preview} ({msg})"
        title_cell = Cell(
            text=func_preview,
            port=None,
            bgcolor=SOLARIZED_LIGHT["blue"],
            bold=True,
            font_color=SOLARIZED_LIGHT["base3"],
        )
        input_cells = []
        output_cells = []
        for input_name in func_query.inputs.keys():
            input_cells.append(Cell(text=input_name, port=input_name, bold=True))
            # html_label.add_row(elts=[Cell(text=input_name, port=input_name)])
        for output_idx in range(len(func_query.outputs)):
            output_cells.append(
                Cell(
                    text=f"output_{output_idx}", port=f"output_{output_idx}", bold=True
                )
            )
        if layout == "bipartite":
            html_label.add_row(cells=[title_cell])
            if len(input_cells + output_cells) > 0:
                html_label.add_row(cells=input_cells + output_cells)
            if memoization_tables is not None:
                column_names = [cell.text for cell in input_cells + output_cells]
                port_names = [cell.port for cell in input_cells + output_cells]
                # remove ports from the table column cells
                for cell in input_cells + output_cells:
                    cell.port = None
                df = memoization_tables[func_query][column_names]
                rows = list(df.head().itertuples(index=False))
                for tup in rows:
                    html_label.add_row(
                        cells=[
                            Cell(text=textwrap.shorten(str(x), 25), bold=True)
                            for x in tup
                        ]
                    )
                # add port names to the cells in the *last* row
                for cell, port_name in zip(html_label.rows[-1], port_names):
                    cell.port = port_name
        elif layout == "computational":
            if len(input_cells) > 0:
                html_label.add_row(input_cells)
            html_label.add_row([title_cell])
            if len(output_cells) > 0:
                html_label.add_row(output_cells)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        node = Node(
            internal_name=str(id(func_query)),
            # label=str(func_query.func_op.sig.ui_name),
            label=html_label.to_html_like_label(),
            color=SOLARIZED_LIGHT["red"],
            shape="plain",
        )
        nodes[func_query] = node
        for input_name, vq in func_query.inputs.items():
            if not vq in nodes:
                continue
            if layout == "bipartite":
                edges.append(
                    Edge(
                        target_node=nodes[vq],
                        source_node=nodes[func_query],
                        source_port=input_name,
                        arrowtail="none",
                        arrowhead="none",
                    )
                )
            elif layout == "computational":
                edges.append(
                    Edge(
                        source_node=nodes[vq],
                        target_node=nodes[func_query],
                        target_port=input_name,
                    )
                )
            else:
                raise ValueError(f"Unknown layout: {layout}")
        for output_name, vq in func_query.outputs.items():
            if not vq in nodes:
                continue
            edges.append(
                Edge(
                    source_node=nodes[func_query],
                    target_node=nodes[vq],
                    source_port=output_name,
                    arrowtail="none" if layout == "bipartite" else None,
                    arrowhead="none" if layout == "bipartite" else None,
                )
            )
    return to_dot_string(nodes=list(nodes.values()), edges=edges, groups=[])


def visualize_graph(
    vqs: Set[ValNode],
    fqs: Set[CallNode],
    names: Optional[Dict[ValNode, str]],
    layout: Literal["computational", "bipartite"] = "computational",
    memoization_tables: Optional[Dict[CallNode, pd.DataFrame]] = None,
    output_path: Optional[Path] = None,
    show_how: Literal["none", "browser", "inline", "open"] = "none",
):
    if names is None:
        names = get_names(
            hints={}, canonical_order=get_canonical_order(vqs=vqs, fqs=fqs)
        )
    dot_string = graph_to_dot(
        vqs=list(vqs),
        fqs=list(fqs),
        names=names,
        layout=layout,
        memoization_tables=memoization_tables,
    )
    if output_path is None:
        tempfile_obj, output_name = tempfile.mkstemp(suffix=".svg")
        output_path = Path(output_name)
    write_output(
        output_path=output_path,
        dot_string=dot_string,
        output_ext="svg",
        show_how=show_how,
    )
    return output_path


def show(*vqs: ValNode):
    vqs, fqs = traverse_all(vqs=list(vqs), direction="both")
    visualize_graph(vqs=vqs, fqs=fqs, show_how="browser")


def extract_names_from_scope(scope: Dict[str, Any]) -> Dict[ValNode, str]:
    """
    Heuristic to get deterministic name for all ValQueries we can find in the
    scope.
    """
    names_per_vq = defaultdict(list)
    for k, v in scope.items():
        if k.startswith("_"):
            continue
        if isinstance(v, Ref) and v._query is not None:
            names_per_vq[v.query].append(k)
        elif isinstance(v, ValNode):
            names_per_vq[v].append(k)
    res = {}
    for vq, names_per_vq in names_per_vq.items():
        if len(names_per_vq) > 1:
            logger.warning(
                f"Found multiple names for {vq}: {names_per_vq}, choosing {sorted(names_per_vq)[0]}"
            )
        res[vq] = names_per_vq[0]
    return res


def get_names(
    hints: Dict[ValNode, str], canonical_order: List[ValNode]
) -> Dict[ValNode, str]:
    """
    Get names for the given oredered list of ValQueries, using the following
    priority:
        - from the object itself;
        - the hints;
        - a_{counter} for the rest.
    """
    counter = Count()
    idx_counter = Count()
    key_counter = Count()
    existing_names = set(hints.values()) | {
        vq.name for vq in canonical_order if vq.name is not None
    }
    res = {}
    for vq in canonical_order:
        if vq.name is not None:
            res[vq] = vq.name
        elif vq in hints:
            res[vq] = hints[vq]
        else:
            if is_key(vq):
                c, prefix = key_counter, "key"
            elif is_idx(vq):
                c, prefix = idx_counter, "idx"
            else:
                c, prefix = counter, "a"
            while f"{prefix}{c.i}" in existing_names:
                c.i += 1
            res[vq] = f"{prefix}{c.i}"
            c.i += 1
    return res


class Count:
    def __init__(self):
        self.i = 0
