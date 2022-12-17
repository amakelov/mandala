from ..common_imports import *
from .model import FuncOp
from ..ui.viz import (
    Node,
    Edge,
    Group,
    SOLARIZED_LIGHT,
    to_dot_string,
    write_output,
    HTMLBuilder,
    Cell,
)
from typing import Literal


class ValQuery:
    """
    Represents an input/output to an operation under the `query` interpretation
    of code.

    This is the equivalent of a `ValueRef` when in a query context. In SQL
    terms, it points to the table of objects. What matters more are the
    constraints it is connected to via the `creator` and `consumers` fields,
    which tell us which tables to join and how.

    There is always a unique `creator`, which is the operation that returned
    this `ValQuery` object, but potentially many `consumers`, which are
    subsequent calls to operations that consume this `ValQuery` object.
    """

    def __init__(self, creator: Optional["FuncQuery"], created_as: Optional[int]):
        self.creator = creator
        self.created_as = created_as
        self.consumers: List["FuncQuery"] = []
        self.consumed_as: List[str] = []
        self.aliases = []
        self.column_name = None

    def add_consumer(self, consumer: "FuncQuery", consumed_as: str):
        self.consumers.append(consumer)
        self.consumed_as.append(consumed_as)

    def neighbors(self) -> List["FuncQuery"]:
        res = []
        if self.creator is not None:
            res.append(self.creator)
        for cons in self.consumers:
            res.append(cons)
        return res

    def named(self, name: str) -> "ValQuery":
        self.column_name = name
        return self


class FuncQuery:
    """
    Represents a call to an operation under the `query` interpretation of code.

    This is the equivalent to a `Call` when in a query context. In SQL terms, it
    points to the memoization table of some function. The `inputs` and `outputs`
    connected to it are the `ValQuery` objects that represent the inputs and
    outputs of this call. They are indexed by *internal* names. See `core.sig.Signature`
    for an explanation.
    """

    def __init__(self, inputs: Dict[str, ValQuery], func_op: FuncOp):
        self.inputs = inputs
        self.outputs = []
        self.func_op = func_op

    def set_outputs(self, outputs: List[ValQuery]):
        self.outputs = outputs

    def neighbors(self) -> List[ValQuery]:
        return [
            x
            for x in itertools.chain(
                self.inputs.values(), self.outputs if self.outputs is not None else []
            )
        ]

    def unlink(self):
        for inp in self.inputs.values():
            idxs = [i for i, x in enumerate(inp.consumers) if x is self]
            inp.consumers = [x for i, x in enumerate(inp.consumers) if i not in idxs]
            inp.consumed_as = [
                x for i, x in enumerate(inp.consumed_as) if i not in idxs
            ]
        for out in self.outputs:
            out.creator = None
            out.created_as = None


def concat_lists(lists: List[list]) -> list:
    return [x for lst in lists for x in lst]


def traverse_all(val_queries: List[ValQuery]) -> Tuple[List[ValQuery], List[FuncQuery]]:
    """
    Extend the given `ValQuery` objects to all objects connected to them through
    function inputs/outputs.
    """
    val_queries_ = [_ for _ in val_queries]
    op_queries_: List[FuncQuery] = []
    found_new = True
    while found_new:
        found_new = False
        val_neighbors = concat_lists([v.neighbors() for v in val_queries_])
        op_neighbors = concat_lists([o.neighbors() for o in op_queries_])
        if any(k not in op_queries_ for k in val_neighbors):
            found_new = True
            for neigh in val_neighbors:
                if neigh not in op_queries_:
                    op_queries_.append(neigh)
        if any(k not in val_queries_ for k in op_neighbors):
            found_new = True
            for neigh in op_neighbors:
                if neigh not in val_queries_:
                    val_queries_.append(neigh)
    return val_queries_, op_queries_


def visualize_computational_graph(
    val_queries: List[ValQuery],
    func_queries: List[FuncQuery],
    layout: Literal["computational", "bipartite"] = "computational",
    memoization_tables: Optional[Dict[FuncQuery, pd.DataFrame]] = None,
    output_path: Optional[Path] = None,
):
    if memoization_tables is not None:
        assert layout == "bipartite"
    nodes = {}  # val/op query -> Node obj
    edges = []
    for val_query in val_queries:
        html_label = HTMLBuilder()
        html_label.add_row(cells=[Cell(text=str(val_query.column_name), port=None)])
        node = Node(
            internal_name=str(id(val_query)),
            # label=str(val_query.column_name),
            label=html_label.to_html_like_label(),
            color=SOLARIZED_LIGHT["blue"],
            shape="plain",
        )
        nodes[val_query] = node
    for func_query in func_queries:
        html_label = HTMLBuilder()
        title_cell = Cell(
            text=func_query.func_op.sig.ui_name,
            port=None,
            bgcolor=SOLARIZED_LIGHT["blue"],
        )
        input_cells = []
        output_cells = []
        for input_name in func_query.inputs.keys():
            input_cells.append(Cell(text=input_name, port=input_name))
            # html_label.add_row(elts=[Cell(text=input_name, port=input_name)])
        for output_idx in range(len(func_query.outputs)):
            output_cells.append(
                Cell(text=f"output_{output_idx}", port=f"output_{output_idx}")
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
                    html_label.add_row(cells=[Cell(text=str(x)) for x in tup])
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
        for input_name, val_query in func_query.inputs.items():
            if layout == "bipartite":
                edges.append(
                    Edge(
                        target_node=nodes[val_query],
                        source_node=nodes[func_query],
                        source_port=input_name,
                        arrowtail="none",
                        arrowhead="none",
                    )
                )
            elif layout == "computational":
                edges.append(
                    Edge(
                        source_node=nodes[val_query],
                        target_node=nodes[func_query],
                        target_port=input_name,
                    )
                )
            else:
                raise ValueError(f"Unknown layout: {layout}")
        for output_idx, val_query in enumerate(func_query.outputs):
            edges.append(
                Edge(
                    source_node=nodes[func_query],
                    target_node=nodes[val_query],
                    source_port=f"output_{output_idx}",
                    arrowtail="none" if layout == "bipartite" else None,
                    arrowhead="none" if layout == "bipartite" else None,
                )
            )
    dot_string = to_dot_string(nodes=list(nodes.values()), edges=edges, groups=[])
    output_path = Path("graph.svg") if output_path is None else output_path
    write_output(output_path=output_path, dot_string=dot_string, output_ext="svg")
