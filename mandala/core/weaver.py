from ..common_imports import *
from .model import FuncOp, Ref
from .tps import Type, ListType, DictType, SetType, AnyType
from .builtins_ import Builtins
from .utils import Hashing, concat_lists
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

    def __init__(
        self,
        creator: Optional["FuncQuery"],
        created_as: Optional[int],
        tp: Optional[Type] = None,
        constraint: Optional[List[str]] = None,
    ):
        self.creator = creator
        self.created_as = created_as
        self.consumers: List["FuncQuery"] = []
        self.consumed_as: List[str] = []
        self.aliases = []
        self.column_name = None
        self.constraint = constraint
        self.tp = tp

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

    def named(self, name: str) -> Any:
        self.column_name = name
        return self

    def __getitem__(self, idx: Union[int, str, "ValQuery"]) -> "ValQuery":
        tp = self.tp or _infer_type(self)
        if isinstance(tp, ListType):
            return BuiltinQueries.GetListItemQuery(
                lst=self, idx=qwrap(idx, tp=tp.elt_type)
            )
        elif isinstance(tp, DictType):
            assert isinstance(idx, str)
            return BuiltinQueries.GetDictItemQuery(
                dct=self, key=qwrap(idx, tp=tp.val_type)
            )
        else:
            raise NotImplementedError(f"Cannot index into query of type {tp}")


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

    @staticmethod
    def link(inputs: Dict[str, ValQuery], func_op: FuncOp) -> "FuncQuery":
        """
        Link a func query into the graph
        """
        result = FuncQuery(inputs=inputs, func_op=func_op)
        result.set_outputs(
            outputs=[
                ValQuery(creator=result, created_as=i, tp=tp)
                for i, tp in zip(range(func_op.sig.n_outputs), func_op.output_types)
            ]
        )
        for k, v in inputs.items():
            v.add_consumer(consumer=result, consumed_as=k)
        return result

    def unlink(self):
        """
        Remove this `FuncQuery` from the graph.
        """
        for inp in self.inputs.values():
            idxs = [i for i, x in enumerate(inp.consumers) if x is self]
            inp.consumers = [x for i, x in enumerate(inp.consumers) if i not in idxs]
            inp.consumed_as = [
                x for i, x in enumerate(inp.consumed_as) if i not in idxs
            ]
        for out in self.outputs:
            out.creator = None
            out.created_as = None


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


def _filter_qdict(dct: Dict[str, Any]) -> Dict[str, ValQuery]:
    return {k: qwrap(v) for k, v in dct.items() if v is not None}


class BuiltinQueries:
    @staticmethod
    def list_relation(
        lst: Optional[ValQuery] = None,
        elt: Optional[Union[ValQuery, Any]] = None,
        idx: Optional[Union[ValQuery, int]] = None,
    ):
        FuncQuery.link(
            inputs=_filter_qdict({"lst": lst, "elt": elt, "idx": idx}),
            func_op=Builtins.list_op,
        )

    @staticmethod
    def dict_relation(
        dct: Optional[ValQuery] = None,
        key: Optional[Union[ValQuery, str]] = None,
        val: Optional[Union[ValQuery, Any]] = None,
    ):
        FuncQuery.link(
            inputs=_filter_qdict({"dct": dct, "key": key, "val": val}),
            func_op=Builtins.dict_op,
        )

    @staticmethod
    def set_relation(st: Optional[ValQuery] = None, elt: Optional[ValQuery] = None):
        FuncQuery.link(
            inputs=_filter_qdict({"st": st, "elt": elt}), func_op=Builtins.set_op
        )

    ############################################################################
    ### special cases
    ############################################################################
    @staticmethod
    def ListQuery(elt: Any, idx: Optional[Any] = None) -> ValQuery:
        result = ValQuery(creator=None, created_as=None, tp=ListType(elt_type=elt.tp))
        BuiltinQueries.list_relation(lst=result, elt=elt, idx=idx)
        return result

    @staticmethod
    def DictQuery(
        val: Optional[ValQuery] = None, key: Optional[ValQuery] = None
    ) -> ValQuery:
        result = ValQuery(creator=None, created_as=None, tp=DictType(val_type=val.tp))
        BuiltinQueries.dict_relation(dct=result, key=key, val=val)
        return result

    @staticmethod
    def SetQuery(elt: ValQuery) -> ValQuery:
        result = ValQuery(creator=None, created_as=None, tp=SetType(elt_type=elt.tp))
        BuiltinQueries.set_relation(st=result, elt=elt)
        return result

    @staticmethod
    def GetListItemQuery(lst: ValQuery, idx: Optional[ValQuery] = None) -> ValQuery:
        elt_tp = lst.tp.elt_type if isinstance(lst.tp, ListType) else None
        result = ValQuery(creator=None, created_as=None, tp=elt_tp)
        BuiltinQueries.list_relation(lst=lst, elt=result, idx=idx)
        return result

    @staticmethod
    def GetDictItemQuery(dct: ValQuery, key: Optional[ValQuery] = None) -> ValQuery:
        val_tp = dct.tp.val_type if isinstance(dct.tp, DictType) else None
        result = ValQuery(creator=None, created_as=None, tp=val_tp)
        BuiltinQueries.dict_relation(dct=dct, key=key, val=result)
        return result

    ############################################################################
    ### syntactic sugar
    ############################################################################
    @staticmethod
    def classify_pattern(obj: Any) -> str:
        if isinstance(obj, list) and obj[1] == Ellipsis:
            return "list"
        elif isinstance(obj, dict) and Ellipsis in obj.keys():
            return "dict"
        elif isinstance(obj, set) and Ellipsis in obj:
            return "set"
        else:
            return "atom"

    @staticmethod
    def parse_list_pattern(obj: list) -> Tuple[ValQuery, List[Any]]:
        # parse a pattern of the form [x, ...]
        children = [obj[0]]
        if isinstance(obj[0], Ref):
            children[0] = qwrap(obj[0])
        if not all([isinstance(x, ValQuery) for x in children]):
            raise NotImplementedError()
        return BuiltinQueries.ListQuery(elt=children[0]), children

    @staticmethod
    def parse_dict_pattern(obj: dict) -> Tuple[ValQuery, List[Any]]:
        # parse a pattern of the form {...: x} | {x: ...} | {x: y, ...:...}
        if len(obj) == 1:
            k = list(obj.keys())[0]
            if k == Ellipsis:
                res, children = BuiltinQueries.DictQuery(val=obj[k]), [obj[k]]
            else:
                res, children = BuiltinQueries.DictQuery(key=k), [k]
        else:
            k = [k for k in list(obj.keys()) if k != Ellipsis][0]
            v = obj[k]
            res, children = BuiltinQueries.DictQuery(key=k, val=v), [k, v]
        if not all([isinstance(x, ValQuery) for x in children]):
            raise NotImplementedError()
        return res, children

    @staticmethod
    def parse_set_pattern(obj: set) -> Tuple[ValQuery, List[Any]]:
        # parse a pattern of the form {x, ...}
        res, children = BuiltinQueries.SetQuery(elt=list(obj)[0]), list(obj)
        if not all([isinstance(x, ValQuery) for x in children]):
            raise NotImplementedError()
        return res, children


def qwrap(obj: Any, tp: Optional[Type] = None) -> ValQuery:
    pattern_type = BuiltinQueries.classify_pattern(obj=obj)
    if pattern_type == "atom":
        if tp is None:
            tp = AnyType()
        # wrap an atom as a pointwise constraint
        if isinstance(obj, ValQuery):
            return obj
        else:
            uid = obj.uid if isinstance(obj, Ref) else Hashing.get_content_hash(obj)
            return ValQuery(
                tp=tp,
                creator=None,
                created_as=None,
                constraint=[uid],
            )
    elif pattern_type == "list":
        return BuiltinQueries.parse_list_pattern(obj=obj)[0]
    elif pattern_type == "dict":
        return BuiltinQueries.parse_dict_pattern(obj=obj)[0]
    elif pattern_type == "set":
        return BuiltinQueries.parse_set_pattern(obj=obj)[0]
    else:
        raise NotImplementedError()


def _infer_type(val_query: ValQuery) -> Type:
    consumer_op_names = [c.func_op.sig.ui_name for c in val_query.consumers]
    mapping = {"__list__": ListType(), "__dict__": DictType(), "__set__": SetType()}
    tps = [mapping.get(x, None) for x in consumer_op_names]
    struct_tps = [x for x in tps if x is not None]
    if len(struct_tps) == 0:
        return AnyType()
    elif len(struct_tps) == 1:
        return struct_tps[0]
    else:
        raise RuntimeError(f"Multiple types for {val_query}: {struct_tps}")


def computational_graph_to_dot(
    val_queries: List[ValQuery],
    func_queries: List[FuncQuery],
    layout: Literal["computational", "bipartite"] = "computational",
    memoization_tables: Optional[Dict[FuncQuery, pd.DataFrame]] = None,
) -> str:
    # if memoization_tables is not None:
    #     assert layout == "bipartite"
    nodes = {}  # val/op query -> Node obj
    edges = []
    col_names = []
    counter = 0
    for val_query in val_queries:
        if val_query.column_name is not None:
            col_names.append(val_query.column_name)
        else:
            col_names.append(f"unnamed_{counter}")
            counter += 1
    for val_query, col_name in zip(val_queries, col_names):
        html_label = HTMLBuilder()
        html_label.add_row(
            cells=[
                Cell(
                    text=str(col_name),
                    port=None,
                    bold=True,
                    bgcolor=SOLARIZED_LIGHT["orange"],
                    font_color=SOLARIZED_LIGHT["base3"],
                )
            ]
        )
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
        func_preview = (
            f'{func_query.func_op.sig.ui_name}({", ".join(func_query.inputs.keys())})'
        )
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
    return to_dot_string(nodes=list(nodes.values()), edges=edges, groups=[])


def visualize_computational_graph(
    val_queries: List[ValQuery],
    func_queries: List[FuncQuery],
    layout: Literal["computational", "bipartite"] = "computational",
    memoization_tables: Optional[Dict[FuncQuery, pd.DataFrame]] = None,
    output_path: Optional[Path] = None,
):
    dot_string = computational_graph_to_dot(
        val_queries=val_queries,
        func_queries=func_queries,
        layout=layout,
        memoization_tables=memoization_tables,
    )
    if output_path is None:
        tempfile_obj, output_name = tempfile.mkstemp(suffix=".svg")
        output_path = Path(output_name)
    write_output(output_path=output_path, dot_string=dot_string, output_ext="svg")
    return output_path
