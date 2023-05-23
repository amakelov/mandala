from collections import Counter
from ..common_imports import *
from ..core.utils import OpKey
from ..core.config import parse_output_idx
from .weaver import ValNode, CallNode
from .compiler import Compiler
from .solver import NaiveQueryEngine
from .viz import get_names
from .graphs import (
    InducedSubgraph,
    is_connected,
    copy_subgraph,
)


class Querier:
    @staticmethod
    def check_df(
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        df: pd.DataFrame,
        funcs: Dict[str, Callable],
    ):
        """
        Check validity of a query result projected from the given graph against
        executables.
        """
        cols = set(df.columns)
        assert cols <= set(vq.name for vq in vqs)
        for fq in fqs:
            func = funcs[fq.func_op.sig.ui_name]
            input_cols: Dict[str, str] = {
                k: vq.name for k, vq in fq.inputs.items() if vq.name in cols
            }
            ouptut_cols: Dict[int, str] = {
                parse_output_idx(k): vq.name
                for k, vq in fq.outputs.items()
                if vq.name in cols
            }
            for i, row in df.iterrows():
                inputs = {k: row[v] for k, v in input_cols.items()}
                outputs = func(**inputs)
                if not (isinstance(outputs, tuple)):
                    outputs = (outputs,)
                for j, v in enumerate(outputs):
                    assert row[ouptut_cols[j]] == v

    @staticmethod
    def execute_naive(
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        selection: List[ValNode],
        memoization_tables: Dict[str, pd.DataFrame],
        filter_duplicates: bool,
        table_evaluator: Callable,
        visualize_steps_at: Optional[Path] = None,
    ) -> pd.DataFrame:
        v_copymap, f_copymap = copy_subgraph(vqs=vqs, fqs=fqs)
        vqs = set(v_copymap.values())
        fqs = set(f_copymap.values())
        select_copies = [v_copymap[vq] for vq in selection]
        tables = {
            f: memoization_tables[f.func_op.sig.versioned_internal_name] for f in fqs
        }
        query_graph = NaiveQueryEngine(
            vqs=vqs,
            fqs=fqs,
            selection=select_copies,
            tables=tables,
            _table_evaluator=table_evaluator,
            _visualize_steps_at=visualize_steps_at,
        )
        logger.debug("Solving query...")
        df = query_graph.solve()
        if filter_duplicates:
            df = df.drop_duplicates(keep="first")
        return df

    @staticmethod
    def compile(
        selection: List[ValNode],
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        version_constraints: Optional[Dict[OpKey, Optional[Set[str]]]],
        filter_duplicates: bool = True,
        call_uids: Optional[Dict[Tuple[str, int], List[str]]] = None,
    ) -> str:
        """
        Execute the given queries and return the result as a pandas DataFrame.
        """
        Querier.add_fq_constraints(fqs=fqs, call_uids=call_uids)
        compiler = Compiler(vqs=vqs, fqs=fqs)
        query = compiler.compile(
            select_queries=selection,
            filter_duplicates=filter_duplicates,
            semantic_version_constraints=version_constraints,
        )
        return query

    @staticmethod
    def add_fq_constraints(
        fqs: Set[CallNode], call_uids: Optional[Dict[Tuple[str, int], List[str]]]
    ):
        if call_uids is None:
            return
        for fq in fqs:
            if not fq.func_op.is_builtin:
                sig = fq.func_op.sig
                op_id = (sig.internal_name, sig.version)
                if op_id in call_uids:
                    fq.constraint = call_uids[op_id]

    @staticmethod
    def validate_query(
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        selection: List[ValNode],
        names: Dict[ValNode, str],
    ):
        if not selection:  # empty selection
            raise ValueError("Empty selection")
        if len(vqs) == 0:  # empty query
            raise ValueError("Query is empty")
        if not is_connected(val_queries=vqs, func_queries=fqs):  # disconnected query
            msg = f"Query is not connected! This could lead to a very large table.\n"
            logger.warning(msg)
        if not len(set(names.values())) == len(names):  # duplicate names
            duplicates = [k for k, v in Counter(names.values()).items() if v > 1]
            raise ValueError("Duplicate names in value queries: " + str(duplicates))

    @staticmethod
    def prepare_projection_query(
        vqs: Set[ValNode],
        fqs: Set[CallNode],
        selection: List[ValNode],
        name_hints: Dict[ValNode, str],
    ):
        graph = InducedSubgraph(vqs=vqs, fqs=fqs)
        v_map, f_map, _ = graph.project()
        validate_projection(
            source_selection=selection,
            v_map=v_map,
            source_selection_names={
                k: v for k, v in name_hints.items() if k in selection
            },
        )
        target_selection = [v_map[vq] for vq in selection]
        ### get the names in the projected graph
        g = InducedSubgraph(vqs=set(v_map.values()), fqs=set(f_map.values()))
        _, _, canonical_topsort = g.canonicalize()
        target_name_hints = {
            v_map[vq]: name for vq, name in name_hints.items() if vq in v_map.keys()
        }
        target_names = get_names(
            hints=target_name_hints,
            canonical_order=[vq for vq in canonical_topsort if isinstance(vq, ValNode)],
        )
        assert set(target_names.keys()) == set(v_map.values())
        return v_map, f_map, target_selection, target_names


def validate_projection(
    source_selection: List[ValNode],
    v_map: Dict[ValNode, ValNode],
    source_selection_names: Dict[ValNode, str],
):
    """
    Check that the selected nodes in the source project to distinct nodes in the
    target. Print out an error message if this is not the case.

    Here `names` is a (partial) dict from source nodes to names
    """
    fibers = defaultdict(list)
    for vq in source_selection:
        fibers[v_map[vq]].append(vq)
    if any(len(fiber) > 1 for fiber in fibers.values()):
        # find the first fiber with more than one query
        fiber = next(fiber for fiber in fibers.values() if len(fiber) > 1)
        raise ValueError(
            f"Ambiguous query: nodes {[source_selection_names.get(x, '?') for x in fiber]} have the "
            f"same role in the computational graph."
        )
