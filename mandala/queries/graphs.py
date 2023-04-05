from typing import Literal
from collections import deque
from ..core.utils import Hashing, invert_dict
from ..core.tps import StructType
from ..common_imports import *
from .weaver import ValQuery, FuncQuery, Node


def get_deps(nodes: Set[Node]) -> Set[Node]:
    res = set()

    def visit(node: Node):
        if node not in res:
            res.add(node)
            for n in node.neighbors("backward"):
                visit(n)

    for n in nodes:
        visit(n)
    return res


def prune(
    vqs: Set[ValQuery], fqs: Set[FuncQuery], selection: List[ValQuery]
) -> Tuple[Set[ValQuery], Set[FuncQuery]]:
    # remove vqs of degree 1 that are not in selection
    pass


def is_connected(val_queries: Set[ValQuery], func_queries: Set[FuncQuery]) -> bool:
    """
    Check if the graph defined by these objects is a connected graph
    """
    if len(val_queries) == 0:
        raise NotImplementedError
    visited: Dict[Node, bool] = {}

    def visit(node: Node):
        if node not in visited.keys():
            visited[node] = True
            for n in node.neighbors("both"):
                visit(n)

    start = list(val_queries)[0]
    visit(start)
    return len(visited) == len(val_queries) + len(func_queries)


def is_component(nodes: Set[Node]) -> bool:
    """
    Check if the given nodes are a full component of the graph
    """
    for node in nodes:
        if any(n not in nodes for n in node.neighbors()):
            return False
    return True


def get_fibers(v_map: Dict[Node, str]) -> Dict[str, Set[Node]]:
    fibers = defaultdict(set)
    for vq, name in v_map.items():
        fibers[name].add(vq)
    return fibers


def hash_groups(groups: Dict[str, Set[Node]]) -> str:
    return Hashing.hash_set(
        {Hashing.hash_set({str(id(v)) for v in g}) for g in groups.values()}
    )


class InducedSubgraph:
    def __init__(self, vqs: Set[ValQuery], fqs: Set[FuncQuery]):
        self.vqs = vqs
        self.fqs = fqs
        self.nodes: Set[Node] = vqs.union(fqs)

    def neighbors(
        self, node: Node, direction: Literal["forward", "backward", "both"]
    ) -> Set[Node]:
        return {n for n in node.neighbors(direction=direction) if n in self.nodes}

    def fq_inputs(self, fq: FuncQuery) -> Dict[str, ValQuery]:
        return {k: v for k, v in fq.inputs.items() if v in self.vqs}

    def fq_outputs(self, fq: FuncQuery) -> Dict[str, ValQuery]:
        return {k: v for k, v in fq.outputs.items() if v in self.vqs}

    def consumers(self, vq: ValQuery) -> List[Tuple[str, FuncQuery]]:
        return [(k, v) for k, v in zip(vq.consumed_as, vq.consumers) if v in self.fqs]

    def creators(self, vq: ValQuery) -> List[Tuple[str, FuncQuery]]:
        return [(k, v) for k, v in zip(vq.created_as, vq.creators) if v in self.fqs]

    def topsort(
        self, canonical_labels: Optional[Dict[Node, str]]
    ) -> Tuple[List[Node], Set[Node], Set[Node]]:
        """
        Get the topsort, the sources, and the sinks. Optionally, provide a
        canonical label for each node to get a canonical topsort.
        """
        # return a topological sort of the graph, and the sources and sinks
        in_degrees = {
            n: len(self.neighbors(node=n, direction="backward")) for n in self.nodes
        }
        sources = {v for v in self.nodes if in_degrees[v] == 0}
        sinks = set()
        res = []
        S = [v for v in self.nodes if in_degrees[v] == 0]
        if canonical_labels is not None:
            S = sorted(S, key=lambda x: canonical_labels[x])
        while len(S) > 0:
            v = S.pop(0)
            res.append(v)
            forward_neighbors = self.neighbors(node=v, direction="forward")
            if len(forward_neighbors) == 0:
                sinks.add(v)
            for w in self.neighbors(node=v, direction="forward"):
                in_degrees[w] -= 1
                if in_degrees[w] == 0:
                    S.append(w)
                    if canonical_labels is not None:
                        # TODO: inefficient, needs a heap instead
                        S = sorted(S, key=lambda x: canonical_labels[x])
        return res, sources, sinks

    def digest_neighbors_fq(self, fq: FuncQuery, colors: Dict[Node, str]) -> str:
        neighs = {**self.fq_inputs(fq=fq), **self.fq_outputs(fq=fq)}
        return Hashing.hash_dict({k: colors[v] for k, v in neighs.items()})

    def digest_neighbors_vq(self, vq: ValQuery, colors: Dict[Node, str]) -> str:
        creator_labels = [
            Hashing.hash_list([k, colors[v]]) for k, v in self.creators(vq=vq)
        ]
        consumer_labels = [
            Hashing.hash_list([k, colors[v]]) for k, v in self.consumers(vq=vq)
        ]
        if isinstance(vq.tp, StructType):
            lst = [
                Hashing.hash_set(set(creator_labels)),
                Hashing.hash_set(set(consumer_labels)),
            ]
        else:
            lst = [
                Hashing.hash_multiset(creator_labels),
                Hashing.hash_multiset(consumer_labels),
            ]
        return Hashing.hash_list(lst)

    def run_color_refinement(
        self, initialization: Dict[Node, str], verbose: bool = False
    ) -> Dict[Node, str]:
        """
        Run the color refinement algorithm on this graph. Return a dict mapping
        each node to its color.
        """
        colors = initialization
        groups = get_fibers(initialization)
        iteration = 0
        while True:
            if verbose:
                logger.info(f"Color refinement: iteration {iteration}")
            new_colors = {}
            for node in self.nodes:
                if isinstance(node, FuncQuery):
                    neighbors_digest = self.digest_neighbors_fq(fq=node, colors=colors)
                elif isinstance(node, ValQuery):
                    neighbors_digest = self.digest_neighbors_vq(vq=node, colors=colors)
                else:
                    raise Exception(f"Unknown node type {type(node)}")
                new_colors[node] = Hashing.hash_list([colors[node], neighbors_digest])
            new_groups = get_fibers(new_colors)
            if hash_groups(new_groups) == hash_groups(groups):
                break
            else:
                groups = new_groups
                colors = new_colors
                iteration += 1
        return colors

    @staticmethod
    def is_homomorphism(
        s: "InducedSubgraph",
        t: "InducedSubgraph",
        v_map: Dict[ValQuery, ValQuery],
        f_map: Dict[FuncQuery, FuncQuery],
    ) -> bool:
        for svq, tvq in v_map.items():
            if not isinstance(svq, ValQuery):
                return False
            if not isinstance(tvq, ValQuery):
                return False
            if not svq.tp == tvq.tp:
                return False
        for sfq, tfq in f_map.items():
            if not isinstance(sfq, FuncQuery):
                return False
            if not isinstance(tfq, FuncQuery):
                return False
            if not sfq.func_op.sig == tfq.func_op.sig:
                return False
            sfq_inps, sfq_outps = s.fq_inputs(fq=sfq), s.fq_outputs(fq=sfq)
            tfq_inps, tfq_outps = t.fq_inputs(fq=tfq), t.fq_outputs(fq=tfq)
            if not set(sfq_inps.keys()) == set(tfq_inps.keys()):
                return False
            if not set(sfq_outps.keys()) == set(tfq_outps.keys()):
                return False
            if not all(v_map[v] == tfq_inps[k] for k, v in sfq_inps.items()):
                return False
            if not all(v_map[v] == tfq_outps[k] for k, v in sfq_outps.items()):
                return False
        return True

    @staticmethod
    def are_canonically_isomorphic(s: "InducedSubgraph", t: "InducedSubgraph") -> bool:
        try:
            s_vlabels, s_flabels, s_topsort = s.canonicalize(strict=True)
            t_vlabels, t_flabels, t_topsort = t.canonicalize(strict=True)
        except AssertionError:
            raise NotImplementedError
        if len(s_vlabels) != len(t_vlabels) or len(s_flabels) != len(t_flabels):
            return False
        t_vinverse, t_finverse = invert_dict(t_vlabels), invert_dict(t_flabels)
        v_map = {vq: t_vinverse[v] for vq, v in s_vlabels.items()}
        f_map = {fq: t_finverse[f] for fq, f in s_flabels.items()}
        if not InducedSubgraph.is_homomorphism(s, t, v_map, f_map):
            return False
        if not InducedSubgraph.is_homomorphism(
            t, s, invert_dict(v_map), invert_dict(f_map)
        ):
            return False
        return True

    def get_projections(
        self, colors: Dict[Node, str]
    ) -> Tuple[Dict[ValQuery, ValQuery], Dict[FuncQuery, FuncQuery]]:
        v_groups = defaultdict(list)
        f_groups = defaultdict(list)
        for node, color in colors.items():
            if isinstance(node, ValQuery):
                v_groups[color].append(node)
            elif isinstance(node, FuncQuery):
                f_groups[color].append(node)
        v_map: Dict[ValQuery, ValQuery] = {}
        f_map: Dict[FuncQuery, FuncQuery] = {}
        for color, gp in v_groups.items():
            representative = gp[0]
            prototype = ValQuery(
                tp=representative.tp, constraint=representative.constraint
            )
            with_constraint = [vq for vq in gp if vq.constraint is not None]
            if len(with_constraint) == 0:
                representative_constraint = None
            elif len(with_constraint) == 1:
                representative_constraint = with_constraint[0].constraint
            else:
                raise ValueError("Multiple constraints for a single value")
            prototype = ValQuery(
                tp=representative.tp, constraint=representative_constraint
            )
            for fq in gp:
                v_map[fq] = prototype
        for color, gp in f_groups.items():
            representative = gp[0]
            rep_inps = self.fq_inputs(fq=representative)
            rep_outps = self.fq_outputs(fq=representative)
            prototype = FuncQuery.link(
                inputs={k: v_map[vq] for k, vq in rep_inps.items()},
                outputs={k: v_map[vq] for k, vq in rep_outps.items()},
                func_op=representative.func_op,
                orientation=representative.orientation,
                constraint=None,
            )
            for fq in gp:
                f_map[fq] = prototype
        t = InducedSubgraph(vqs=set(v_map.values()), fqs=set(f_map.values()))
        assert self.is_homomorphism(s=self, t=t, v_map=v_map, f_map=f_map)
        return v_map, f_map

    def canonicalize(
        self, strict: bool = False, method: str = "wl1"
    ) -> Tuple[Dict[ValQuery, str], Dict[FuncQuery, str], List[Node]]:
        """
        Return canonical labels for each node, as well as a canonical topological sort.
        """
        initialization = {}
        for vq in self.vqs:
            initialization[vq] = "null"
        for fq in self.fqs:
            initialization[fq] = fq.func_op.sig.versioned_internal_name
        colors = self.run_color_refinement(initialization=initialization)
        v_colors = {vq: colors[vq] for vq in self.vqs}
        f_colors = {fq: colors[fq] for fq in self.fqs}
        if strict:
            assert len(set(v_colors.values())) == len(v_colors)
            assert len(set(f_colors.values())) == len(f_colors)
        canonical_topsort, sources, sinks = self.topsort(
            canonical_labels={**v_colors, **f_colors}
        )
        return v_colors, f_colors, canonical_topsort

    def project(
        self,
    ) -> Tuple[Dict[ValQuery, ValQuery], Dict[FuncQuery, FuncQuery], List[Node]]:
        v_colors, f_colors, topsort = self.canonicalize()
        colors: Dict[Node, str] = {**v_colors, **f_colors}
        return *self.get_projections(colors=colors), topsort


def get_canonical_order(vqs: Set[ValQuery], fqs: Set[FuncQuery]) -> List[ValQuery]:
    g = InducedSubgraph(vqs=vqs, fqs=fqs)
    _, _, canonical_topsort = g.canonicalize()
    return [vq for vq in canonical_topsort if isinstance(vq, ValQuery)]


def copy_subgraph(
    vqs: Set[ValQuery], fqs: Set[FuncQuery]
) -> Tuple[Dict[ValQuery, ValQuery], Dict[FuncQuery, FuncQuery]]:
    """
    Copy the subgraph supported on the given nodes. Return maps from the
    original nodes to their copies.
    """
    v_map = {v: ValQuery(tp=v.tp, constraint=v.constraint, name=v.name) for v in vqs}
    f_map = {}
    for fq in fqs:
        inputs = {k: v_map[v] for k, v in fq.inputs.items() if v in vqs}
        outputs = {k: v_map[v] for k, v in fq.outputs.items() if v in vqs}
        f_map[fq] = FuncQuery.link(
            inputs=inputs,
            outputs=outputs,
            func_op=fq.func_op,
            orientation=fq.orientation,
            constraint=fq.constraint,
        )
    return v_map, f_map
