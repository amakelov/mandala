from .common_imports import *
from .common_imports import sess
import textwrap
from .utils import (
    get_nullable_union,
    get_setdict_union,
    get_setdict_intersection,
    get_dict_union_over_keys,
    get_dict_intersection_over_keys,
    get_adjacency_union,
    get_adjacency_intersection,
    get_adj_from_edges,
    invert_dict,
    get_nullable_intersection,
)
from .model import Call, Ref, Op, __make_list__

from .viz import Node, Edge, SOLARIZED_LIGHT, to_dot_string, write_output


def get_name_proj(op: Op) -> Callable[[str], str]:
    if op.name == __make_list__.name:
        return lambda x: "elts" if x.startswith("elts_") else x
    else:
        return lambda x: x


def get_reverse_proj(call: Call) -> Callable[[str], Set[str]]:
    if call.op.name == __make_list__.name:
        return lambda x: {k for k in call.inputs} if x == "elts" else {x}
    else:
        return lambda x: {x}


class ComputationFrame:
    """
    A view of a slice of storage.

    - use `.info()` to get summary information about this `ComputationFrame`
    - use `.info(*variables)` or `.info(*functions)` to get information 
    about specific variables/functions in this `ComputationFrame`
    """

    def __init__(
        self,
        storage: "Storage",
        # graph schema
        inp: Dict[str, Dict[str, Set[str]]] = None,  # node name -> input name -> {node name}
        out: Dict[str, Dict[str, Set[str]]] = None,  # node name -> output name -> {node name}
        # graph instance data
        vs: Dict[str, Set[str]] = None,  # variable name -> {history_id}: the set of refs in the variable
        fs: Dict[str, Set[str]] = None,  # function name -> {history_id}: the set of calls in the function
        refinv: Dict[str, Set[str]] = None,  # (ref) history_id -> {variable name}: the set of variables containing the ref
        callinv: Dict[str, Set[str]] = None,  # (call) history_id -> {function name}: the set of functions containing the call
        creator: Dict[str, str] = None,  # (ref) history_id -> (call) history_id: the call that created the ref, if any
        consumers: Dict[str, Set[str]] = None,  # (ref) history_id -> {(call) history_id}: the calls that consume the ref
        # objects
        refs: Dict[str, Union[Ref, Any]] = None,  # history_id -> Ref
        calls: Dict[str, Call] = None,  # history_id -> Call
    ):
        self.storage = storage
        self.inp = {} if inp is None else inp
        self.out = {} if out is None else out

        self.vs = {} if vs is None else vs
        self.fs = {} if fs is None else fs
        self.refinv = {} if refinv is None else refinv
        self.callinv = {} if callinv is None else callinv
        self.creator = {} if creator is None else creator
        self.consumers = {} if consumers is None else consumers

        self.refs = {} if refs is None else refs
        self.calls = {} if calls is None else calls

    def _check(self):
        """
        Check all invariants not enforced by the data structure
        """
        assert get_nullable_union(*self.vs.values()) == set(self.refs.keys())
        assert get_nullable_union(*self.fs.values()) == set(self.calls.keys())
        assert get_nullable_union(*self.refinv.values()) <= set(self.vs.keys())
        assert get_nullable_union(*self.callinv.values()) <= set(self.fs.keys())
        assert set(self.creator.keys()) <= set(self.refs.keys())
        assert set(self.creator.values()) <= set(self.calls.keys())
        assert set(self.consumers.keys()) <= set(self.refs.keys())
        assert get_nullable_union(*self.consumers.values()) <= set(self.calls.keys())
        edges = self.edges()
        all_nodes = self.vs.keys() | self.fs.keys()
        assert all([src in all_nodes and dst in all_nodes for src, dst, _ in edges])
        # check the calls
        for fname, call_uids in self.fs.items():
            #! we check that, for each call belonging to an op node, all the
            # inputs of this call for which there is an edge belong in the graph,
            # and similarly for all the outputs. This is the least obvious
            # invariant.
            calls = [self.calls[call_uid] for call_uid in call_uids]
            for call in calls:
                for input_name, inp in call.inputs.items():
                    input_proj = get_name_proj(call.op)(input_name)
                    for input_var in self.inp[fname].get(input_proj, set()):
                        assert inp.hid in self.vs[input_var]
                        assert input_var in self.refinv[inp.hid]
                for output_name, out in call.outputs.items():
                    output_proj = get_name_proj(call.op)(output_name)
                    for output_var in self.out[fname].get(output_proj, set()):
                        assert out.hid in self.vs[output_var]
                        assert output_var in self.refinv[out.hid]
        assert all(k == ref.hid for k, ref in self.refs.items())
        assert all(k == call.hid for k, call in self.calls.items())

    @property
    def vnames(self) -> Set[str]:
        return set(self.vs.keys())

    @property
    def fnames(self) -> Set[str]:
        return set(self.fs.keys())

    @property
    def nodes(self) -> Set[str]:
        return self.vnames | self.fnames

    ############################################################################
    ### atomic operations on the CF (and their batched versions)
    ############################################################################
    def add_var(self, vname: Optional[str]) -> str:
        if vname is None:
            vname = self.get_new_vname("v")
        self.vs[vname] = set()
        self.inp[vname] = {}
        self.out[vname] = {}
        return vname

    def drop_node(self, node: str):
        if node in self.vs:
            self.drop_var(node)
        elif node in self.fs:
            self.drop_func(node)
        else:
            raise ValueError(f"Node {node} not found in the CF")

    def drop_var(self, vname: str):
        for (src, dst, label) in self.edges():
            if src == vname or dst == vname:
                self.drop_edge(src, dst, label)
        hids = list(self.vs[vname])
        for hid in hids:
            self.drop_ref(vname, hid)
        del self.vs[vname]
        del self.inp[vname]
        del self.out[vname]

    def rename_var(self, vname: str, new_vname: str, inplace: bool = False) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        res.vs[new_vname] = res.vs[vname]
        res.inp[new_vname] = res.inp[vname]
        res.out[new_vname] = res.out[vname]
        del res.vs[vname]
        del res.inp[vname]
        del res.out[vname]
        # now, rename the node in the adjacency lists
        for node in res.nodes:
            for neighbor_nodes in res.inp[node].values():
                if vname in neighbor_nodes:
                    neighbor_nodes.remove(vname)
                    neighbor_nodes.add(new_vname)
            for neighbor_nodes in res.out[node].values():
                if vname in neighbor_nodes:
                    neighbor_nodes.remove(vname)
                    neighbor_nodes.add(new_vname)
        for ref in res.vs[new_vname]:
            res.refinv[ref].remove(vname)
            res.refinv[ref].add(new_vname)
        return res if not inplace else None
    
    def rename(self, 
               vars: Optional[Dict[str, str]] = None,
               funcs: Optional[Dict[str, str]] = None,
               inplace: bool = False) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        if vars is not None:
            for vname, new_vname in vars.items():
                res.rename_var(vname, new_vname, inplace=True)
        if funcs is not None:
            raise NotImplementedError
        return res if not inplace else None

    def add_func(
        self,
        fname: Optional[str],
    ) -> str:
        if fname is None:
            fname = self.get_new_fname("f")
        logging.debug(
            f"Adding function {fname}, called from {inspect.stack()[1].function}"
        )
        self.fs[fname] = set()
        self.inp[fname] = {}
        self.out[fname] = {}
        return fname

    def drop_func(self, fname: str):
        for (src, dst, label) in self.edges():
            if src == fname or dst == fname:
                self.drop_edge(src, dst, label)
        hids = self.fs[fname]
        for hid in hids:
            self.drop_call(fname, hid)
        del self.fs[fname]
        del self.inp[fname]
        del self.out[fname]


    def add_edge(self, src: str, dst: str, label: str):
        if label not in self.out[src]:
            self.out[src][label] = set()
        self.out[src][label].add(dst)
        if label not in self.inp[dst]:
            self.inp[dst][label] = set()
        self.inp[dst][label].add(src)

    def drop_edge(self, src: str, dst: str, label: str):
        self.out[src][label].remove(dst)
        self.inp[dst][label].remove(src)

    def edges(self) -> List[Tuple[str, str, str]]:
        return [
            (src, dst, label)
            for src, dsts_dict in self.out.items()
            for label, dsts in dsts_dict.items()
            for dst in dsts
        ]

    def add_ref(self, vname: str, ref: Ref):
        self.refs[ref.hid] = ref
        self.vs[vname].add(ref.hid)
        if ref.hid not in self.refinv:
            self.refinv[ref.hid] = set()
        self.refinv[ref.hid].add(vname)

    def drop_ref(self, vname: str, hid: str):
        self.vs[vname].remove(hid)
        if (
            len(self.refinv[hid]) == 1
        ):  # if this is the only variable containing the ref
            del self.refs[hid]
            del self.refinv[hid]
            if hid in self.creator:
                del self.creator[hid]
            if hid in self.consumers:
                del self.consumers[hid]
        else:  # if there are other variables containing the ref
            self.refinv[hid].remove(vname)

    def add_call(self, fname: str, call: Call, with_refs: bool):
        self.calls[call.hid] = call
        self.fs[fname].add(call.hid)
        if call.hid not in self.callinv:
            self.callinv[call.hid] = set()
        self.callinv[call.hid].add(fname)
        if with_refs:
            for input_name, input_ref in call.inputs.items():
                input_proj = get_name_proj(call.op)(input_name)
                # skip over inputs not tracked by the graph
                if input_proj not in self.inp[fname]:
                    continue
                # add to consumers
                if input_ref.hid not in self.consumers:
                    self.consumers[input_ref.hid] = set()
                self.consumers[input_ref.hid].add(call.hid)
                # add ref to corresponding variables
                for vname in self.inp[fname][input_proj]:
                    self.add_ref(vname, input_ref)
            for output_name, output_ref in call.outputs.items():
                output_proj = get_name_proj(call.op)(output_name)
                # skip over outputs not tracked by the graph
                if output_proj not in self.out[fname]:
                    continue
                # add to creator
                self.creator[output_ref.hid] = call.hid
                # add ref to corresponding variables
                for vname in self.out[fname][output_proj]:
                    self.add_ref(vname, output_ref)

    def drop_call(self, fname: str, hid: str):
        """
        Drop a call from the given function node only. This checks
        if the call is referenced by multiple function nodes.
        """
        if (
            len(self.callinv[hid]) == 1
        ):  # if this is the only function containing the call
            call = self.calls[hid]
            del self.calls[hid]
            del self.callinv[hid]
            for output in call.outputs.values():
                del self.creator[output.hid]
            for inp in call.inputs.values():
                if inp.hid in self.consumers:
                    self.consumers[inp.hid].remove(hid)
                    if len(self.consumers[inp.hid]) == 0:
                        del self.consumers[inp.hid]
        else:  # if there are other functions containing the call
            self.callinv[hid].remove(fname)

    def in_neighbors(self, node: str) -> Set[str]:
        return get_nullable_union(*[self.inp[node][k] for k in self.inp[node]])

    def out_neighbors(self, node: str) -> Set[str]:
        return get_nullable_union(*[self.out[node][k] for k in self.out[node]])

    def in_neighbor_elts(self, node: str, hids: Set[str]) -> Dict[str, Set[str]]:
        pass

    def in_edges(self, node: str) -> Set[Tuple[str, str, str]]:
        """
        Return (source, destination, label) tuples for all edges pointing to the
        given node.
        """
        return {(src, dst, label) for src, dst, label in self.edges() if dst == node}

    def out_edges(self, node: str) -> Set[Tuple[str, str, str]]:
        """
        Return (source, destination, label) tuples for all edges pointing from
        the given node.
        """
        return {(src, dst, label) for src, dst, label in self.edges() if src == node}

    @property
    def sources(self) -> Set[str]:
        return {node for node in self.vs.keys() if len(self.inp[node]) == 0}

    def get_source_elts(self) -> Dict[str, Set[str]]:
        """
        Get a view of the elements in the CF which are not connected as outputs
        to any element in the CF.

        Important distinctions from other natural ways of defining "sources":
            - this is **not the same** as getting the elements in the source nodes
            of the graph. There could be an intermediate node where some of the
            elements have a history in the graph and some don't.
            - this is also **not the same** as getting the elements that do not
            have a creator *anywhere* in the graph. For example, there could be 
            a ref in some node which has a creator call somewhere in the graph,
            but the node is not connected as an output to the node containing
            that call.
        """
        res = {}
        for node in self.nodes:
            in_edges = self.in_edges(node)
            hids_with_creators = set()
            for src, dst, label in in_edges:
                hids_with_creators |= self.get_adj_elts_edge((src, dst, label), self.sets[src], "forward")
            res[node] = self.sets[node] - hids_with_creators
        return res

    @property
    def sinks(self) -> Set[str]:
        return {node for node in self.vs.keys() if len(self.out[node]) == 0}

    def get_sink_elts(self) -> Dict[str, Set[str]]:
        """
        Get a view of the elements in the CF which are not connected as inputs
        to any element in the CF.

        Similar caveats as for `get_source_elts`.
        """
        res = {}
        for node in self.nodes:
            out_edges = self.out_edges(node)
            hids_with_consumers = set()
            for src, dst, label in out_edges:
                hids_with_consumers |= self.get_adj_elts_edge((src, dst, label), self.sets[dst], "back")
            res[node] = self.sets[node] - hids_with_consumers
        return res

    ############################################################################
    ### core "low-level" interface
    ############################################################################
    def get_names_projecting_to(self, call_hid: str, label: str) -> Set[str]:
        """
        Get the names of the inputs of the call with the given hid that project
        to the given label.
        """
        return get_reverse_proj(self.calls[call_hid])(label)

    def get_io_proj(self, call_hid: str, name: str) -> str:
        """
        Get the edge label the given I/O name of the call projects to
        """
        return get_name_proj(self.calls[call_hid].op)(name)

    def get_adj_elts_edge(
        self,
        edge: Tuple[str, str, str],
        hids: Set[str],
        direction: Literal["back", "forward"],
    ) -> Set[str]:
        """
        Return the elements connected to the given elements along the given edge
        """
        src, dst, label = edge
        assert direction in ("back", "forward")
        if direction == "back":
            if (
                dst in self.vs
            ):  # this is a variable node; we are looking for the calls that created the hids
                assert hids <= self.vs[dst]  # make sure the hids are valid
                return {
                    self.creator[hid] for hid in hids if hid in self.creator
                } & self.fs[src]
            else:  # this is an op node; we are looking for the inputs of the calls that have the hids as outputs
                assert hids <= self.fs[dst]  # make sure the hids are valid
                return self.vs[src] & {
                    self.calls[call_hid].inputs[name].hid
                    for call_hid in hids
                    for name in self.get_names_projecting_to(call_hid, label)
                    if name in self.calls[call_hid].inputs
                }
        else:  # direction == "forward"
            if (
                src in self.vs
            ):  # this is a variable node; we are looking for the calls that consume the hids
                assert hids <= self.vs[src]
                return {
                    call_hid
                    for call_hid in self.fs[dst]
                    if any(
                        self.calls[call_hid].inputs[name].hid in hids
                        for name in self.get_names_projecting_to(call_hid, label)
                        if name in self.calls[call_hid].inputs.keys()
                    )
                }
            else:  # this is an op node; we are looking for the outputs of the calls that have the hids as inputs
                assert hids <= self.fs[src]
                return {
                    self.calls[call_hid].outputs[name].hid
                    for call_hid in hids
                    for name in self.get_names_projecting_to(call_hid, label)
                    if name in self.calls[call_hid].outputs
                } & self.vs[dst]

    def get_adj_elts(
        self, node: str, hids: Set[str], direction: Literal["back", "forward", "both"]
    ) -> Dict[str, Set[str]]:
        """
        Given a node, and a subset of the elements in this node, return a view
        of the elements of the adjacent nodes (in the given direction(s)) that
        are connected to the given elements.
        """
        res = {}
        if direction in ["back", "both"]:
            edges = self.in_edges(node)
            res.update(
                {
                    src: self.get_adj_elts_edge((src, dst, label), hids, "back")
                    for (src, dst, label) in edges
                }
            )
        if direction in ["forward", "both"]:
            edges = self.out_edges(node)
            res.update(
                {
                    dst: self.get_adj_elts_edge((src, dst, label), hids, "forward")
                    for (src, dst, label) in edges
                }
            )
        return res

    def select_nodes(self, nodes: Iterable[str]) -> "ComputationFrame":
        """
        Get the induced computation frame on these nodes, taking all edges
        connecting pairs of these nodes.
        """
        #! must copy all sets etc. to avoid modifying the original CF
        nodes_not_in_cf = set(nodes) - self.nodes
        if len(nodes_not_in_cf) > 0:
            raise ValueError(f"Nodes {nodes_not_in_cf} not found in the CF")
        res = self.copy()
        nodes = set(nodes)
        edges = {
            (src, dst, label)
            for src, dst, label in res.edges()
            if (src in nodes and dst in nodes)
        }
        out, inp = get_adj_from_edges(edges, node_support=nodes)
        vs = {vname: hids for vname, hids in res.vs.items() if vname in nodes}
        fs = {fname: call_uids for fname, call_uids in res.fs.items() if fname in nodes}
        ref_hids = get_nullable_union(*vs.values())
        call_hids = get_nullable_union(*fs.values())
        refs = {hid: ref for hid, ref in res.refs.items() if hid in ref_hids}
        calls = {hid: call for hid, call in res.calls.items() if hid in call_hids}
        refinv = {
            hid: vnames & set(nodes)
            for hid, vnames in res.refinv.items()
            if hid in ref_hids
        }
        callinv = {
            hid: fnames & set(nodes)
            for hid, fnames in res.callinv.items()
            if hid in call_hids
        }
        creator = {
            ref_hid: call_hid
            for ref_hid, call_hid in res.creator.items()
            if ref_hid in ref_hids and call_hid in call_hids
        }
        consumers = {
            ref_hid: (consumer_call_hids & call_hids)
            for ref_hid, consumer_call_hids in res.consumers.items()
            if ref_hid in ref_hids
        }
        return ComputationFrame(
            storage=res.storage,
            inp=inp,
            out=out,
            vs=vs,
            fs=fs,
            refinv=refinv,
            callinv=callinv,
            creator=creator,
            consumers=consumers,
            refs=refs,
            calls=calls,
        )

    def select_subsets(self, elts: Dict[str, Set[str]]) -> "ComputationFrame":
        """
        Restrict the CF to the given elements in each node
        """
        #! must copy all sets etc. to avoid modifying the original CF
        res = self.copy()
        assert set(elts.keys()) == res.nodes
        sets = res.sets
        assert all(elts[node] <= sets[node] for node in elts.keys())
        out = res.out
        inp = res.inp
        vs = {node: elts[node] for node in res.vnames}
        fs = {node: elts[node] for node in res.fnames}
        ref_hids_subset = get_nullable_union(*vs.values())
        call_hids_subset = get_nullable_union(*fs.values())
        refs = {hid: res.refs[hid] for hid in ref_hids_subset}
        calls = {hid: res.calls[hid] for hid in call_hids_subset}
        refinv = {hid: res.refinv[hid] for hid in ref_hids_subset}
        callinv = {hid: res.callinv[hid] for hid in call_hids_subset}
        creator = {
            hid: res.creator[hid]
            for hid in ref_hids_subset
            if hid in res.creator and res.creator[hid] in call_hids_subset
        }
        consumers = {
            hid: res.consumers[hid] & call_hids_subset
            for hid in ref_hids_subset
            if hid in res.consumers
        }
        return ComputationFrame(
            storage=res.storage,
            inp=inp,
            out=out,
            vs=vs,
            fs=fs,
            refinv=refinv,
            callinv=callinv,
            creator=creator,
            consumers=consumers,
            refs=refs,
            calls=calls,
        )

    def downstream(self, *nodes: str, how: Literal["strong", "weak"] = "strong") -> "ComputationFrame":
        """
        Return the sub-`ComputationFrame` representing the computations
        downstream of the elements in the given `nodes`.

        The result is obtained by "simulating" the computation encoded in the
        computation frame's graph, starting from the elements in the `nodes` and
        going only forward in the graph, adding only calls for which all inputs
        downstream of `nodes` have already been added to the result.

        In particular, this removes the dependents of "parasitic" values which
        were computed by calls that are not downstream of `nodes`, but were
        originally part of the computation frame.
        """
        downstream_nodes = set()
        sources = set(nodes)
        while len(sources) > 0:
            source = sources.pop()
            downstream_nodes.add(source)
            for label, dsts in self.out[source].items():
                sources |= dsts
        res = self.select_nodes(downstream_nodes)
        downstream_view = res.get_reachable_elts(
            initial_state={node: res.sets[node] for node in nodes},
            direction="forward",
            how=how,
        )
        return res.select_subsets(downstream_view)

    def upstream(self, *nodes: str, how: Literal["strong", "weak"] = "strong") -> "ComputationFrame":
        """
        Return the sub-`ComputationFrame` representing the computations upstream
        of the elements in the given `nodes`.

        Like `downstream`, but in reverse.
        """
        upstream_nodes = set()
        sinks = set(nodes)
        while len(sinks) > 0:
            sink = sinks.pop()
            upstream_nodes.add(sink)
            for label, srcs in self.inp[sink].items():
                sinks |= srcs
        res = self.select_nodes(upstream_nodes)
        upstream_view = res.get_reachable_elts(
            initial_state={node: res.sets[node] for node in nodes},
            direction="back",
            how=how,
        )
        return res.select_subsets(upstream_view)

    def midstream(self, *nodes: str, how: Literal["strong", "weak"] = "strong") -> "ComputationFrame":
        """
        Get the subprogram "spanned by" the given nodes. 

        Specifically, this means:
        - obtain all nodes on paths between the given nodes;
        - restrict to the subgraph induced by these nodes;
        - restrict to the computations which are reachable (in a sense
        controlled by the `how` parameter) from **the source elements in this
        subgraph that belong to a node in `nodes`**. 
        """
        paths = set()
        for start_node in nodes:
            for end_node in nodes:
                if start_node != end_node:
                    paths |= self.get_all_vertex_paths(start_node, end_node)
        midstream_nodes = get_nullable_union(*[set(path) for path in paths])
        logging.debug(f'Selected {midstream_nodes} from {self.nodes}')
        # first, restrict to the subgraph induced by the midstream nodes
        pre_res = self.select_nodes(midstream_nodes)
        # now, figure out the source elements
        source_elts = {k: v for k, v in pre_res.get_source_elts().items() if k in nodes}
        # compute reachability
        res_view = pre_res.get_reachable_elts(
            initial_state=source_elts, 
            direction="forward",
            how=how,
        )
        return pre_res.select_subsets(res_view)
        # return res.downstream(*res.sources, how=how)

    ############################################################################
    ### union and intersection (over shared namespaces)
    ############################################################################
    @staticmethod
    def _binary_union(
        a: "ComputationFrame", b: "ComputationFrame"
    ) -> "ComputationFrame":
        """
        What it says on the tin: take the union over both the topology and the
        instance data of the two computation frames.
        """
        inp = get_adjacency_union(a.inp, b.inp)
        out = get_adjacency_union(a.out, b.out)
        vs = get_setdict_union(a.vs, b.vs)
        fs = get_setdict_union(a.fs, b.fs)
        refinv = get_setdict_union(a.refinv, b.refinv)
        callinv = get_setdict_union(a.callinv, b.callinv)
        #! this is correct because there is at most 1 creator per ref
        creator = get_dict_union_over_keys(a.creator, b.creator)
        consumers = get_setdict_union(a.consumers, b.consumers)
        refs = get_dict_union_over_keys(a.refs, b.refs)
        calls = get_dict_union_over_keys(a.calls, b.calls)
        res = ComputationFrame(
            storage=a.storage,
            inp=inp,
            out=out,
            vs=vs,
            fs=fs,
            refinv=refinv,
            callinv=callinv,
            creator=creator,
            consumers=consumers,
            refs=refs,
            calls=calls,
        )
        res._check()  #! not every union results in a valid CF
        return res

    @staticmethod
    def _binary_intersection(
        a: "ComputationFrame", b: "ComputationFrame"
    ) -> "ComputationFrame":
        """
        ditto, but for intersection
        """
        inp = get_adjacency_intersection(a.inp, b.inp)
        out = get_adjacency_intersection(a.out, b.out)
        vs = get_setdict_intersection(a.vs, b.vs)
        fs = get_setdict_intersection(a.fs, b.fs)
        refinv = get_setdict_intersection(a.refinv, b.refinv)
        callinv = get_setdict_intersection(a.callinv, b.callinv)
        creator = get_dict_intersection_over_keys(a.creator, b.creator)
        consumers = get_setdict_intersection(a.consumers, b.consumers)
        refs = get_dict_intersection_over_keys(a.refs, b.refs)
        calls = get_dict_intersection_over_keys(a.calls, b.calls)
        res = ComputationFrame(
            storage=a.storage,
            inp=inp,
            out=out,
            vs=vs,
            fs=fs,
            refinv=refinv,
            callinv=callinv,
            creator=creator,
            consumers=consumers,
            refs=refs,
            calls=calls,
        )
        # every intersection is a valid CF
        return res

    @staticmethod
    def _binary_setwise_difference(
        a: "ComputationFrame", b: "ComputationFrame"
    ) -> "ComputationFrame":
        """
        Remove all elements in `b` that are present in respective nodes of `a`,
        then turn the result into a valid `ComputationFrame` by enforcing the
        invariants.
        """
        to_keep = {}  # node -> {element}
        for node, st in a.sets.items():
            if node not in b.sets:
                to_keep[node] = st.copy()  #! we must copy the set to avoid side effects
            else:
                to_keep[node] = st - b.sets[node]
        res = a.select_subsets(elts=to_keep)
        #! not every symmetric difference results in a valid CF.
        # We fix the mess by enforcing invariants via computational reachability
        res = res.drop_unreachable(direction="forward", how="strong")
        res.cleanup(inplace=True)
        return res

    @staticmethod
    def _binary_difference(
        a: "ComputationFrame", b: "ComputationFrame"
    ) -> "ComputationFrame":
        """
        Remove elements AND nodes/edges that are in b from a. The topology of `a`
        is NOT preserved in this case.
        """
        raise NotImplementedError()

    @staticmethod
    def union(*cfs: "ComputationFrame") -> "ComputationFrame":
        from functools import reduce

        return reduce(ComputationFrame._binary_union, cfs)

    @staticmethod
    def intersection(*cfs: "ComputationFrame") -> "ComputationFrame":
        from functools import reduce

        return reduce(ComputationFrame._binary_intersection, cfs)

    def __or__(self, other: "ComputationFrame") -> "ComputationFrame":
        return ComputationFrame._binary_union(self, other)

    def __and__(self, other: "ComputationFrame") -> "ComputationFrame":
        return ComputationFrame._binary_intersection(self, other)

    def __sub__(self, other: "ComputationFrame") -> "ComputationFrame":
        """
        WARNING: the difference operator "-" defaults to setwise difference, not
        the difference of the graph structure.
        """
        return ComputationFrame._binary_setwise_difference(self, other)

    ############################################################################
    ### back/forward expansion
    ############################################################################
    def _group_calls(
        self,
        calls: List[Call],
        available_nodes: Iterable[str],
        by: Literal["inputs", "outputs"],
    ) -> Dict[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]]]], List[Call]]:
        """
        Group calls by how they connect to existing variables in the CF along
        either their inputs or outputs. The calls may or may not be already
        present in the CF; the groups are only based on where the inputs/outputs
        of the calls are found in the CF.

        The (canonical) key of the group is a tuple of:
        - the op id;
        - a (sorted) tuple of elements of the form
            (input/output label, sorted connected variables to this input/output)
        """
        call_groups = {}
        for call in calls:
            op_id = call.op.id
            identifying_refs = call.inputs if by == "inputs" else call.outputs
            identifying_variables = {}  # input/output label -> {variable name}
            for io_name, ref in identifying_refs.items():
                label = get_name_proj(op=call.op)(io_name)
                if ref.hid in self.refs:
                    connected_vnames = self.refinv[ref.hid] & set(available_nodes)
                    if not connected_vnames:
                        continue
                    identifying_variables[label] = connected_vnames
            # get a hashable canonical representation of the group
            group_id = (
                op_id,
                tuple(
                    sorted(
                        (label, tuple(sorted(variables)))
                        for label, variables in identifying_variables.items()
                    )
                ),
            )
            if group_id not in call_groups:
                call_groups[group_id] = []
            call_groups[group_id].append(call)
        return call_groups

    def _expand_from_call_groups(
        self,
        call_groups: Dict[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]]]], List[Call]],
        side_to_glue: Literal["inputs", "outputs"],
    ):
        for group_id, group_calls in sorted(call_groups.items()):
            op_id, connected_vnames_tuple = group_id
            label_to_connected_vnames = {
                label: vnames for label, vnames in connected_vnames_tuple
            }
            funcname = self.add_func(fname=self.get_new_fname(op_id))
            for label, connected_vnames in label_to_connected_vnames.items():
                if not connected_vnames:
                    continue
                for connected_vname in connected_vnames:
                    src = funcname if side_to_glue == "outputs" else connected_vname
                    dst = connected_vname if side_to_glue == "outputs" else funcname
                    self.add_edge(src, dst, label)
            if side_to_glue == "outputs":
                # dual_side_labels = get_nullable_union(*[set(call.inputs.keys()) for call in group_calls])
                dual_side_labels_list = []
                for call in group_calls:
                    name_projector = get_name_proj(call.op)
                    dual_side_labels_list.append(
                        set([name_projector(name) for name in call.inputs])
                    )
                dual_side_labels = get_nullable_union(*dual_side_labels_list)
            elif side_to_glue == "inputs":
                # dual_side_labels = get_nullable_union(*[set(call.outputs.keys()) for call in group_calls])
                dual_side_labels_list = []
                for call in group_calls:
                    name_projector = get_name_proj(call.op)
                    dual_side_labels_list.append(
                        set([name_projector(name) for name in call.outputs])
                    )
                dual_side_labels = get_nullable_union(*dual_side_labels_list)
            for label in dual_side_labels:
                varname = self.add_var(vname=self.get_new_vname(label))
                src = varname if side_to_glue == "outputs" else funcname
                dst = funcname if side_to_glue == "outputs" else varname
                self.add_edge(src, dst, label)
            for call in group_calls:
                self.add_call(funcname, call, with_refs=True)

    def get_creators(self, indexer: Union[str, Set[str]]):
        # return information about the calls that created the variables in the
        # indexer without actually adding them to the CF
        raise NotImplementedError()

    def get_consumers(self, indexer: Union[str, Set[str]]):
        # return information about the calls that consume the variables in the
        # indexer without actually adding them to the CF
        raise NotImplementedError()

    def expand_back(
        self,
        varnames: Optional[Union[str, Set[str]]] = None,
        skip_existing: bool = True,
        inplace: bool = False,
    ) -> Optional["ComputationFrame"]:
        """
        Join the calls that created the given variables.
        """
        res = self if inplace else self.copy()
        if varnames is None:  # full expansion until fixed point
            while True:
                current_size = len(res.nodes)
                # sources = res.sources
                sources = res.vs.keys()
                res.expand_back(sources, inplace=True, skip_existing=skip_existing)
                new_size = len(res.nodes)
                if new_size == current_size:
                    break
            return res if not inplace else None
        if isinstance(varnames, str):
            varnames = {varnames}
        ref_uids = set.union(*[res.vs[v] for v in varnames])
        calls = res.storage.get_creators(ref_uids)
        if skip_existing:
            calls = [call for call in calls if call.hid not in res.calls]
        call_groups = res._group_calls(calls, by="outputs", available_nodes=varnames)
        logging.debug(f"Found call groups with keys: {call_groups.keys()}")
        res._expand_from_call_groups(
            call_groups,
            side_to_glue="outputs",
        )
        return res if not inplace else None

    def expand_forward(
        self,
        varnames: Optional[Union[str, Set[str]]] = None,
        skip_existing: bool = True,
        inplace: bool = False,
    ) -> Optional["ComputationFrame"]:
        """
        Join the calls that consume the given variables.
        """
        res = self if inplace else self.copy()
        if varnames is None:  # full expansion until fixed point
            while True:
                current_size = len(res.nodes)
                # sinks = res.sinks
                sinks = res.vs.keys()
                res.expand_forward(sinks, inplace=True, skip_existing=skip_existing)
                new_size = len(res.nodes)
                if new_size == current_size:
                    break
            return res if not inplace else None
        if isinstance(varnames, str):
            varnames = {varnames}
        ref_hids = set.union(*[res.vs[v] for v in varnames])
        calls = res.storage.get_consumers(ref_hids)
        if skip_existing:
            calls = [call for call in calls if call.hid not in res.calls]
        call_groups = res._group_calls(calls, by="inputs", available_nodes=varnames)
        res._expand_from_call_groups(
            call_groups,
            side_to_glue="inputs",
        )
        return res if not inplace else None

    def expand(
        self,
        inplace: bool = False,
        skip_existing: bool = True,
    ) -> Optional["ComputationFrame"]:
        """
        Expand the computation frame by joining all calls that are not currently
        in the CF.
        """
        res = self if inplace else self.copy()
        res.expand_back(inplace=True, skip_existing=skip_existing)
        res.expand_forward(inplace=True, skip_existing=skip_existing)
        return res if not inplace else None

    def complete_func(self, fname: str, direction: Literal["inputs", "outputs"]):
        """
        Join vertices for the inputs/outputs of a function node if they are
        currently missing.
        """
        if direction == "inputs":
            connected_labels = self.inp[fname].keys()
        elif direction == "outputs":
            connected_labels = self.out[fname].keys()
        new_vnames = {}
        for call_hid in self.fs[fname]:
            call = self.calls[call_hid]
        raise NotImplementedError()

    ############################################################################
    ### traversal
    ############################################################################
    def topsort(self) -> List[str]:
        """
        Return a topological sort of the nodes in the graph
        """
        # Kahn's algorithm
        in_degrees = {node: 0 for node in self.vs.keys() | self.fs.keys()}
        for src, dsts_dict in self.out.items():
            for label, dsts in dsts_dict.items():
                for dst in dsts:
                    in_degrees[dst] += 1
        sources = [node for node, in_degree in in_degrees.items() if in_degree == 0]
        result = []
        while len(sources) > 0:
            source = sources.pop()
            result.append(source)
            for label, dsts in self.out.get(source, {}).items():
                for dst in dsts:
                    in_degrees[dst] -= 1
                    if in_degrees[dst] == 0:
                        sources.append(dst)
        return result

    def sort_nodes(self, nodes: Iterable[str]) -> List[str]:
        """
        Return a topological sort of the given nodes
        """
        return [node for node in self.topsort() if node in nodes]

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe whose columns contain node names, return a new
        dataframe where the columns are sorted topologically.
        """
        assert set(df.columns) <= self.nodes
        sorted_columns = self.sort_nodes(df.columns)
        return df[sorted_columns]

    def get_all_vertex_paths(self, start: str, end: str) -> Set[Tuple[str, ...]]:
        """
        Get all vertex paths from start to end

        !!! NOTE: must modify if we want to support cycles
        """
        if end in self.out_neighbors(node=start):
            return {(start, end)}
        else:
            paths = set()
            for neighbor in self.out_neighbors(node=start):
                for path in self.get_all_vertex_paths(neighbor, end):
                    paths.add((start,) + path)
            return paths

    def eval(
        self,
        *nodes: str,
        values: Literal["refs", "objs"] = "objs",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        A one-stop shop for loading values from a `ComputationFrame` into a
        pandas dataframe.
        """
        if len(nodes) == 0:
            nodes = tuple(self.nodes)
        return self.get_df(*nodes, include_calls=True, values=values, verbose=verbose)

    def get_df(
        self,
        *nodes: str,
        values: Literal["refs", "objs"] = "objs",
        lazy_vars: Optional[Iterable[str]] = None,
        verbose: bool = True,
        include_calls: bool = True,
    ) -> pd.DataFrame:
        """
        A general method for extracting data from the computation frame.

        It works as follows:
        - finds the nodes on paths between any two nodes in `nodes`
        - among these nodes, considers only the elements that are results of
        (partial) computations starting from the inputs (sources) of this
        subgraph
        - produces the joint history of the outputs (sinks) of this subgraph
        """
        if len(nodes) == 0:
            nodes = tuple(self.nodes)
        elif len(nodes) == 1:
            values = list(self.get_var_values(nodes[0]))
            return pd.DataFrame(values, columns=[nodes[0]])
        restricted_cf = self.midstream(*nodes)
        if verbose:
            graph_desc = restricted_cf.get_graph_desc()
            graph_desc = textwrap.indent(graph_desc, "    ")
            print(
                f"Extracting tuples from the computation graph:\n{graph_desc}"
            )
        vnames = {
            x
            for x, sink_elts in restricted_cf.get_sink_elts().items()
            if x in restricted_cf.vnames and len(sink_elts) > 0
        }
        df = restricted_cf.get_joint_history_df(
            vnames=vnames, how="outer", include_calls=include_calls
        )
        # depending on `include_calls`, we may have dropped some columns in `nodes`
        df = df[[x for x in list(nodes) if x in df.columns]]
        if values == "refs":
            res = df
        elif values == "objs":
            res = restricted_cf.eval_df(df, skip_cols=lazy_vars)
        return self._sort_df(res)

    ############################################################################
    ### evaluation
    ############################################################################
    def attach(self):
        self.storage.attach(list(self.refs.values()))

    def eval_df(self, 
                df: pd.DataFrame,
                skip_cols: Optional[Iterable[str]] = None,
                skip_calls: bool = False
                ) -> pd.DataFrame:
        """
        Main tool to evaluate dataframes of `Ref`s and `Call`s by applying
        `unwrap` to chosen columns.
        """
        if len(df) == 0:
            return df 
        # figure out which columns contain what kinds of objects
        def classify_obj(obj: Union[Ref, Call, Any]) -> str:
            if isinstance(obj, Ref):
                return "ref"
            elif isinstance(obj, Call):
                return "call"
            else:
                return "value"
        col_types = {col: classify_obj(df[col].iloc[0]) for col in df.columns}
        if skip_calls:
            df = df[[col for col, t in col_types.items() if t != "call"]]
        if skip_cols is None:
            values = self.storage.unwrap(df.values.tolist())
            return pd.DataFrame(values, columns=df.columns)
        else:
            columns_dict = {col: df[col] if col in skip_cols else self.storage.unwrap(df[col].values.tolist()) for col in df.columns}
            return pd.DataFrame(columns_dict)

    def get(self, hids: Set[str]) -> Set[Ref]:
        return {self.refs[hid] for hid in hids}

    def get_var_values(self, vname: str) -> Set[Ref]:
        return {self.refs[ref_uid] for ref_uid in self.vs[vname]}

    def get_func_table(self, fname: str) -> pd.DataFrame:
        rows = []
        for call_uid in self.fs[fname]:
            call = self.calls[call_uid]
            call_data = {}
            for input_name, input_ref in call.inputs.items():
                call_data[input_name] = (
                    input_ref if input_ref.hid in self.refs else None
                )
            for output_name, output_ref in call.outputs.items():
                call_data[output_name] = (
                    output_ref if output_ref.hid in self.refs else None
                )
            rows.append(call_data)
        return pd.DataFrame(rows)

    @staticmethod
    def _unify_subobjects(
        a: Dict[str, Set[str]], b: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        return {k: a.get(k, set()) | b.get(k, set()) for k in a.keys() | b.keys()}

    @staticmethod
    def _is_subobject(a: Dict[str, Set[str]], b: Dict[str, Set[str]]) -> bool:
        return all([a[k] <= b[k] for k in a.keys() & b.keys()]) and all(
            [k in b for k in a.keys()]
        )

    def get_direct_history(
        self, node: str, hids: Set[str], include_calls: bool
    ) -> Dict[str, Set[str]]:
        """
        Given some hids of refs belonging to a variable,
        - find the creator calls (if any) for these refs contained in
        in-neighbors of the variable
        - for each call, trace the in-edges of its function node to find the
        inputs of the call
        - gather all this data in the form of a {vname: {hid}} dict
        - optionally, include the calls themselves in the result as {fname: {hid}}
        pairs.

        Note that there could be many copies of the creator calls and their
        inputs in the CF, but we only gather the ones reachable along edges that
        are connected to the variable in the appropriate way.
        """
        total_res = {node: hids}
        if node not in self.vnames:
            raise NotImplementedError()
        for hid in hids:
            if hid not in self.creator:
                continue
            res = {}
            creator_hid = self.creator[hid]
            creator_call = self.calls[creator_hid]
            creator_inv_proj = get_reverse_proj(creator_call)
            # creator_input_uids = {r.hid for r in creator_call.inputs.values()}
            # nodes where creator call appears
            creator_fnames = self.callinv[creator_hid] & self.in_neighbors(node=node)
            for creator_fname in creator_fnames:
                if include_calls:
                    if creator_fname not in res:
                        res[creator_fname] = set()
                    res[creator_fname].add(creator_hid)
                creator_node_in_edges = self.in_edges(node=creator_fname)
                for edge_source, _, edge_label in creator_node_in_edges:
                    inv_proj_input_names = creator_inv_proj(edge_label)
                    for inv_input_name in inv_proj_input_names:
                        if (
                            inv_input_name in creator_call.inputs and
                            creator_call.inputs[inv_input_name].hid
                            in self.vs[edge_source]
                        ):
                            if edge_source not in res:
                                res[edge_source] = set()
                            res[edge_source].add(
                                creator_call.inputs[inv_input_name].hid
                            )
            total_res = ComputationFrame._unify_subobjects(total_res, res)
        return total_res

    def get_total_history(
        self,
        node: str,
        hids: Set[str],
        result: Optional[Dict[str, Set[str]]] = None,
        include_calls: bool = False,
    ) -> Dict[str, Set[str]]:
        direct_history = self.get_direct_history(
            node, hids, include_calls=include_calls
        )
        if result is not None and ComputationFrame._is_subobject(
            direct_history, result
        ):
            return result
        else:
            # recursively get the history of the history
            total_res = direct_history
            for node in direct_history.keys():
                if node not in self.vnames:
                    continue
                total_res = ComputationFrame._unify_subobjects(
                    total_res,
                    self.get_total_history(
                        node,
                        direct_history[node],
                        result=total_res,
                        include_calls=include_calls,
                    ),
                )
            return total_res

    def get_history_df(self, vname: str, include_calls: bool = False) -> pd.DataFrame:
        rows = []
        for history_id in self.vs[vname]:
            total_history = self.get_total_history(
                vname, {history_id}, include_calls=include_calls
            )
            total_history_objs = {
                vname: {self.refs[ref_uid] for ref_uid in ref_uids}
                for vname, ref_uids in total_history.items()
                if vname in self.vnames
            }
            total_history_calls = (
                {}
                if not include_calls
                else {
                    fname: {self.calls[call_hid] for call_hid in call_hids}
                    for fname, call_hids in total_history.items()
                    if fname in self.fnames
                }
            )
            # unpack singletons
            total_history_objs = {
                k: list(v)[0] if len(v) == 1 else v
                for k, v in itertools.chain(
                    total_history_objs.items(), total_history_calls.items()
                )
            }
            rows.append(total_history_objs)
        df = pd.DataFrame(rows)
        return self._sort_df(df)

    def get_joint_history_df(
        self,
        vnames: Iterable[str],
        how: Literal["inner", "outer"] = "outer",
        include_calls: bool = True,
    ) -> pd.DataFrame:
        def extract_hids(
            x: Union[None, Ref, Call, Set[Ref], Set[Call]]
        ) -> Union[None, str, Set[str]]:
            if pd.isnull(x):
                return None
            elif isinstance(x, (Ref, Call)):
                return x.hid
            else:
                return {elt.hid for elt in x}

        def eval_hids(
            hids: Union[None, str, Set[str]]
        ) -> Union[None, Ref, Call, Tuple[Union[Ref, Call]]]:
            if pd.isnull(hids):
                return None
            elif isinstance(hids, str):
                return self.refs[hids] if hids in self.refs else self.calls[hids]
            else:
                return tuple(
                    sorted(
                        {
                            self.refs[hid] if hid in self.refs else self.calls[hid]
                            for hid in hids
                        },
                        key=lambda x: x.hid,
                    )
                )

        history_dfs = [
            self.get_history_df(vname, include_calls=include_calls).applymap(
                extract_hids
            )
            for vname in vnames
        ]
        result = history_dfs[0]
        for df in history_dfs[1:]:
            shared_cols = set(result.columns) & set(df.columns)
            result = pd.merge(
                result, df, how=how, on=list(shared_cols), suffixes=("", "")
            )
        # go back to refs
        result = result.applymap(eval_hids)
        return self._sort_df(result)

    @property
    def sets(self) -> Dict[str, Set[str]]:
        return {**self.vs, **self.fs}

    @property
    def values(self) -> Dict[str, Union[Set[Ref], Set[Call]]]:
        var_values = {vname: self.get_var_values(vname) for vname in self.vnames}
        op_values = {
            fname: {self.calls[call_hid] for call_hid in call_hids}
            for fname, call_hids in self.fs.items()
        }
        return {**var_values, **op_values}

    def apply(
        self, f: Callable, to: Literal["refs", "vals"] = "vals"
    ) -> "ComputationFrame":
        """
        Apply a function to the values of the CF
        """
        res = self.copy()
        if to == "refs":
            res.refs = {hid: f(ref) for hid, ref in res.refs.items()}
        elif to == "vals":
            res.refs = {
                hid: f(res.storage.unwrap(ref)) for hid, ref in res.refs.items()
            }
        else:
            raise ValueError(f'Got unexpected value for "to": {to}')
        return res

    ############################################################################
    ### selection
    ############################################################################
    def copy(self) -> "ComputationFrame":
        return ComputationFrame(
            storage=self.storage,
            inp=copy.deepcopy(self.inp),
            out=copy.deepcopy(self.out),
            vs=copy.deepcopy(self.vs),
            fs=copy.deepcopy(self.fs),
            refinv=copy.deepcopy(self.refinv),
            callinv=copy.deepcopy(self.callinv),
            creator=copy.deepcopy(self.creator),
            consumers=copy.deepcopy(self.consumers),
            refs={k: v for k, v in self.refs.items()},
            calls={k: v for k, v in self.calls.items()},
        )

    def drop(
        self,
        nodes: Iterable[str],
        inplace: bool = False,
        with_dependents: bool = False,
    ) -> Optional["ComputationFrame"]:
        if with_dependents:
            raise NotImplementedError()
        res = self if inplace else self.copy()
        for node in nodes:
            if node in res.vnames:
                res.drop_var(node)
            elif node in res.fnames:
                res.drop_func(node)
            else:
                raise ValueError(f"Node {node} not found")
        return res if not inplace else None

    def __getitem__(
        self, indexer: Union[str, Iterable[str], "ComputationFrame"]
    ) -> "ComputationFrame":
        if isinstance(indexer, str):
            return self.select_nodes([indexer])
        elif isinstance(indexer, ComputationFrame):
            raise NotImplementedError()
        elif all([isinstance(k, str) for k in indexer]):
            return self.select_nodes(list(sorted(indexer)))

    def get_reachable_elts(
        self,
        initial_state: Dict[str, Set[str]],
        how: Literal["strong", "weak"],
        direction: Literal["forward", "back"],
        pad: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        Find all reachable elements from the given view along the given
        direction, under either computational (strong) or relational (weak)
        reachability.

        Computational reachability:
            When going forward, reachability means that a call is reachable iff
            *all* its inputs *in nodes of the graph* are reachable. When going
            backward, reachability means that a call is reachable iff *all* its
            outputs *in nodes of the graph* are reachable.

        Relational reachability:
            When going forward, reachability means that a call is reachable iff
            any of its inputs are reachable. When going backward, reachability
            means that a call is reachable iff any of its outputs are reachable.
        """
        result = {k: v.copy() for k, v in initial_state.items()}
        node_order = self.topsort() if direction == "forward" else self.topsort()[::-1]

        def get_input_elts(node, adj_edges):
            adj_elts = []  # will be a list of neighbor elts along the `adj_edges`.
            for edge in adj_edges:
                other_endpoint = edge[0] if direction == "forward" else edge[1]
                hids = result.get(other_endpoint, set())
                adj_elts.append(
                    self.get_adj_elts_edge(edge=edge, hids=hids, direction=direction)
                )
            return adj_elts

        for node in node_order:
            adj_edges = (
                self.in_edges(node) if direction == "forward" else self.out_edges(node)
            )

            if node in self.vnames:
                # the flavor of reachability doesn't matter here, because every
                # ref is created by at most 1 call. We simply take the union
                # over all elements connected as inputs.
                adj_elts = get_input_elts(node, adj_edges)
                to_add = get_nullable_union(*adj_elts)
            else: # node in self.fnames
                if how != "strong":
                    # we have to do the same thing
                    adj_elts = get_input_elts(node, adj_edges)
                    to_add = get_nullable_union(*adj_elts)
                else:
                    # interesting case
                    to_add = set()
                    for call_hid in self.sets[node]:
                        call = self.calls[call_hid]
                        io_refs = call.inputs if direction == "forward" else call.outputs
                        io_name_projections = {name: self.get_io_proj(call_hid=call_hid, name=name) for name in io_refs.keys()}
                        io_label_to_names = invert_dict(io_name_projections)
                        # now, we must check that for each label there's something at the other end
                        is_reachable = True
                        for edge in adj_edges:
                            src, dst, label = edge
                            if label in io_label_to_names.keys():
                                other_end = src if direction == "forward" else dst
                                other_end_hids = result.get(other_end, set())
                                io_ref_hids = {io_refs[name].hid for name in io_label_to_names[label]}
                                if not io_ref_hids <= other_end_hids:
                                    is_reachable = False
                                    break
                        if is_reachable:
                            to_add.add(call_hid)
            result[node] = to_add | result.get(node, set())
        if pad:
            for node in self.nodes - set(result.keys()):
                result[node] = set()
        return result

    def _add_adj_calls(self, vars_view: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        res = vars_view.copy()
        for vname in res.keys():
            new = self.get_adj_elts(node=vname, hids=res[vname], direction="both")
            res = get_setdict_union(res, new)
        return res

    def __lt__(self, other: Union["ComputationFrame", Any]) -> "ComputationFrame":
        if isinstance(other, ComputationFrame):
            raise NotImplementedError()
        else:  # this is a comparison against a "scalar"
            vars_view = {
                vname: {
                    hid
                    for hid in self.vs[vname]
                    if self.storage.unwrap(self.refs[hid]) < other
                }
                for vname in self.vnames
            }
            full_view = self._add_adj_calls(vars_view=vars_view)
            return self.select_subsets(full_view)

    def isin(
        self,
        values: Any,
        by: Literal["hid", "cid", "val"] = "val",
        node_class: Literal["var", "func"] = "var",
    ) -> "ComputationFrame":
        subsets = {}
        if node_class != "var":
            raise NotImplementedError()
        # first, gather the variable hids
        for vname, hids in self.vs.items():
            if by == "val":
                subsets[vname] = {
                    hid for hid in hids if self.storage.unwrap(self.refs[hid]) in values
                }
            elif by == "hid":
                subsets[vname] = {hid for hid in hids if hid in values}
            elif by == "cid":
                subsets[vname] = {hid for hid in hids if self.refs[hid].cid in values}
        # now, add their neighboring calls
        subsets = self._add_adj_calls(subsets)
        return self.select_subsets(subsets)

    ############################################################################
    ### ummmm... more stuff
    ############################################################################
    def merge_into(
        self, node_to_merge: str, merge_into: str, inplace: bool = False
    ) -> Optional["ComputationFrame"]:
        """
        Given a node whose elements are a subset of the elements of another
        node, merge the subset into the superset, and remove the subset node
        (redirecting all its edges to the superset node).
        """
        res = self if inplace else self.copy()
        if not res.sets[node_to_merge] <= res.sets[merge_into]:
            raise ValueError(
                f"Node {node_to_merge} is not a subset of node {merge_into}"
            )
        for src, dst, label in res.in_edges(node_to_merge):
            res.drop_edge(src, dst, label)
            res.add_edge(src, merge_into, label)
        for src, dst, label in res.out_edges(node_to_merge):
            res.drop_edge(src, dst, label)
            res.add_edge(merge_into, dst, label)
        res.drop_node(node_to_merge)
        return res if not inplace else None

    def merge(self, vars: Set[str], new_name: Optional[str] = None) -> str:
        raise NotImplementedError()

    def split(
        self,
        var: str,
    ):
        raise NotImplementedError()

    def drop_unreachable(
        self,
        direction: Literal["forward", "back"],
        how: Literal["strong", "weak"] = "strong",
    ) -> "ComputationFrame":
        """
        Drop all elements that are not *computationally* reachable from the
        elements that belong to the sources of the CF.
        """
        initial_state = (
            self.get_source_elts() if direction == "forward" else self.get_sink_elts()
        )
        to_keep = self.get_reachable_elts(
            initial_state=initial_state, direction=direction, how=how
        )
        return self.select_subsets(to_keep)

    def simplify(self):
        """
        Look for
        - empty vertices to drop;
        - (?) vertices that are subsets of one another to merge;
        """
        for node in list(self.nodes):
            if not self.sets[node]:
                self.drop_node(node)
        # for node in list(self.nodes):
        #     for other_node in list(self.nodes):
        #         if node == other_node: continue
        #         if node in self.nodes and other_node in self.nodes:
        #             if self.sets[node] <= self.sets[other_node]:
        #                 self.merge_into(node, other_node, inplace=True)

    def cleanup(self, inplace: bool = False) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        # remove empty nodes
        for node in list(res.nodes):
            if not res.sets[node]:
                res.drop_node(node)
        return res if not inplace else None

    ############################################################################
    ### constructors
    ############################################################################
    @staticmethod
    def from_op(storage: "Storage", f: Op) -> "ComputationFrame":
        call_hids = storage.call_storage.execute_df(
            f'SELECT call_history_id FROM calls WHERE op="{f.name}"'
        )["call_history_id"].values.tolist()
        calls = storage.mget_call(hids=call_hids, lazy=True)
        # calls = {
        #     call_hid: storage.get_call(call_hid, lazy=True) for call_hid in call_hids
        # }
        input_names = set([k for call in calls for k in call.inputs.keys()])
        output_names = set([k for call in calls for k in call.outputs.keys()])
        res = ComputationFrame(
            refs={},
            calls={},
            vs={},
            fs={},
            refinv={},
            callinv={},
            inp={},
            out={},
            creator={},
            consumers={},
            storage=storage,
        )
        res.add_func(fname=res.get_new_fname(f.name))
        for input_name in input_names:
            input_label = get_name_proj(op=f)(input_name)
            input_var = res.add_var(vname=res.get_new_vname(input_label))
            res.add_edge(input_var, f.name, input_label)
        for output_name in output_names:
            output_label = get_name_proj(op=f)(output_name)
            output_var = res.add_var(vname=res.get_new_vname(output_label))
            res.add_edge(f.name, output_var, output_label)
        for call in calls:
            res.add_call(f.name, call, with_refs=True)
        return res

    @staticmethod
    def from_refs(storage: "Storage", refs: Iterable[Ref]) -> "ComputationFrame":
        res = ComputationFrame(storage=storage)
        vname = res.add_var(vname=res.get_new_vname("v"))
        for ref in refs:
            res.add_ref(vname, ref)
        return res

    ############################################################################
    ### deletion
    ############################################################################
    def delete_calls(self):
        """
        Delete the calls referenced by this CF from the storage
        """
        hids = set(self.calls.keys())
        self.storage.drop_calls(hids=hids, delete_dependents=True)

    def delete_calls_from_df(self, df: pd.DataFrame):
        """
        Delete the calls referenced by the given dataframe from the storage
        """
        hids = set()
        for col in df.columns:
            hids |= {elt.hid for elt in df[col] if isinstance(elt, Call)}
        self.storage.drop_calls(hids=hids, delete_dependents=True)

    ############################################################################
    ### helpers for pretty printing
    ############################################################################
    def get_new_vname(self, name_hint: str) -> str:
        if name_hint not in self.vs:
            return name_hint
        i = 0
        while f"{name_hint}_{i}" in self.vs:
            i += 1
        return f"{name_hint}_{i}"

    def get_new_fname(self, name_hint: str) -> str:
        if name_hint not in self.fs:
            return name_hint
        i = 0
        while f"{name_hint}_{i}" in self.fs:
            i += 1
        return f"{name_hint}_{i}"

    def get_graph_desc(self) -> str:
        lines = []
        ### if there are any isolated variables, add them first
        isolated_vars = {
            vname
            for vname in self.vnames
            if not self.in_edges(vname) and not self.out_edges(vname)
        }
        if isolated_vars:
            lines.append(", ".join(sorted(isolated_vars)))
        for fname in self.sort_nodes(self.fnames):
            input_name_to_vars = self.inp[fname] # input name -> {variables connected}
            output_name_to_vars = self.out[fname] # output name -> {variables connected}
            
            output_names = output_name_to_vars.keys()
            # sort the output names according to their order as returns
            ordered_output_names = sorted(output_names, key=lambda x: int(x.split('_')[1]))
            # add the output names to the output variables in order to avoid
            # confusion when outputs are present only partially in the graph
            output_labels = copy.deepcopy(output_name_to_vars)
            for output_name in output_labels.keys():
                output_labels[output_name] = {f'{varname}' for varname in output_labels[output_name]}
            lhs = ", ".join([" | ".join(output_labels[k]) + f"@{k}" for k in ordered_output_names])
            rhs = ", ".join(
                [
                    f"{k}={' | '.join(input_name_to_vars[k])}"
                    for k in input_name_to_vars.keys()
                    if len(input_name_to_vars[k]) > 0
                ]
            )
            lines.append(f"{lhs} = {fname}({rhs})")
        lines = "\n".join(lines)
        return lines

    def draw(self, show_how: str = "inline"):
        """
        Draw the computational graph for this CF using graphviz, and annotate
        the nodes with some additional information.
        """
        vnodes = {vname: Node(color=SOLARIZED_LIGHT['blue'], label=vname, internal_name=vname, additional_lines=f'({len(self.vs[vname])} refs)',
                              additional_lines_format={ 'color': 'blue', 'point-size': 10}) for vname in self.vnames}
        fnodes = {fname: Node(color=SOLARIZED_LIGHT['red'], label=fname, internal_name=fname,
                                additional_lines=f'({len(self.fs[fname])} calls)',
                                additional_lines_format={ 'color': 'red', 'point-size': 10}
                              ) for fname in self.fnames}
        nodes = {**vnodes, **fnodes}
        edges = []
        for src, dst, label in self.edges():
            edges.append(
                Edge(source_node=nodes[src], 
                     target_node=nodes[dst],
                     source_port=None, 
                     target_port=None,
                     label=label)
            )
        dot_string = to_dot_string(nodes=list(nodes.values()),
                                   edges=edges, 
                                   groups=[],
                                   rankdir="LR")
        write_output(dot_string, output_ext='svg', output_path=None, show_how=show_how,)


    def print_graph(self):
        print(self.get_graph_desc())

    def __repr__(self) -> str:
        graph_desc = self.get_graph_desc()
        graph_desc = textwrap.indent(graph_desc, "    ")
        summary_line = f"{self.__class__.__name__} with {len(self.vs)} variable(s) ({len(self.refs)} unique refs), {len(self.fs)} operation(s) ({len(self.calls)} unique calls)"
        graph_title_line = "Computational graph:"
        # info_line = "Use `.info()` to get more information about the CF, or `.info(*variable_name)` and `.info(*operation_name)` to get information about specific variable(s)/operation(s)"
        # info_lines = textwrap.wrap(info_line, width=80)
        # info_lines = "\n".join(info_lines)
        return f"{summary_line}\n{graph_title_line}\n{graph_desc}"

    def get_var_stats(self) -> pd.DataFrame:
        rows = []
        for vname, ref_uids in self.vs.items():
            rows.append(
                {
                    "name": vname,
                    "num_values": len(ref_uids),
                }
            )
        return pd.DataFrame(rows)

    def get_func_stats(self) -> pd.DataFrame:
        rows = []
        for fname, call_uids in self.fs.items():
            rows.append(
                {
                    "name": fname,
                    "num_calls": len(call_uids),
                }
            )
        return pd.DataFrame(rows)

    def _get_prettytable_str(self, df: pd.DataFrame) -> str:
        import prettytable
        from io import StringIO

        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        pt = prettytable.from_csv(output)
        return pt.get_string()

    def var_info(self, vname: str):
        all_hids = self.vs[vname]
        creator_parts = []
        for (fname, _, output_name) in self.in_edges(node=vname):
            creator_call_hids = self.get_adj_elts_edge(
                edge=(fname, vname, output_name), hids=self.vs[vname], direction="back"
            )
            creator_parts.append(
                f"{vname}={fname}::{output_name} ({len(creator_call_hids)} calls)"
            )
        creator_desc = f"Creator ops: {' | '.join(creator_parts)}"
        consumer_parts = []
        for (_, fname, input_name) in self.out_edges(node=vname):
            consumer_call_hids = self.get_adj_elts_edge(
                edge=(vname, fname, input_name),
                hids=self.vs[vname],
                direction="forward",
            )
            consumer_parts.append(
                f"{fname}({input_name}={vname}, ...) ({len(consumer_call_hids)} calls)"
            )
        consumer_desc = f"Consumer ops: {' | '.join(consumer_parts)}"
        print(f"Variable {vname} with {len(all_hids)} value(s)")
        print(creator_desc)
        print(consumer_desc)

    def func_info(self, fname: str):
        pass

    def info(self, *nodes: str):
        if len(nodes) > 0:  # we want info about specific nodes
            pass
        else:  # we want info about the entire CF
            print(
                f"{self.__class__.__name__} with {len(self.vs)} variable(s) ({len(self.refs)} references to values), {len(self.fs)} operation(s) ({len(self.calls)} references to calls)"
            )
            print("Computational graph:")
            try:
                from rich.syntax import Syntax
                from rich.panel import Panel

                # use a light theme for the syntax highlighting
                p = Panel(
                    Syntax(
                        self.get_graph_desc(),
                        "python",
                        # choose a theme like solarized-light
                        theme="solarized-light",
                    ),
                    title=None,
                    expand=True,
                )
                # show the rich syntax object
                rich.print(p)
            except ImportError:
                print(textwrap.indent(str(self), "    "))
            var_df = self.get_var_stats()
            func_df = self.get_func_stats()
            try:
                print("Variables:")
                if len(var_df) > 0:
                    print(textwrap.indent(self._get_prettytable_str(var_df), "    "))
                print("Operations:")
                if len(func_df) > 0:
                    print(textwrap.indent(self._get_prettytable_str(func_df), "    "))
            except ImportError:
                print("Variables:")
                print(textwrap.indent(str(var_df), "  "))
                print("Functions:")
                print(textwrap.indent(str(func_df), "  "))

    def _ipython_key_completions_(self):
        return self.nodes


from .storage import Storage
