from .common_imports import *
from .common_imports import sess
from .config import Config
import textwrap
import pprint
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
    almost_topological_sort,
    get_edges_in_paths
)
from .model import Call, Ref, Op, __make_list__, RefCollection, CallCollection

from .viz import Node, Edge, SOLARIZED_LIGHT, to_dot_string, write_output

if Config.has_prettytable:
    import prettytable
from io import StringIO


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
    A high-level view of a slice of storage that enables bulk operations on the
    memoized computation graph.

    - use `.info()` to get summary information about this `ComputationFrame`
    - use `.info(*variables)` or `.info(*functions)` to get information 
    about specific variables/functions in this `ComputationFrame`

    # `ComputationFrame` from first principles
    The storage is in general a collection of memoized `Call`s and their
    input/output `Ref`s (even collections like lists are represented as
    collections of `Call`s binding their elements to the collection object
    itself).  

    However, this collection frequently has interesting structure: the same 
    or nearly the same compositions of functions are called multiple times, 
    with different arguments. It is this structure we typically have in mind
    when we think about the computations we've done.


    # How `ComputationFrame` generalizes `pandas.DataFrame`
    - a dataframe has an ordered collection of columns; a `ComputationFrame`
    generalizes this to a **computational graph**, where nodes are variables and
    operations on them. The edges of the graph define a (partial) order on the
    variables and operations.
    - a dataframe has an ordered collection of rows; a `ComputationFrame` has an
    unordered collection of **computations that follow the given computational
    graph**. A computation is allowed to follow the computation graph
    *partially*, meaning that it doesn't have to provide values for all
    variables and/or calls for all operations in the graph.

    # Consistency with `pandas.DataFrame` interfaces
    To make it easier to work with `ComputationFrame`s, we provide methods that
    mimic the `pandas.DataFrame` API to the extent possible:
    - methods that modify the `ComputationFrame` (such as `.drop(), .rename()`)
    come with an `inplace` parameter, such that when `inplace=True`, the method
    modifies the `ComputationFrame` in place and returns `None`, while when
    `inplace=False` (default), the method returns a new `ComputationFrame` with
    the modifications.
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
    
    def ops(self) -> Dict[str, Op]:
        """
        Return a dict of {fname -> Op for the `Call`s in the function node}
        """
        res = {}
        for fname, call_hids in self.fs.items():
            if not call_hids:
                raise NotImplementedError("TODO: handle empty function nodes")
            res[fname] = self.calls[next(iter(call_hids))].op
        return res

    @property
    def nodes(self) -> Set[str]:
        return self.vnames | self.fnames
    
    def refs_by_var(self) -> Dict[str, Set[Ref]]:
        return {vname: {self.refs[hid] for hid in hids} for vname, hids in self.vs.items()}
    
    def calls_by_func(self) -> Dict[str, Set[Call]]:
        return {fname: {self.calls[hid] for hid in hids} for fname, hids in self.fs.items()}

    ############################################################################
    ### modifying the CF
    ############################################################################
    ### bulk operations
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
            res.drop_node(node, inplace=True)
        return res if not inplace else None

    def drop_node(self, node: str, inplace: bool = False) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        if node in res.vs:
            logging.debug(f"Dropping variable {node}")
            res.drop_var(node, inplace=True)
        elif node in res.fs:
            logging.debug(f"Dropping function {node}")
            res.drop_func(node, inplace=True)
        else:
            raise ValueError(f"Node {node} not found in the CF")
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


    ### var operations
    def _add_var(self, vname: Optional[str]) -> str:
        """
        Internal method to add a variable node to the computation frame inplace.
        """
        res = self
        if vname is None:
            vname = res.get_new_vname("v")
        res.vs[vname] = set()
        res.inp[vname] = {}
        res.out[vname] = {}
        return vname

    def drop_var(self, vname: str, inplace: bool = False
                 ) -> Optional["ComputationFrame"]:
        """
        Remove the given variable, all adjacent edges in the graph, and all
        `Ref`s this variable points to, as long as they are not pointed to by
        any other variable. 
        """
        res = self if inplace else self.copy()
        for (src, dst, label) in res.edges():
            if src == vname or dst == vname:
                res._drop_edge(src, dst, label)
        hids = list(res.vs[vname])
        for hid in hids:
            res.drop_ref(vname, hid)
        del res.vs[vname]
        del res.inp[vname]
        del res.out[vname]
        return res if not inplace else None

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
    
    def _add_func(
        self,
        fname: Optional[str],
    ) -> str:
        """
        Internal method to add a function node to the computation frame inplace.
        """
        if fname is None:
            fname = self.get_new_fname("f")
        logging.debug(
            f"Adding function {fname}, called from {inspect.stack()[1].function}"
        )
        self.fs[fname] = set()
        self.inp[fname] = {}
        self.out[fname] = {}
        return fname

    def drop_func(self, fname: str, inplace: bool = False) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        for (src, dst, label) in res.edges():
            if src == fname or dst == fname:
                res._drop_edge(src, dst, label)
        hids = res.fs[fname]
        for hid in hids:
            res.drop_call(fname, hid)
        del res.fs[fname]
        del res.inp[fname]
        del res.out[fname]
        return res if not inplace else None

    def _add_edge(self, src: str, dst: str, label: str, allow_existing: bool = False) -> None:
        """
        Internal method to add an edge to the computation frame inplace.
        """
        if (src, dst, label) in self.edges():
            if not allow_existing:
                raise ValueError(f"Edge ({src}, {dst}, {label}) already exists")
            return
        if label not in self.out[src]:
            self.out[src][label] = set()
        self.out[src][label].add(dst)
        if label not in self.inp[dst]:
            self.inp[dst][label] = set()
        self.inp[dst][label].add(src)

    def _drop_edge(self, src: str, dst: str, label: str):
        """
        Internal method to remove an edge from the computation frame inplace.
        """
        self.out[src][label].remove(dst)
        self.inp[dst][label].remove(src)

    def edges(self) -> List[Tuple[str, str, str]]:
        return [
            (src, dst, label)
            for src, dsts_dict in self.out.items()
            for label, dsts in dsts_dict.items()
            for dst in dsts
        ]

    def add_ref(self, vname: str, ref: Ref, allow_existing: bool = False):
        if ref.hid in self.vs[vname]:
            if not allow_existing:
                raise ValueError(f"Ref {ref.hid} already exists in variable {vname}")
            return
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
            if vname in self.refinv[hid]:
                self.refinv[hid].remove(vname)

    def add_call(self, fname: str, call: Call, with_refs: bool, allow_existing: bool = False):
        if call.hid in self.fs[fname]:
            if not allow_existing:
                raise ValueError(f"Call {call.hid} already exists")
            return
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
                    self.add_ref(vname, input_ref, allow_existing=True)
            for output_name, output_ref in call.outputs.items():
                output_proj = get_name_proj(call.op)(output_name)
                # skip over outputs not tracked by the graph
                if output_proj not in self.out[fname]:
                    continue
                # add to creator
                self.creator[output_ref.hid] = call.hid
                # add ref to corresponding variables
                for vname in self.out[fname][output_proj]:
                    self.add_ref(vname, output_ref, allow_existing=True)

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
        """
        Get the nodes that are connected to the given node as inputs
        """
        return get_nullable_union(*[self.inp[node][k] for k in self.inp[node]])

    def out_neighbors(self, node: str) -> Set[str]:
        """
        Get the nodes that are connected to the given node as outputs
        """
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
    
    def get_reachable_nodes(self, nodes: Set[str], direction: Literal["back", "forward"]) -> Set[str]:
        """
        Get the nodes that can be reached going forward/back from the given
        nodes. 
        """
        visited = set()
        def dfs(node):
            visited.add(node)
            for neighbor in self.in_neighbors(node) if direction == "back" else self.out_neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor)
        for node in nodes:
            dfs(node)
        
        return visited

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
        downstream_nodes = self.get_reachable_nodes(set(nodes), direction="forward")
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
        upstream_nodes = self.get_reachable_nodes(set(nodes), direction="back")
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
        edges = set()
        for start_node in nodes:
            for end_node in nodes:
                if start_node != end_node:
                    edges |= self.get_all_edges_on_paths_between(start_node, end_node)
        midstream_nodes = get_nullable_union(*[set(edge) for edge in edges])
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
        reuse_existing: bool,
        verbose: bool = False,
    ):
        """
        Core internal function that actually expands the computation frame from
        the given call groups. 

        - `reuse_existing`: if True, this will attempt to reuse existing nodes
        in the CF (both operations and variables) instead of creating new ones.
        Specifically, this means that if in the expansion process a function
        node is to be added containing calls that are a subset of a superset 
        of some other node's calls, the function node will be reused. Similarly,
        if a variable node is to be added containing refs that are a subset of
        a superset of some other node's refs, the variable node will be reused.

        Naming conventions below:
        - "labels" refer to edge labels in the graph
        - "names" refer to input/output names of calls, which are projected to
        labels
        """
        for group_id, group_calls in sorted(call_groups.items()): # sort to ensure deterministic behavior
            if verbose:
                # make a readable description of the call group
                op_name, connections = group_id
                connections_str = ", ".join(
                    f"{label}: {vnames}" for label, vnames in connections
                )
                print(f"Expanding call group for op '{op_name}' with {len(group_calls)} calls, connections:\n{textwrap.indent(connections_str, '  ')}")
            op_id, connected_vnames_tuple = group_id
            label_to_connected_vnames = {
                label: vnames for label, vnames in connected_vnames_tuple
            }
            found_func_match = False
            if reuse_existing:
                group_hids = {call.hid for call in group_calls}
                found_func_match = False
                for fname in sorted(self.fs):
                    if group_hids <= self.fs[fname] or (self.fs[fname] <= group_hids and len(self.fs[fname]) != 0):
                        found_func_match = True
                        break
                    elif self.ops()[fname].name == op_id:
                        found_func_match = True
                        break
            if found_func_match:
                if verbose: print(f"Reusing function {fname} for group {group_id}")
            else:
                ### create the function node for this group
                fname = self._add_func(fname=self.get_new_fname(op_id))
                if verbose: print(f"Adding function {fname} for group {group_id}")
            
            ### create edges on the side to glue
            for label, connected_vnames in sorted(label_to_connected_vnames.items()): # sort to ensure deterministic behavior
                if not connected_vnames:
                    continue
                for connected_vname in sorted(connected_vnames): # sort to ensure deterministic behavior
                    src = fname if side_to_glue == "outputs" else connected_vname
                    dst = connected_vname if side_to_glue == "outputs" else fname
                    self._add_edge(src, dst, label, allow_existing=reuse_existing)

            ### figure out what new variables to add to the dual side to the one
            ### we're gluing
            if side_to_glue == "outputs":
                # dual_side_labels = get_nullable_union(*[set(call.inputs.keys()) for call in group_calls])
                dual_side_labels_list = []
                for call in group_calls:
                    name_projector = get_name_proj(call.op)
                    dual_side_labels_list.append(
                        set([name_projector(name) for name in call.inputs])
                    )
                dual_side_labels = sorted(get_nullable_union(*dual_side_labels_list))
            elif side_to_glue == "inputs":
                # dual_side_labels = get_nullable_union(*[set(call.outputs.keys()) for call in group_calls])
                dual_side_labels_list = []
                for call in group_calls:
                    name_projector = get_name_proj(call.op)
                    dual_side_labels_list.append(
                        set([name_projector(name) for name in call.outputs])
                    )
                dual_side_labels = sorted(get_nullable_union(*dual_side_labels_list))

            ### create the variables for the dual side to the one we're gluing
            existing_dual_side_labels = set(self.inp[fname].keys()) if side_to_glue == "outputs" else set(self.out[fname].keys())
            for label in dual_side_labels:
                if label in existing_dual_side_labels:
                    continue
                found_var_match = False
                if reuse_existing:
                    if side_to_glue == "outputs":
                        ref_hids = {call.inputs[name].hid for call in group_calls for name in call.inputs if get_name_proj(call.op)(name) == label and name in call.inputs}
                    else:
                        ref_hids = {call.outputs[name].hid for call in group_calls for name in call.outputs if get_name_proj(call.op)(name) == label and name in call.outputs}
                    found_var_match = False
                    for vname in sorted(self.vs.keys()):
                        if ref_hids <= self.vs[vname] or (self.vs[vname] <= ref_hids and len(self.vs[vname]) != 0):
                            found_var_match = True
                            break
                if found_var_match:
                    if verbose: print(f"Reusing variable {vname} when adding {label} to function {fname}")
                else:
                    vname = self._add_var(vname=self.get_new_vname(label))
                src = vname if side_to_glue == "outputs" else fname
                dst = fname if side_to_glue == "outputs" else vname
                self._add_edge(src, dst, label)
            
            ### finally, add the calls to the CF
            for call in group_calls:
                self.add_call(fname, call, with_refs=True, allow_existing=reuse_existing)

    def get_creators(self, indexer: Union[str, Set[str]]):
        # return information about the calls that created the variables in the
        # indexer without actually adding them to the CF
        raise NotImplementedError()

    def get_consumers(self, indexer: Union[str, Set[str]]):
        # return information about the calls that consume the variables in the
        # indexer without actually adding them to the CF
        raise NotImplementedError()
    
    def _expand_unidirectional(
            self,
            direction: Literal["back", "forward"],
            reuse_existing: bool,
            recursive: bool,
            varnames: Optional[Union[str, Iterable[str]]] = None,
            skip_existing: bool = False,
            inplace: bool = False,
            verbose: bool = False,
    ) -> Optional["ComputationFrame"]:
        res = self if inplace else self.copy()
        if varnames is None:
            while True: # recurse until a fixed point is reached
                current_size = len(res.nodes)
                varnames = res.vs.keys()
                res._expand_unidirectional(
                    direction=direction,
                    varnames=varnames,
                    skip_existing=skip_existing,
                    inplace=True,
                    verbose=verbose,
                    recursive=recursive,
                    reuse_existing=reuse_existing,
                )
                new_size = len(res.nodes)
                if new_size == current_size:
                    break
                if not recursive:
                    break
            return res if not inplace else None
        if isinstance(varnames, str):
            varnames = {varnames}
        
        # TODO: when expanding forward, we don't really want to start from the 
        # sink elements, because there may be non-sink elements that don't have
        # all their consumer calls in the CF!
        expandable_elts = res.get_source_elts() if direction == "back" else res.get_sink_elts()
        expandable_elts = {k: v for k, v in expandable_elts.items() if k in varnames}
        if verbose:
            print(f'Found the following number elements to expand in direction {direction}:\n{textwrap.indent(pprint.pformat({k: len(v) for k, v in expandable_elts.items()}), "  ")}')
        ref_hids = set.union(*expandable_elts.values())
        calls = res.storage.get_creators(ref_hids) if direction == "back" else res.storage.get_consumers(ref_hids)
        if verbose: print(f'Found {len(calls)} calls to expand')
        if skip_existing:
            calls = [call for call in calls if call.hid not in res.calls]
        side_to_glue = "outputs" if direction == "back" else "inputs"
        call_groups = res._group_calls(calls, 
                                       by=side_to_glue,
                                       available_nodes=varnames)
        # find the call groups subsumed by the current graph
        res._expand_from_call_groups(
            call_groups,
            side_to_glue=side_to_glue,
            verbose=verbose,
            reuse_existing=reuse_existing,
        )
        return res if not inplace else None
    
    def expand_back(
        self,
        varnames: Optional[Union[str, Iterable[str]]] = None,
        recursive: bool = False,
        # whether to skip calls that already exist in the CF (even if they're
        # not connected to the given variables)
        skip_existing: bool = False, 
        inplace: bool = False,
        verbose: bool = False,
        reuse_existing: bool = True,
        
    ) -> Optional["ComputationFrame"]:
        """
        Join to the CF the calls that created all refs in the given variables
        that currently do not have a connected creator call in the CF.

        If such refs are found, this will result to the addition of 
        - new function nodes for the calls that created these refs;
        - new variable nodes for the *inputs* of these calls.

        The number of these nodes and how they connect to the CF will depend on
        the structure of the calls that created the refs. 

        Arguments:
        - `varnames`: the names of the variables to expand; if None, expand all
        the `Ref`s that don't have a creator call in any function node of the CF
        that is connected to the `Ref`'s variable node as an output.
        - `recursive`: if True, keep expanding until a fixed point is reached
        """
        return self._expand_unidirectional(
            direction="back",
            recursive=recursive,
            varnames=varnames,
            skip_existing=skip_existing,
            inplace=inplace,
            verbose=verbose,
            reuse_existing=reuse_existing,
        )

    def expand_forward(
        self,
        varnames: Optional[Union[str, Set[str]]] = None,
        recursive: bool = False,
        skip_existing: bool = False,
        inplace: bool = False,
        verbose: bool = False,
        reuse_existing: bool = True,
    ) -> Optional["ComputationFrame"]:
        """
        Join the calls that consume the given variables; see `expand_back` (the 
        dual operation) for more details.
        """
        return self._expand_unidirectional(
            direction="forward",
            recursive=recursive,
            varnames=varnames,
            skip_existing=skip_existing,
            inplace=inplace,
            verbose=verbose,
            reuse_existing=reuse_existing,
        )

    def expand_all(
        self,
        inplace: bool = False,
        skip_existing: bool = False,
        verbose: bool = False,
        reuse_existing: bool = True,
    ) -> Optional["ComputationFrame"]:
        """
        Expand the computation frame by repeatedly applying `expand_back` and
        `expand_forward` until a fixed point is reached.
        """
        res = self if inplace else self.copy()
        cur_size = len(res.refs) + len(res.calls)
        while True:
            res.expand_back(inplace=True, skip_existing=skip_existing, verbose=verbose, reuse_existing=reuse_existing, recursive=True)
            res.expand_forward(inplace=True, skip_existing=skip_existing, verbose=verbose, reuse_existing=reuse_existing, recursive=True)
            new_size = len(res.refs) + len(res.calls)
            if new_size == cur_size:
                break
            else:
                cur_size = new_size
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
    def topsort_modulo_sccs(self) -> List[str]:
        """
        Return an "almost topological" sort of the nodes in the graph:
        - if the graph is acyclic, this is a topological sort
        - if not, this returns an ordering which is 
            - an arbitrary order *within* each strongly connected component,
            - but a topological order *between* the SCCs
        """
        # first, try Kahn's algorithm
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
        if not len(result) == len(self.nodes):
            # use Tarjan's algorithm to find strongly connected components
            # and return an almost-topological sort
            logging.debug("The computation graph contains cycles; falling back to an almost-topological sort (arbitrary order within each strongly connected component)")
            graph = {node: set() for node in self.nodes}
            for src, dst, _ in self.edges():
                graph[src].add(dst)
            result = almost_topological_sort(graph) 
        return result

    def sort_nodes(self, nodes: Iterable[str]) -> List[str]:
        """
        Return a topological sort of the given nodes
        """
        return [node for node in self.topsort_modulo_sccs() if node in nodes]

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe whose columns contain node names, return a new
        dataframe where the columns are sorted topologically.
        """
        assert set(df.columns) <= self.nodes
        sorted_columns = self.sort_nodes(df.columns)
        return df[sorted_columns]

    def get_all_edges_on_paths_between(self, start: str, end: str) -> Set[Tuple[str, str]]:
        """
        Get the *edges* belonging to any paths from start to end
        """
        graph = {src: set() for src in self.nodes}
        for src, dst, _ in self.edges():
            graph[src].add(dst)
        edges_on_paths = get_edges_in_paths(
            graph=graph,
            start=start,
            end=end,
        )
        return edges_on_paths
        # old implementation (when there are no cycles)
        # if end in self.out_neighbors(node=start):
        #     return {(start, end)}
        # else:
        #     paths = set()
        #     for neighbor in self.out_neighbors(node=start):
        #         for path in self.get_all_vertex_paths(neighbor, end):
        #             paths.add((start,) + path)
        #     return paths

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
        return self.df(*nodes, include_calls=True, values=values, verbose=verbose)

    def df(
        self,
        *nodes: str,
        values: Literal["refs", "objs"] = "objs",
        lazy_vars: Optional[Iterable[str]] = None,
        verbose: bool = False,
        include_calls: bool = True,
        join_how: Literal["inner", "outer"] = "outer",
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
        if verbose:
            print(f'Found variables {vnames} containing final elements')
        df = restricted_cf.get_joint_history_df(
            varnames=vnames, how=join_how,
            # excluding calls is best done at the end, because we may miss
            # some useful join conditions if we exclude them here
            include_calls=True, 
            verbose=verbose,
        )
        if not include_calls:
            df = df[[col for col in df.columns if col not in restricted_cf.fnames]]
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
        def classify_obj(obj: Union[Ref, Call, Any, RefCollection, CallCollection]) -> str:
            if isinstance(obj, (Ref, RefCollection)):
                return "ref"
            elif isinstance(obj, (Call, CallCollection)):
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
        self, vname: str, hids: Set[str], include_calls: bool
    ) -> Dict[str, Set[str]]:
        """
        Core internal function for getting the direct history of a set of refs.
        Return a view of the refs (and optionally calls) that immediately
        precede the given refs in the computation graph.

        Given some hids of refs belonging to a variable,
        - find the creator calls (if any) for these refs contained in
        in-neighbors of the variable
        - for each call, trace the in-edges of its function node to find the
        inputs of the call
        - gather all this data in the form of a {vname: {hid}} dict
            - NOTE: the values of the dictionary are SETS of hids, because there
            could be multiple paths leading to the same ref. This is especially
            relevant when there are data structures present in the computation
            graph.
        - optionally, include the calls themselves in the result as {fname: {hid}}
        pairs.

        Note that there could be many copies of the creator calls and their
        inputs in the CF, but we only gather the ones reachable along edges that
        are connected to the variable in the appropriate way.
        """
        total_res = {vname: hids}
        if vname not in self.vnames:
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
            creator_fnames = self.callinv[creator_hid] & self.in_neighbors(node=vname)
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
        """
        Return a view of the refs (and optionally calls) that precede the given
        refs in the computation graph.
        """
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

    def get_history_df(
            self, vname: str, include_calls: bool = True,
            verbose: bool = False
            ) -> pd.DataFrame:
        """
        Returns a dataframe where the rows represent the views of the full
        history of all the refs in the variable `vname`.
        """
        rows = []
        for hid in self.vs[vname]:
            total_history = self.get_total_history(
                vname, {hid}, include_calls=include_calls
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
        if verbose:
            print(f'    For variable {vname}, found dependencies in nodes {df.columns}')
        return self._sort_df(df)

    def get_joint_history_df(
        self,
        varnames: Iterable[str],
        how: Literal["inner", "outer"] = "outer",
        include_calls: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Get the joint computational history of the given variables. This means:
        - for each variable, find the dependencies of all the refs in it
        - this results in a table where each row is a view of the full history
        of the refs in the variable
        - then, join these tables across `varnames` on the common columns

        The join is outer by default, because the histories of the variables may
        be heterogeneous.

        NOTE: the order in which the joins are made matters. For example,
        consider this graph:
        ```python
        with storage:
            for i in range(5):
                model, train_acc = train_model(i)
                if i % 2 == 0:
                    eval_acc = eval_model(model)
        
        cf = storage.cf(train_model).expand_all()
        cf.df(
            ['eval_acc', 'train_acc', 'model'] # suppose this is the order of the joins
            )
        In the first join, the `model` column exists only for the `eval_acc`
        variable, so the join will create some nulls in this column. 

        This is why all outputs of a function node should be joined before
        processing their dependencies. 

        TODO: check if using topological sort is enough to guarantee the
        correct order of joins.
        ```
        """

        sorted_varnames = self.sort_nodes(nodes=varnames)

        def extract_hids(
            x: Union[None, Ref, Call, Set[Ref], Set[Call]]
        ) -> Union[None, str, Set[str]]:
            """
            Convert a Ref or Call object to its hid, or a set of Refs/Calls to
            a canonical hashable representation.
            """
            if pd.isnull(x):
                return None
            elif isinstance(x, (Ref, Call)):
                return x.hid
            else:
                # necessary to make it canonically hashable 
                return tuple(sorted({elt.hid for elt in x}))

        def eval_hids(
            hids: Union[None, str, Set[str]]
        ) -> Union[None, Ref, Call, RefCollection, CallCollection]:
            if pd.isnull(hids):
                return None
            elif isinstance(hids, str):
                return self.refs[hids] if hids in self.refs else self.calls[hids]
            else:
                sorted_objs = tuple(
                    sorted(
                        {
                            self.refs[hid] if hid in self.refs else self.calls[hid]
                            for hid in hids
                        },
                        key=lambda x: x.hid,
                    )
                )
                if next(iter(hids)) in self.refs:
                    return RefCollection(sorted_objs)
                elif next(iter(hids)) in self.calls:
                    return CallCollection(sorted_objs)
                else:
                    raise ValueError(
                        f"Got unexpected value for hids: {next(iter(hids))}"
                    )


        history_dfs = [
            self.get_history_df(vname, include_calls=include_calls, verbose=verbose).applymap(
                extract_hids
            )
            for vname in sorted_varnames
        ]
        result = history_dfs[0]
        for df, varname in zip(history_dfs[1:], sorted_varnames[1:]):
            shared_cols = set(result.columns) & set(df.columns)
            if verbose:
                print(f'   Merging history for the variable {varname} on columns: {shared_cols}')
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
            direction: Literal["forward", "back"],
            how: Literal["strong", "weak"],
            pad: bool = True,):
        """
        Find all reachable elements from the given view along the given
        direction, under either computational (strong) or graph (weak)
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
        
        NOTE: when there are cycles in the graph, this becomes a bit more
        complicated, because we may need to traverse some cycles multiple times
        to get the full reachability information.
        """
        result = {k: v.copy() for k, v in initial_state.items()}

        def get_adj_elts(adj_edges):
            """
            Given edges adjacent to some node, return the elements of this node
            reachable from *any* of the other ends of these edges. 
            """
            adj_elts = set()
            for edge in adj_edges:
                other_endpoint = edge[0] if direction == "forward" else edge[1]
                hids = result.get(other_endpoint, set())
                adj_elts |= (
                    self.get_adj_elts_edge(edge=edge, hids=hids, direction=direction)
                )
            return adj_elts

        def is_strongly_reachable_call(
                fname: str,
                call_hid: str,
                adj_edges: Set[Tuple[str, str, str]],
                direction: Literal["forward", "back"],
                ):
            """
            Core function to determine "strong" reachability of a call from the
            current state.

            A call is strongly reachable in the forward direction if, for each
            input label connected to the call's function node, the corresponding
            input(s) of the call exist(s) in some variable connected by an edge
            with this label.
            """
            call = self.calls[call_hid]
            io_refs = call.inputs if direction == "forward" else call.outputs
            io_name_projections = {name: self.get_io_proj(call_hid=call_hid, name=name) for name in io_refs.keys()}
            # edge label -> inputs/outputs of the call covered by this label
            io_label_to_names = invert_dict(io_name_projections)
            labels_present = set(label for _, _, label in adj_edges)
            io_label_to_names = {k: v for k, v in io_label_to_names.items() if k in labels_present}
            # now, we must check that for each input/output there's something at
            # the other end(s)
            io_names_found = set()
            total_names_to_find = sum([len(v) for v in io_label_to_names.values()])
            if total_names_to_find == 0 and call_hid not in result.get(fname, set()):
                return False # no inputs/outputs to find, and the call is not in the result
            for edge in adj_edges:
                src, dst, label = edge
                if label in io_label_to_names.keys():
                    other_end = src if direction == "forward" else dst
                    other_end_hids = result.get(other_end, set())
                    io_ref_hids = {name: io_refs[name].hid for name in io_label_to_names[label]}
                    io_names_found |= {name for name, hid in io_ref_hids.items() if hid in other_end_hids}
            res = len(io_names_found) == total_names_to_find
            return res

        # we apply each function until a fixed point is reached
        current_size = sum([len(v) for v in result.values()])
        # TODO: to optimize this, we record the nodes which have changed
        # vertices_updated_last_iter = set(initial_state.keys())
        while True:
            for node in self.nodes:
                adj_edges = self.in_edges(node=node) if direction == "forward" else self.out_edges(node=node)
                if node in self.vnames:
                    to_add = get_adj_elts(adj_edges)
                else: # node in self.fnames
                    if how == "weak":
                        to_add = get_adj_elts(adj_edges)
                    else:
                        to_add = set()
                        for call_hid in self.fs[node] - result.get(node, set()):
                            if is_strongly_reachable_call(node, call_hid, adj_edges, direction):
                                to_add.add(call_hid)
                result[node] = to_add | result.get(node, set())
            new_size = sum([len(v) for v in result.values()])
            if new_size == current_size:
                break
            else:
                current_size = new_size
        if pad:
            for node in self.nodes - set(result.keys()):
                result[node] = set()
        return result

    def get_reachable_elts_acyclic(
        self,
        initial_state: Dict[str, Set[str]],
        how: Literal["strong", "weak"],
        direction: Literal["forward", "back"],
        pad: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        """
        result = {k: v.copy() for k, v in initial_state.items()}
        node_order = self.topsort_modulo_sccs() if direction == "forward" else self.topsort_modulo_sccs()[::-1]

        def get_adj_elts(node, adj_edges):
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
                adj_elts = get_adj_elts(node, adj_edges)
                to_add = get_nullable_union(*adj_elts)
            else: # node in self.fnames
                if how != "strong":
                    # we have to do the same thing
                    adj_elts = get_adj_elts(node, adj_edges)
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
    def move_ref(self, source_vname: str, target_vname: str, 
                 hid: str, inplace: bool = True) -> Optional["ComputationFrame"]:
        """
        Move a ref from one variable to another.
        """
        res = self if inplace else self.copy()
        if hid not in res.vs[source_vname]:
            raise ValueError(f"Ref {hid} not in variable {source_vname}")
        res.vs[source_vname].remove(hid)
        res.vs[target_vname].add(hid)
        #! update self.refinv
        self.refinv[hid].remove(source_vname)
        self.refinv[hid].add(target_vname)
        return res if not inplace else None

    def merge_into(
        self, node_to_merge: str, merge_into: str, inplace: bool = False
    ) -> Optional["ComputationFrame"]:
        """
        Given a node whose elements are a subset of the elements of another
        node, merge the subset into the superset, and remove the subset node
        (redirecting all its edges to the superset node).
        """
        res = self if inplace else self.copy()
        ### first, update the elements
        if not res.sets[node_to_merge] <= res.sets[merge_into]:
            for hid in res.sets[node_to_merge] - res.sets[merge_into]:
                self.move_ref(node_to_merge, merge_into, hid, inplace=True)
        # for hid in res.sets[node_to_merge]:
        #     # remove from the refinv
        #     if node_to_merge in res.refinv[hid]:
        #         res.refinv[hid].remove(node_to_merge)
        ### then, update the graph
        for src, dst, label in res.in_edges(node_to_merge):
            res._drop_edge(src, dst, label)
            res._add_edge(src, merge_into, label, allow_existing=True)
        for src, dst, label in res.out_edges(node_to_merge):
            res._drop_edge(src, dst, label)
            res._add_edge(merge_into, dst, label, allow_existing=True)
        res.drop_node(node_to_merge, inplace=True)
        return res if not inplace else None
    
    def merge_vars(self, inplace: bool = False) -> Optional["ComputationFrame"]:
        """
        If variables are found that have an intersection, merge the smaller
        into the larger until no such intersections are found.
        """
        res = self if inplace else self.copy()
        current_size = len(res.vs)
        while True:
            found = False
            for v1 in list(res.vs):
                for v2 in list(res.vs):
                    if v1 == v2:
                        continue
                    if res.vs[v1] <= res.vs[v2] and len(res.vs[v1]) < len(res.vs[v2]):
                        res.merge_into(v1, v2, inplace=True)
                        found = True
                    if found:
                        break
                if found:
                    break
            new_size = len(res.vs)
            if new_size == current_size:
                break
            else:
                current_size = new_size
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
        )["call_history_id"].unique().tolist()
        calls = storage.mget_call(hids=call_hids, in_memory=True)
        # ensure deterministic order of inputs and outputs
        input_names = sorted(set([k for call in calls for k in call.inputs.keys()]))
        output_names = sorted(set([k for call in calls for k in call.outputs.keys()]))
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
        res._add_func(fname=res.get_new_fname(f.name))
        for input_name in input_names:
            input_label = get_name_proj(op=f)(input_name)
            input_var = res._add_var(vname=res.get_new_vname(input_label))
            res._add_edge(input_var, f.name, input_label)
        for output_name in output_names:
            output_label = get_name_proj(op=f)(output_name)
            output_var = res._add_var(vname=res.get_new_vname(output_label))
            res._add_edge(f.name, output_var, output_label)
        for call in calls:
            res.add_call(f.name, call, with_refs=True)
        return res

    @staticmethod
    def from_refs(storage: "Storage", refs: Iterable[Ref]) -> "ComputationFrame":
        res = ComputationFrame(storage=storage)
        vname = res._add_var(vname=res.get_new_vname("v"))
        for ref in refs:
            res.add_ref(vname, ref)
        return res
    
    @staticmethod
    def from_vars(storage: "Storage", vars: Dict[str, Set[Ref]]) -> "ComputationFrame":
        res = ComputationFrame(storage=storage)
        for vname, refs in vars.items():
            vname = res._add_var(vname=vname)
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
        self.storage.drop_calls(calls_or_hids=hids, delete_dependents=True)

    def delete_calls_from_df(self, df: pd.DataFrame):
        """
        Delete the calls referenced by the given dataframe from the storage
        """
        hids = set()
        for col in df.columns:
            hids |= {elt.hid for elt in df[col] if isinstance(elt, Call)}
        self.storage.drop_calls(calls_or_hids=hids, delete_dependents=True)

    ############################################################################
    ### helpers for pretty printing
    ############################################################################
    def get_new_vname(self, name_hint: str) -> str:
        if '_' in name_hint and name_hint.split('_')[-1].isdigit():
            # this is of the form "v_i" for some integer i
            # we don't want to make it "v_i_0" but rather find the smallest 
            # integer i' such that "v_i'" is not in the graph
            v = name_hint.split('_')[0]
            if v == "output":
                # this is a special case
                return self.get_new_vname('var_0')
            i = int(name_hint.split('_')[-1])
            while f"{v}_{i}" in self.vs:
                i += 1
            return f"{v}_{i}"
        elif name_hint not in self.vs:
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
    
    def print_graph(self):
        print(self.get_graph_desc())

    def get_graph_desc(self) -> str:
        """
        Return a string representation of the computation graph of this CF
        """
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
            op = self.ops()[fname]
            input_name_to_vars = self.inp[fname] # input name -> {variables connected}
            output_name_to_vars = self.out[fname] # output name -> {variables connected}
            
            output_names = output_name_to_vars.keys()
            if op.output_names is not None:
                output_to_idx = {name: idx for idx, name in enumerate(op.output_names)}
            else:
                output_to_idx = {name: int(name.split('_')[1]) for name in output_names}
            # sort the output names according to their order as returns
            ordered_output_names = sorted(output_names, key=lambda x: output_to_idx[x])
            # add the output names to the output variables in order to avoid
            # confusion when outputs are present only partially in the graph
            output_labels = copy.deepcopy(output_name_to_vars)
            for output_name in output_labels.keys():
                output_labels[output_name] = {f'{varname}' for varname in output_labels[output_name]}
            lhs = ", ".join([('(' if len(output_labels[k]) > 1 else '') +\
                              " | ".join(output_labels[k]) +\
                              (')' if len(output_labels[k]) > 1 else '') + f"@{k}" 
                             for k in ordered_output_names])
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

    def draw(self,
             show_how: Optional[str] = "inline", 
             path: Optional[str] = None,
             verbose: bool = False,
             print_dot: bool = False,
             orientation: Literal["LR", "TB"] = "TB",
             ):
        """
        Draw the computational graph for this CF using graphviz, and annotate
        the nodes with some additional information.
        """
        if verbose:
            # precompute statistics
            sink_elts = self.get_sink_elts()
            source_elts = self.get_source_elts()
            num_versions = {fname: len(set(self.calls[hid].semantic_version for hid in self.fs[fname])) for fname in self.fnames}
            edge_counts = {(src, dst, label): len(self.get_adj_elts_edge(
                (src, dst, label),
                hids=self.fs[dst] if dst in self.fnames else self.fs[src],
                direction="back" if dst in self.fnames else "forward",))
                for src, dst, label in self.edges()}
        vnodes = {}
        for vname in self.vnames:
            # a little summary of the variable
            additional_lines = [f"{len(self.vs[vname])} values"]
            additional_lines_formats = [{'color': 'blue', 'point-size': 10}]
            if verbose: 
                num_source = len(source_elts[vname])
                num_sink = len(sink_elts[vname])
                parts = []
                if num_source > 0:
                    parts.append(f"{num_source} sources")
                if num_sink > 0:
                    parts.append(f"{num_sink} sinks")
                if num_source > 0 or num_sink > 0:
                    additional_lines[0] += f" ({'/'.join(parts)})"
            vnodes[vname] = Node(
                color=SOLARIZED_LIGHT["blue"],
                label=vname,
                label_size=12,
                internal_name=vname,
                additional_lines=additional_lines,
                additional_lines_formats=additional_lines_formats,
            )
        fnodes = {}
        for fname in self.fnames:
            op_name = self.calls[next(iter(self.fs[fname]))].op.name
            if verbose:
                fname_versions = num_versions[fname]
                if fname_versions > 1:
                    op_name = f"{op_name} ({fname_versions} versions)"
            fnodes[fname] = Node(color=SOLARIZED_LIGHT['red'], 
                              label=fname,
                              label_size=12,
                              internal_name=fname,
                              additional_lines=[f'@op:{op_name}', f'{len(self.fs[fname])} calls'],
                              additional_lines_formats=[{ 'color': 'base03', 'point-size': 10}, { 'color': 'red', 'point-size': 10} ]
                              )
        nodes = {**vnodes, **fnodes}
        edges = []
        for src, dst, label in self.edges():
            display_label = label
            ops = self.ops()
            if dst in ops and ops[dst].name == "__make_list__" and label == "elts":
                display_label = "*elts"
            if verbose:
                display_label = f'{display_label}\\n({edge_counts[(src, dst, label)]} values)'
            edges.append(
                Edge(source_node=nodes[src], 
                     target_node=nodes[dst],
                     source_port=None, 
                     target_port=None,
                     label=display_label)
            )
        dot_string = to_dot_string(nodes=list(nodes.values()),
                                   edges=edges, 
                                   groups=[],
                                   rankdir=orientation)
        if print_dot:
            print(dot_string)
        output_ext = 'svg' if path is None else path.split('.')[-1]
        write_output(dot_string, output_ext='svg', output_path=path, show_how=show_how,)


    def __repr__(self) -> str:
        graph_desc = self.get_graph_desc()
        graph_desc = textwrap.indent(graph_desc, "    ")
        summary_line = f"{self.__class__.__name__} with:\n    {len(self.vs)} variable(s) ({len(self.refs)} unique refs)\n    {len(self.fs)} operation(s) ({len(self.calls)} unique calls)"
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
        if not Config.has_prettytable:
            # fallback
            logger.info(
                "Install the `prettytable` package to get a prettier output for the `info` method."
            )
            return str(df)

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
