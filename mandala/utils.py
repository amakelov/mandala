from .common_imports import *
import joblib
import io
import inspect
import sqlite3
from .config import *
from abc import ABC, abstractmethod
from typing import Hashable, TypeVar
if Config.has_prettytable:
    import prettytable

def dataframe_to_prettytable(df: pd.DataFrame) -> str:
    if not Config.has_prettytable:
        # fallback to pandas printing
        logger.info(
            "Install the 'prettytable' package to get prettier tables in the console."
        )
        return df.to_string()
    # Initialize a PrettyTable object
    table = prettytable.PrettyTable()
    
    # Set the column names
    table.field_names = df.columns.tolist()
    
    # Add rows to the table
    for row in df.itertuples(index=False):
        table.add_row(row)
    
    # Return the pretty-printed table as a string
    return table.get_string()


def serialize(obj: Any) -> bytes:
    """
    ! this may lead to different serializations for objects x, y such that x
    ! == y in Python. This is because of things like set ordering, which is not
    ! determined by the contents of the set. For example, {1, 2} and {2, 1} would
    ! `serialize()` to different things, but they would be equal in Python.
    """
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def deserialize(value: bytes) -> Any:
    buffer = io.BytesIO(value)
    return joblib.load(buffer)


def _conservative_equality_check(safe_value: Any, unknown_value: Any) -> bool:
    """
    An equality checker that treats `safe_value` as a "simple" type, but is 
    conservative about how __eq__ can be applied to `unknown_value`. This is
    necessary when comparing against e.g. numpy arrays.
    """
    if type(safe_value) != type(unknown_value):
        return False
    if isinstance(unknown_value, (int, float, str, bytes, bool, type(None))):
        return safe_value == unknown_value
    # handle some common cases
    if isinstance(unknown_value, np.ndarray):
        return np.array_equal(safe_value, unknown_value)
    elif isinstance(unknown_value, pd.DataFrame):
        return safe_value.equals(unknown_value)
    else:
        # fall back to the default equality check
        return safe_value == unknown_value


def get_content_hash(obj: Any) -> str:
    if hasattr(obj, "__get_mandala_dict__"):
        obj = obj.__get_mandala_dict__()
    if Config.has_torch:
        # TODO: ideally, should add a label to distinguish this from a numpy
        # array with the same contents!
        obj = tensor_to_numpy(obj) 
    if isinstance(obj, pd.DataFrame):
        # DataFrames cause collisions for joblib hashing for some reason
        # TODO: the below may be incomplete
        obj = {
            "columns": obj.columns,
            "values": obj.values,
            "index": obj.index,
        }
    result = joblib.hash(obj)  # this hash is canonical wrt python collections
    if result is None:
        raise RuntimeError("joblib.hash returned None")
    return result


def dump_output_name(index: int, output_names: Optional[List[str]] = None) -> str:
    sess.d()
    if output_names is not None and index < len(output_names):
        return output_names[index]
    else:
        return f"output_{index}"


def parse_output_name(name: str) -> int:
    return int(name.split("_")[-1])


def get_setdict_union(
    a: Dict[str, Set[str]], b: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    return {k: a.get(k, set()) | b.get(k, set()) for k in a.keys() | b.keys()}


def get_setdict_intersection(
    a: Dict[str, Set[str]], b: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    return {k: a[k] & b[k] for k in a.keys() & b.keys()}


def get_dict_union_over_keys(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {k: a[k] if k in a else b[k] for k in a.keys() | b.keys()}


def get_dict_intersection_over_keys(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Dict[str, Any]:
    return {k: a[k] for k in a.keys() & b.keys()}


def get_adjacency_union(
    a: Dict[str, Dict[str, Set[str]]], b: Dict[str, Dict[str, Set[str]]]
) -> Dict[str, Dict[str, Set[str]]]:
    return {
        k: get_setdict_union(a.get(k, {}), b.get(k, {})) for k in a.keys() | b.keys()
    }


def get_adjacency_intersection(
    a: Dict[str, Dict[str, Set[str]]], b: Dict[str, Dict[str, Set[str]]]
) -> Dict[str, Dict[str, Set[str]]]:
    return {k: get_setdict_intersection(a[k], b[k]) for k in a.keys() & b.keys()}


def get_nullable_union(*sets: Set[str]) -> Set[str]:
    return set.union(*sets) if len(sets) > 0 else set()


def get_nullable_intersection(*sets: Set[str]) -> Set[str]:
    return set.intersection(*sets) if len(sets) > 0 else set()


def get_adj_from_edges(
    edges: Set[Tuple[str, str, str]], node_support: Optional[Set[str]] = None
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, Set[str]]]]:
    """
    Given edges, convert them into the adjacency representation used by the
    `ComputationFrame` class.
    """
    out = {}
    inp = {}
    for src, dst, label in edges:
        if src not in out:
            out[src] = {}
        if label not in out[src]:
            out[src][label] = set()
        out[src][label].add(dst)
        if dst not in inp:
            inp[dst] = {}
        if label not in inp[dst]:
            inp[dst][label] = set()
        inp[dst][label].add(src)
    if node_support is not None:
        for node in node_support:
            if node not in out:
                out[node] = {}
            if node not in inp:
                inp[node] = {}
    return out, inp


def parse_returns(
    sig: inspect.Signature,
    returns: Any,
    nout: Union[Literal["auto", "var"], int],
    output_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return two dicts based on the returns:
    - {output name: output value}
    - {output name: output type annotation}, where things like `Tuple[T, ...]` are expanded.
    """
    ### figure out the number of outputs, and convert them to a tuple
    if nout == "auto":  # infer from the returns
        if isinstance(returns, tuple):
            nout = len(returns)
            returns_tuple = returns
        else:
            nout = 1
            returns_tuple = (returns,)
    elif nout == "var":
        assert isinstance(returns, tuple)
        nout = len(returns)
        returns_tuple = returns
    else:  # nout is an integer
        assert isinstance(nout, int)
        assert isinstance(returns, tuple)
        assert len(returns) == nout
        returns_tuple = returns
    ### get the dict of outputs
    outputs_dict = {
        dump_output_name(i, output_names): returns_tuple[i] for i in range(nout)
    }
    ### figure out the annotations
    annotations_dict = {}
    output_annotation = sig.return_annotation
    if output_annotation is inspect._empty:  # no annotation
        annotations_dict = {k: Any for k in outputs_dict.keys()}
    else:
        if (
            hasattr(output_annotation, "__origin__")
            and output_annotation.__origin__ is tuple
        ):
            if (
                len(output_annotation.__args__) == 2
                and output_annotation.__args__[1] == Ellipsis
            ):
                annotations_dict = {
                    k: output_annotation.__args__[0] for k in outputs_dict.keys()
                }
            else:
                annotations_dict = {
                    k: output_annotation.__args__[i]
                    for i, k in enumerate(outputs_dict.keys())
                }
        else:
            assert nout == 1
            annotations_dict = {k: output_annotation for k in outputs_dict.keys()}
    return outputs_dict, annotations_dict


def unwrap_decorators(
    obj: Callable, strict: bool = True
) -> Union[types.FunctionType, types.MethodType]:
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    if not isinstance(obj, (types.FunctionType, types.MethodType)):
        msg = f"Expected a function or method, but got {type(obj)}"
        if strict:
            raise RuntimeError(msg)
        else:
            logger.debug(msg)
    return obj

def is_subdict(a: Dict, b: Dict) -> bool:
    """
    Check that all keys in `a` are in `b` with the same value.
    """
    return all((k in b and a[k] == b[k]) for k in a)

_KT, _VT = TypeVar("_KT"), TypeVar("_VT")
def invert_dict(d: Dict[_KT, _VT]) -> Dict[_VT, List[_KT]]:
    """
    Invert a dictionary
    """
    out = {}
    for k, v in d.items():
        if v not in out:
            out[v] = []
        out[v].append(k)
    return out


################################################################################
### CFs
################################################################################
def find_strongly_connected_components(graph: Dict[str, Set[str]]) -> Tuple[Tuple[str,...],...]:
    """
    Find the strongly connected components of a directed graph using Tarjan's
    algorithm. The graph is represented as a dictionary mapping nodes to lists
    of their neighbors.

    Inputs:
    - graph: a dictionary mapping node IDs to sets of their neighbors. Even if
        a node has no neighbors, it should be included in the dictionary with an
        empty set as the value.
    """
    def dfs(node):
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in graph[node]:
            if neighbor not in indexes:
                dfs(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack[neighbor]:
                lowlinks[node] = min(lowlinks[node], indexes[neighbor])

        if lowlinks[node] == indexes[node]:
            # this is the "root" of a strongly connected component
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    # each vertex will be assigned an index, in the order they are visited
    # initial index is 0 (will be assigned to the first vertex visited)
    index = 0 
    # vertex ID -> index, immutable once assigned
    indexes = {} 
    # vertex ID -> index, maintains the lowest index of a vertex on the stack
    # reachable from the given vertex (so can be updated during the DFS)
    lowlinks = {} 
    # vertex ID -> bool, indicator of vertices currently on the stack
    on_stack = {}
    # stack of vertices
    stack = []
    # list of strongly connected components discovered
    sccs = []

    for node in graph:
        if node not in indexes:
            dfs(node)

    # ensure that the output is deterministic
    return tuple(sorted(tuple(sorted(scc)) for scc in sccs))

def create_super_graph(graph: Dict[str, Set[str]], 
                       sccs: Tuple[Tuple[str,...],...]
                       ) -> Dict[str, Set[int]]:
    """
    Given the original graph and the strongly connected components, create a
    supergraph where each node is an SCC and there is an edge from SCC A to SCC
    B if there is an edge from a node in A to a node in B in the original graph.
    """
    # map from node ID to SCC ID (an integer)
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i

    super_graph = {}
    for node in graph:
        scc_id = node_to_scc[node]
        if scc_id not in super_graph:
            super_graph[scc_id] = set()
        for neighbor in graph[node]:
            neighbor_scc_id = node_to_scc[neighbor]
            if scc_id != neighbor_scc_id:
                super_graph[scc_id].add(neighbor_scc_id)

    return super_graph

T = TypeVar("T", bound=Hashable)
def topological_sort(graph: Dict[T, Set[T]]) -> List[T]:
    """
    Topological sort of a directed acyclic graph using depth-first search.
    """
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)

    visited = set()
    result = []

    for node in sorted(graph.keys()):
        if node not in visited:
            dfs(node)

    return result[::-1]

def almost_topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    """
    An almost-topological sort of a directed graph:
    - between SCCs, the order is topological
    - within an SCC, the order is arbitrary
    """
    sccs = find_strongly_connected_components(graph)
    super_graph = create_super_graph(graph, sccs)
    sorted_super_nodes = topological_sort(super_graph)
    
    result = []
    for super_node in sorted_super_nodes:
        result.extend(sccs[super_node])
    
    return result

def get_edges_in_paths(
        graph: Dict[str, Set[str]],
        start: str,
        end: str) -> Set[Tuple[str, str]]:
    """
    Find all edges belonging to some *simple* path from A to B in a directed
    graph that may contain cycles. 
    """
    def dfs(node, path):
        if node == end:
            # We've found a path to B, mark all edges on this path
            for edge in path:
                valid_edges.add(edge)
            return

        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, path + [(node, neighbor)])
            elif (node, neighbor) not in valid_edges and (node, neighbor) not in path:
                # If we've visited this neighbor before but this edge isn't marked yet,
                # it might be part of a cycle that leads to B
                dfs(neighbor, path + [(node, neighbor)])
        if node in visited:
            # we may have removed `node` already if it was part of a cycle
            visited.remove(node)

    visited = set()
    valid_edges = set()
    dfs(start, [])
    return valid_edges


################################################################################
### user interaction
################################################################################
def ask_user(question: str, valid_options: List[str]) -> str:
    """
    Ask the user a question and return their response.
    """
    prompt = f"{question} "
    while True:
        print(prompt)
        response = input().strip().lower()
        if response in valid_options:
            return response
        else:
            print(f"Invalid response: {response}")


def mock_input(prompts):
    ### simulate user input non-interactively
    it = iter(prompts)
    def mock_input_func(*args):
        return next(it)
    return mock_input_func
