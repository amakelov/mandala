from .common_imports import *
import joblib
import io
import inspect
import prettytable
import sqlite3
from .config import *
from abc import ABC, abstractmethod

def dataframe_to_prettytable(df: pd.DataFrame) -> str:
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
