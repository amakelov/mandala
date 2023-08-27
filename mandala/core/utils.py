import typing
import hashlib
import textwrap
from weakref import WeakKeyDictionary
from .config import *
from ..common_imports import *

if Config.has_cityhash:
    import cityhash

OpKey = Tuple[str, int]


def get_uid() -> str:
    """
    Generate a sequence of 32 hexadecimal characters using the operating
    system's "randomness".
    """
    return "{}".format(binascii.hexlify(os.urandom(16)).decode("utf-8"))


def get_full_uid(uid: str, causal_uid: str) -> str:
    return f"{uid}.{causal_uid}"


def parse_full_uid(full_uid: str) -> Tuple[str, str]:
    uid, causal_uid = full_uid.rsplit(".", 1)
    return uid, causal_uid


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def get_fibers_as_lists(mapping: Dict[_KT, _VT]) -> Dict[_VT, List[_KT]]:
    fibers = defaultdict(list)
    for vq, name in mapping.items():
        fibers[name].append(vq)
    return fibers


def is_subdict(a: Dict, b: Dict) -> bool:
    """
    Check that all keys in `a` are in `b` with the same value.
    """
    return all((k in b and a[k] == b[k]) for k in a)


def invert_dict(d: Dict) -> Dict:
    """
    Invert a dictionary, assuming that all values are unique.
    """
    return {v: k for k, v in d.items()}


def concat_lists(lists: List[list]) -> list:
    return [x for lst in lists for x in lst]


class Hashing:
    """
    Helpers for hashing e.g. function inputs and call metadata.
    """

    @staticmethod
    def get_content_hash_blake2b(obj: Any) -> str:
        # if Config.has_torch and isinstance(obj, torch.Tensor):
        #     #! torch tensors do not have a deterministic hash under the below
        #     # method
        #     obj = obj.cpu().numpy()
        stream = io.BytesIO()
        joblib.dump(value=obj, filename=stream)
        stream.seek(0)
        m = hashlib.blake2b()
        m.update(str((stream.read())).encode())
        return m.hexdigest()

    @staticmethod
    def get_cityhash(obj: Any) -> str:
        stream = io.BytesIO()
        joblib.dump(value=obj, filename=stream)
        stream.seek(0)
        h = cityhash.CityHash128(stream.read())
        digest = h.to_bytes(16, "little")
        s = binascii.b2a_hex(digest)
        res = s.decode()
        return res

    @staticmethod
    def get_joblib_hash(obj: Any) -> str:
        if hasattr(obj, "__get_mandala_dict__"):
            obj = obj.__get_mandala_dict__()
        if Config.has_torch:
            obj = tensor_to_numpy(obj)
        result = joblib.hash(obj)
        if result is None:
            raise RuntimeError("joblib.hash returned None")
        return result

    if Config.content_hasher == "blake2b":
        get_content_hash = get_content_hash_blake2b
    elif Config.content_hasher == "cityhash":
        get_content_hash = get_cityhash
    elif Config.content_hasher == "joblib":
        get_content_hash = get_joblib_hash
    else:
        raise ValueError("Unknown content hasher: {}".format(Config.content_hasher))

    ### deterministic hashing of common collections
    @staticmethod
    def hash_list(elts: List[str]) -> str:
        return Hashing.get_content_hash(elts)

    @staticmethod
    def hash_dict(elts: Dict[str, str]) -> str:
        key_order = sorted(elts.keys())
        return Hashing.get_content_hash([(k, elts[k]) for k in key_order])

    @staticmethod
    def hash_set(elts: Set[str]) -> str:
        return Hashing.get_content_hash(sorted(elts))

    @staticmethod
    def hash_multiset(elts: List[str]) -> str:
        return Hashing.get_content_hash(sorted(elts))


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


if Config.has_torch:

    def tensor_to_numpy(obj: Union[torch.Tensor, dict, list, tuple, Any]) -> Any:
        """
        Recursively convert PyTorch tensors in a data structure to numpy arrays.

        Parameters
        ----------
        obj : any
            The input data structure.

        Returns
        -------
        any
            The data structure with tensors converted to numpy arrays.
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, dict):
            return {k: tensor_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(tensor_to_numpy(v) for v in obj)
        else:
            return obj
