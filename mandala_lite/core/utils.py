import hashlib
from .config import *
from ..common_imports import *

if Config.has_cityhash:
    import cityhash


def get_uid() -> str:
    """
    Generate a sequence of 32 hexadecimal characters using the operating
    system's "randomness".
    """
    return "{}".format(binascii.hexlify(os.urandom(16)).decode("utf-8"))


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


class Hashing:
    """
    Helpers for hashing e.g. function inputs and call metadata.
    """

    @staticmethod
    def get_content_hash_blake2b(obj: Any) -> str:
        # if Config.has_torch and isinstance(obj, torch.Tensor):
        #     #! torch tensors do not have a deterministic hash under the below
        #     #method
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

    if Config.content_hasher == "blake2b":
        get_content_hash = get_content_hash_blake2b
    elif Config.content_hasher == "cityhash":
        get_content_hash = get_cityhash
    else:
        raise ValueError("Unknown content hasher: {}".format(Config.content_hasher))
