from .storage import Storage, noop
from .model import op, Ignore, NewArgDefault, wrap_atom, ValuePointer
from .tps import MList, MDict
from .deps.tracers.dec_impl import track

from .common_imports import sess


def pprint_dict(d) -> str:
    return '\n'.join([f"    {k}: {v}" for k, v in d.items()])