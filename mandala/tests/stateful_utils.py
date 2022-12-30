from collections import OrderedDict
from mandala.common_imports import *
from mandala.all import *
from mandala.tests.utils import *
from mandala.core.utils import Hashing, invert_dict
from mandala.core.compiler import *
from mandala.core.weaver import ValQuery, FuncQuery
import string


def combine_inputs(*args, **kwargs) -> str:
    return Hashing.get_content_hash(obj=(args, kwargs))


def generate_deterministic(seed: str, n_outputs: int) -> List[str]:
    result = []
    current = seed
    for i in range(n_outputs):
        new = Hashing.get_content_hash(obj=current)
        result.append(new)
        current = new
    return result


def random_string(size: int = 10) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(size))


def make_op(
    ui_name: str,
    input_names: List[str],
    n_outputs: int,
    defaults: Dict[str, Any],
    version: int = 0,
    deterministic: bool = True,
) -> FuncOp:
    """
    Generate a deterministic function with given interface
    """
    sig = Signature(
        ui_name=ui_name,
        input_names=set(input_names),
        n_outputs=n_outputs,
        version=version,
        defaults=defaults,
    )
    if n_outputs == 0:
        f = lambda *args, **kwargs: None
    elif n_outputs == 1:
        f = lambda *args, **kwargs: generate_deterministic(
            seed=combine_inputs(*args, **kwargs), n_outputs=1
        )[0]
    else:
        f = lambda *args, **kwargs: tuple(
            generate_deterministic(
                seed=combine_inputs(*args, **kwargs), n_outputs=n_outputs
            )
        )
    return FuncOp._from_data(f=f, sig=sig)
