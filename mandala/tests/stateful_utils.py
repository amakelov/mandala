from collections import OrderedDict
from mandala.common_imports import *
from mandala.all import *
from mandala.tests.utils import *
from mandala.core.utils import Hashing, invert_dict
from mandala.queries.compiler import *
from mandala.queries.weaver import ValQuery, FuncQuery
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


TEMPLATE = """
def {name}({inputs}) -> {output_annotation}:
    if {n_outputs} == 0:
        return None
    elif {n_outputs} == 1:
        return generate_deterministic(seed=combine_inputs({inputs}),
        n_outputs=1)[0]
    else:
        return tuple(generate_deterministic(seed=combine_inputs({inputs}), n_outputs={n_outputs}))
"""


def make_func(
    ui_name: str,
    input_names: List[str],
    n_outputs: int,
) -> types.FunctionType:
    """
    Generate a deterministic function with given interface
    """
    inputs = ", ".join(input_names)
    output_annotation = (
        "None"
        if n_outputs == 0
        else "Any"
        if n_outputs == 1
        else f"Tuple[{', '.join(['Any'] * n_outputs)}]"
    )
    code = TEMPLATE.format(
        name=ui_name,
        inputs=inputs,
        output_annotation=output_annotation,
        n_outputs=n_outputs,
    )
    f = compile(code, "<string>", "exec")
    exec(f)
    f = locals()[ui_name]
    return f


def make_func_from_sig(sig: Signature) -> types.FunctionType:
    return make_func(sig.ui_name, list(sig.input_names), sig.n_outputs)


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
        input_annotations={k: Any for k in input_names},
        output_annotations=[Any] * n_outputs,
    )
    f = make_func(ui_name, input_names, n_outputs)
    return FuncOp._from_data(func=f, sig=sig)
