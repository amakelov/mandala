from ..common_imports import *
from ..core.model import FuncOp, Ref, wrap_atom
from ..core.wrapping import unwrap, causify_atom
from ..core.config import Config, MODES
from ..queries.weaver import ValQuery, qwrap, prepare_query
from textwrap import shorten


T = TypeVar("T")


def wrap_ui(obj: T, recurse: bool = True) -> T:
    if isinstance(obj, Ref):
        return obj
    elif type(obj) in (list, tuple):
        if recurse:
            return type(obj)(wrap_ui(v, recurse=recurse) for v in obj)
        else:
            return obj
    elif type(obj) is dict:
        if recurse:
            return {k: wrap_ui(v, recurse=recurse) for k, v in obj.items()}
        else:
            return obj
    else:
        res = wrap_atom(obj)
        causify_atom(ref=res)
        return res


def wrap_inputs(inputs: Dict[str, Any]) -> Dict[str, Ref]:
    # check if we allow implicit wrapping
    if Config.autowrap_inputs:
        return {k: wrap_atom(v) for k, v in inputs.items()}
    else:
        assert all(isinstance(v, Ref) for v in inputs.values())
        return inputs


def bind_inputs(args, kwargs, mode: str, func_op: FuncOp) -> Dict[str, Any]:
    """
    Given args and kwargs passed by the user from python, this adds defaults
    and returns a dict where they are indexed via internal names.
    """
    if mode == MODES.query:
        bound_args = func_op.py_sig.bind_partial(*args, **kwargs)
        inputs_dict = dict(bound_args.arguments)
        input_tps = func_op.input_types
        inputs_dict = {
            k: qwrap(obj=v, tp=input_tps[k], strict=True)
            for k, v in inputs_dict.items()
        }
    else:
        bound_args = func_op.py_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs_dict = dict(bound_args.arguments)
        return inputs_dict
    return inputs_dict


def format_as_outputs(
    outputs: Union[List[Ref], List[ValQuery]]
) -> Union[None, Any, Tuple[Any]]:
    if len(outputs) == 0:
        return None
    elif len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)


def debug_call(
    func_name: str,
    memoized: bool,
    wrapped_inputs: Dict[str, Ref],
    wrapped_outputs: List[Ref],
    io_truncate: Optional[int] = 20,
):
    shortener = lambda s: shorten(
        repr(unwrap(s)), width=io_truncate, break_long_words=True
    )
    inputs_str = ", ".join(f"{k}={shortener(v)}" for k, v in wrapped_inputs.items())
    outputs_str = ", ".join(shortener(v) for v in wrapped_outputs)
    logging.info(
        f'{"(memoized)" if memoized else ""}: {func_name}({inputs_str}) ---> {outputs_str}'
    )
