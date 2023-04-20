from ..common_imports import *
from ..core.model import FuncOp, Ref, wrap_atom, Call
from ..core.wrapping import unwrap, causify_atom
from ..core.config import Config, MODES
from ..queries.weaver import ValQuery, qwrap, prepare_query
from ..deps.model import TerminalData
from ..deps.utils import get_dep_key_from_func
from textwrap import shorten


T = TypeVar("T")


def check_determinism(
    observed_semver: Optional[str],
    stored_semver: Optional[str],
    stored_output_uids: List[str],
    observed_output_uids: List[str],
    func_op: FuncOp,
):
    # check deterministic behavior
    if stored_semver != observed_semver:
        raise ValueError(
            f"Detected non-deterministic dependencies for function "
            f"{func_op.sig.ui_name} after recomputation of transient values."
        )
    if len(stored_output_uids) != len(observed_output_uids):
        raise ValueError(
            f"Detected non-deterministic number of outputs for function "
            f"{func_op.sig.ui_name} after recomputation of transient values."
        )
    if observed_output_uids != stored_output_uids:
        raise ValueError(
            f"Detected non-deterministic outputs for function "
            f"{func_op.sig.ui_name} after recomputation of transient values. "
            f"{observed_output_uids} != {stored_output_uids}"
        )


def get_terminal_data(func_op: FuncOp, call: Call) -> TerminalData:
    return TerminalData(
        op_internal_name=func_op.sig.internal_name,
        op_version=func_op.sig.version,
        call_content_version=call.content_version,
        call_semantic_version=call.semantic_version,
        dep_key=get_dep_key_from_func(func=func_op.func),
    )


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
