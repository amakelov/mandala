from .sig import Signature
from ..common_imports import *
from .config import Config

if Config.has_torch:
    import torch

    def sig_from_jit_script(
        f: torch.jit.ScriptFunction, version: int
    ) -> Tuple[Signature, inspect.Signature]:
        """
        Parse a torch.jit.ScriptFunction into a FuncOp.

        @torch.jit.script-decorated functions must follow a special format:
            - there can't be a variable number of arguments
            - there can't be keyword arguments with defaults
        """
        # get the names of the inputs
        input_names = [arg.name for arg in f.schema.arguments]
        # get the number of outputs
        n_outputs = len(f.schema.returns)
        # get the default values for the inputs of the ScriptFunction
        defaults = {}
        for argument in f.schema.arguments:
            if argument.default_value is not None:
                # defaults are assigned non-null values by the JIT based on
                # inferred type and default value in the signature.
                defaults[argument.name] = argument.default_value
        parameters = OrderedDict()
        for arg in f.schema.arguments:
            kind = (
                inspect.Parameter.KEYWORD_ONLY
                if arg.kwarg_only
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            if defaults.get(arg.name) is not None:
                param = inspect.Parameter(
                    name=arg.name, kind=kind, default=defaults[arg.name]
                )
            else:
                param = inspect.Parameter(name=arg.name, kind=kind)
            parameters[arg.name] = param
        if n_outputs == 0:
            return_annotation = inspect._empty
        elif n_outputs == 1:
            return_annotation = f.schema.returns[0].type
        else:
            return_annotation = tuple([r.type for r in f.schema.returns])
        sess.d = locals()
        py_sig = inspect.Signature(
            parameters=list(parameters.values()),
            return_annotation=return_annotation,
            __validate_parameters__=True,
        )
        return (
            Signature(
                ui_name=str(f.name),
                input_names=set(input_names),
                n_outputs=n_outputs,
                defaults=defaults,
                version=version,
            ),
            py_sig,
        )
