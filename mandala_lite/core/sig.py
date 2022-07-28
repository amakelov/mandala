from ..common_imports import *
from .utils import get_uid


class DefaultSentinel:
    """
    To distinguish a missing default value from a default value with the value
    `None`.
    """

    pass


class Signature:
    """
    Holds the relevant metadata for a memoized function, which includes
        - the function's name
        - the version,
        - the input names, default values, and number of outputs
        - (optional) superop status

    Responsible for manipulating this state and keeping it consistent.
    """

    def __init__(
        self,
        name: str,
        version: int,
        input_names: Set[str],
        n_outputs: int,
        defaults: Dict[str, Any]
    ):
        self.name = name
        self.version = version
        self.input_names = input_names
        self.n_outputs = n_outputs
        self.defaults = defaults

    ############################################################################
    ### PURE methods for manipulating the signature
    ### to avoid broken state
    ############################################################################
    def update(self, new: "Signature") -> "Signature":
        """
        Return an updated version of this signature based on a new signature.

        This takes care of
            - checking that the new signature is compatible with the old one

        TODO: return a description of the updates for downstream needs
        """
        # it is an internal error if you call this on signatures of different
        # versions
        assert new.version == self.version
        if not set.issubset(set(self.input_names), set(new.input_names)):
            raise ValueError("Removing inputs is not supported")
        if not self.n_outputs == new.n_outputs:
            raise ValueError("Changing the number of outputs is not supported")
        new_defaults = new.defaults
        if any({k not in new_defaults for k in self.defaults}):
            raise ValueError("Dropping defaults is not supported")
        if {k: new_defaults[k] for k in self.defaults} != self.defaults:
            raise ValueError("Changing defaults is not supported")
        res = copy.deepcopy(self)
        for k in new.input_names:
            if k not in res.input_names:
                res.create_input(name=k, default=new_defaults.get(k, DefaultSentinel))
        return res

    def create_input(self, name: str, default: Any = DefaultSentinel) -> "Signature":
        """
        Add an input to this signature, with optional default value
        """
        if name in self.input_names:
            raise ValueError(f'Input "{name}" already exists')
        res = copy.deepcopy(self)
        res.input_names.add(name)
        if default is not DefaultSentinel:
            res.defaults[name] = default
        return res

    def rename(self, new_name: str) -> "Signature":
        """
        Change the name
        """
        res = copy.deepcopy(self)
        res.name = new_name
        return res

    def rename_input(self, name: str, new_name: str) -> "Signature":
        """
        Change the name of an input
        """
        res = copy.deepcopy(self)
        res.input_names.remove(name)
        res.input_names.add(new_name)
        return res

    @staticmethod
    def from_py(
        name: str, version: int, sig: inspect.Signature
    ) -> "Signature":
        """
        Create a `Signature` from a Python function's signature and the other
        necessary metadata.
        """
        input_names = set(
            [
                param.name
                for param in sig.parameters.values()
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
        )
        return_annotation = sig.return_annotation
        if (
            hasattr(return_annotation, "__origin__")
            and return_annotation.__origin__ is tuple
        ):
            n_outputs = len(return_annotation.__args__)
        elif return_annotation is inspect._empty:
            n_outputs = 0
        else:
            n_outputs = 1
        defaults = {
            param.name: param.default
            for param in sig.parameters.values()
            if param.default is not inspect.Parameter.empty
        }
        return Signature(
            name=name,
            input_names=input_names,
            n_outputs=n_outputs,
            defaults=defaults,
            version=version,
        )
