from ..common_imports import *
from .utils import get_uid, Hashing


class Signature:
    """
    Holds the relevant metadata for a memoized function, which includes
        - the function's external (human-facing) and internal (used by storage)
        name,
        - the external and internal input names (and the mapping between them),
        - the version,
        - and the default values.
        - (optional) superop status
    Responsible for manipulating this state and keeping it consistent.
    The internal name of the function is an immutable UID that is used to
    identify the function throughout its entire lifetime for the storage it is
    connected to. The external name is what the function is named in the source
    code, and can be changed efficiently (i.e., without having to change all
    stored calls, which refer to the internal name).
    The internal input names are immutable UIDs generated at the time of
    creation of each function input to similarly enable efficient renaming of
    the human-facing (external) function arguments.
    NOTE: in most of the core code, internal names are used. The mapping from
    external to internal names is performed as close as possible to the
    user-facing code.
    """

    def __init__(
        self,
        name: str,
        input_names: Set[str],
        n_outputs: int,
        defaults: Dict[str, Any],
        version: int,
    ):
        self.name = name
        self.input_names = input_names
        self.defaults = defaults
        self.n_outputs = n_outputs
        self.version = version
        self._internal_name = None
        # external name -> internal name
        self._ext_to_int_input_map = None
        # internal input name -> UID of default value
        self._new_input_defaults_uids = {}
    
    @property
    def versioned_name(self) -> str:
        """
        Return the version-qualified name of this signature
        """
        return f"{self.name}_{self.version}"

    @property
    def internal_name(self) -> str:
        if self._internal_name is None:
            raise ValueError("Internal name not set")
        return self._internal_name

    @property
    def ext_to_int_input_map(self) -> Dict[str, str]:
        if self._ext_to_int_input_map is None:
            raise ValueError("Internal name not set")
        return self._ext_to_int_input_map

    @property
    def has_internal_data(self) -> bool:
        return self._internal_name is not None

    @property
    def internal_input_names(self) -> Set[str]:
        return set(self.ext_to_int_input_map.values())
    
    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Signature)
            and self.name == other.name
            and self.input_names == other.input_names
            and self.n_outputs == other.n_outputs
            and self.defaults == other.defaults
            and self.version == other.version
            and self._internal_name == other._internal_name
            and self._ext_to_int_input_map == other._ext_to_int_input_map
            and self._new_input_defaults_uids == other._new_input_defaults_uids
        )

    ############################################################################
    ### PURE methods for manipulating the signature
    ### to avoid broken state
    ############################################################################
    def _generate_internal(self) -> "Signature":
        """
        Assign internal names to random UIDs.
        """
        res = copy.deepcopy(self)
        res._internal_name, res._ext_to_int_input_map = get_uid(), {
            k: get_uid() for k in self.input_names
        }
        return res

    def update(self, new: "Signature") -> Tuple["Signature", dict]:
        """
        Return an updated version of this signature based on a new signature, 
        plus a description of the updates.

        This takes care of
            - checking that the new signature is compatible with the old one
            - generating names for new inputs.
        
        Returns:
            - new `Signature` object
            - a dictionary of {new input name: default value} for any new inputs
              that were created
        """
        # it is an internal error if you call this on signatures of different
        # versions
        assert new.version == self.version
        if not set.issubset(
            set(self.input_names), set(new.input_names)
        ):
            raise ValueError("Removing inputs is not supported")
        if not self.n_outputs == new.n_outputs:
            raise ValueError("Changing the number of outputs is not supported")
        new_defaults = new.defaults
        if any({k not in new_defaults for k in self.defaults}):
            raise ValueError("Dropping defaults is not supported")
        if {k: new_defaults[k] for k in self.defaults} != self.defaults:
            raise ValueError("Changing defaults is not supported")
        new_sig = copy.deepcopy(self)
        updates = {}
        for k in new.input_names:
            if k not in new_sig.input_names:
                # this means a new input is being created
                if k not in new_defaults:
                    raise ValueError(f'All new arguments must be created with defaults!')
                new_sig = new_sig.create_input(name=k, default=new_defaults[k])
                updates[k] = new_defaults[k]
        return new_sig, updates

    def create_input(self, name: str, default) -> "Signature":
        """
        Add an input to this signature, with optional default value
        """
        if name in self.input_names:
            raise ValueError(f'Input "{name}" already exists')
        if not self.has_internal_data:
            raise ValueError("Cannot add inputs to a signature without internal data")
        res = copy.deepcopy(self)
        res.input_names.add(name)
        internal_name = get_uid()
        res.ext_to_int_input_map[name] = internal_name
        res.defaults[name] = default
        default_uid = Hashing.get_content_hash(obj=default)
        res._new_input_defaults_uids[name] = default_uid
        return res

    def rename(self, new_name: str) -> "Signature":
        """
        Change the external name
        """
        res = copy.deepcopy(self)
        res.name = new_name
        return res

    def rename_input(self, name: str, new_name: str) -> "Signature":
        """
        Change the external name of an input
        """
        res = copy.deepcopy(self)
        internal_name = self.ext_to_int_input_map[name]
        res.input_names.remove(name)
        res.input_names.add(new_name)
        del res.ext_to_int_input_map[name]
        res.ext_to_int_input_map[new_name] = internal_name
        return res

    @staticmethod
    def from_py(
        name: str, version: int, sig: inspect.Signature,
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

        

# class Signature:
#     """
#     Holds the relevant metadata for a memoized function, which includes
#         - the function's name
#         - the version,
#         - the input names, default values, and number of outputs
#         - (optional) superop status
# 
#     Responsible for manipulating this state and keeping it consistent.
#     """
# 
#     def __init__(
#         self,
#         name: str,
#         version: int,
#         input_names: Set[str],
#         n_outputs: int,
#         defaults: Dict[str, Any]
#     ):
#         self.name = name
#         self.version = version
#         self.input_names = input_names
#         self.n_outputs = n_outputs
#         self.defaults = defaults
#     
#     @property
#     def qualified_name(self) -> str:
#         """
#         Return the qualified name of this signature
#         """
#         return f"{self.name}_{self.version}"
# 
#     ############################################################################
#     ### PURE methods for manipulating the signature
#     ### to avoid broken state
#     ############################################################################
#     def update(self, new: "Signature") -> "Signature":
#         """
#         Return an updated version of this signature based on a new signature.
# 
#         This takes care of
#             - checking that the new signature is compatible with the old one
# 
#         TODO: return a description of the updates for downstream needs
#         """
#         # it is an internal error if you call this on signatures of different
#         # versions
#         assert new.version == self.version
#         if not set.issubset(set(self.input_names), set(new.input_names)):
#             raise ValueError("Removing inputs is not supported")
#         if not self.n_outputs == new.n_outputs:
#             raise ValueError("Changing the number of outputs is not supported")
#         new_defaults = new.defaults
#         if any({k not in new_defaults for k in self.defaults}):
#             raise ValueError("Dropping defaults is not supported")
#         if {k: new_defaults[k] for k in self.defaults} != self.defaults:
#             raise ValueError("Changing defaults is not supported")
#         res = copy.deepcopy(self)
#         for k in new.input_names:
#             if k not in res.input_names:
#                 res.create_input(name=k, default=new_defaults.get(k, DefaultSentinel))
#         return res
# 
#     def create_input(self, name: str, default: Any = DefaultSentinel) -> "Signature":
#         """
#         Add an input to this signature, with optional default value
#         """
#         if name in self.input_names:
#             raise ValueError(f'Input "{name}" already exists')
#         res = copy.deepcopy(self)
#         res.input_names.add(name)
#         if default is not DefaultSentinel:
#             res.defaults[name] = default
#         return res
# 
#     def rename(self, new_name: str) -> "Signature":
#         """
#         Change the name
#         """
#         res = copy.deepcopy(self)
#         res.name = new_name
#         return res
# 
#     def rename_input(self, name: str, new_name: str) -> "Signature":
#         """
#         Change the name of an input
#         """
#         res = copy.deepcopy(self)
#         res.input_names.remove(name)
#         res.input_names.add(new_name)
#         return res
# 
#     @staticmethod
#     def from_py(
#         name: str, version: int, sig: inspect.Signature
#     ) -> "Signature":
#         """
#         Create a `Signature` from a Python function's signature and the other
#         necessary metadata.
#         """
#         input_names = set(
#             [
#                 param.name
#                 for param in sig.parameters.values()
#                 if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
#             ]
#         )
#         return_annotation = sig.return_annotation
#         if (
#             hasattr(return_annotation, "__origin__")
#             and return_annotation.__origin__ is tuple
#         ):
#             n_outputs = len(return_annotation.__args__)
#         elif return_annotation is inspect._empty:
#             n_outputs = 0
#         else:
#             n_outputs = 1
#         defaults = {
#             param.name: param.default
#             for param in sig.parameters.values()
#             if param.default is not inspect.Parameter.empty
#         }
#         return Signature(
#             name=name,
#             input_names=input_names,
#             n_outputs=n_outputs,
#             defaults=defaults,
#             version=version,
#         )