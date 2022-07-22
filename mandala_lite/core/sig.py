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
    def __init__(self, name:str, external_input_names:Set[str], n_outputs:int, 
                 defaults:Dict[str, Any], version:int, is_super:bool=False):
        self.name = name
        self.external_input_names = external_input_names
        self.defaults = defaults
        self.n_outputs = n_outputs
        self.version = version
        self.internal_name = ''
        self.is_super = is_super
        # external name -> internal name
        self.input_mapping = {}
    
    @property
    def internal_input_names(self) -> Set[str]:
        return set(self.input_mapping.values())
    
    ############################################################################ 
    ### PURE methods for manipulating the signature 
    ### to avoid broken state 
    ############################################################################ 
    def set_internal(self, internal_name:str, input_mapping:dict) -> 'Signature':
        """
        Set the internal names for this signature. This is used when
        synchronizing with storage.
        """
        res = copy.deepcopy(self)
        res.internal_name, res.input_mapping = internal_name, input_mapping
        return res

    def generate_internal(self) -> 'Signature':
        """
        Assign internal names to random UIDs. 
        """
        res = copy.deepcopy(self)
        res.internal_name, res.input_mapping = get_uid(), {k: get_uid() for k in self.external_input_names}
        return res

    def update(self, new:'Signature') -> 'Signature':
        """
        Return an updated version of this signature based on a new signature. 
        
        This takes care of
            - checking that the new signature is compatible with the old one
            - generating names for new inputs.
        """ 
        if not set.issubset(set(self.external_input_names), set(new.external_input_names)):
            raise ValueError('Removing inputs is not supported')
        if not self.n_outputs == new.n_outputs:
            raise ValueError('Changing the number of outputs is not supported')
        new_defaults = new.defaults
        if {k: new_defaults[k] for k in self.defaults} != self.defaults:
            raise ValueError('Changing defaults is not supported')
        if self.is_super != new.is_super:
            raise ValueError('Changing superop status is not supported')
        res = copy.deepcopy(self)
        for k in new.external_input_names:
            if k not in res.external_input_names:
                res.create_input(name=k, default=new_defaults.get(k, DefaultSentinel))
        return res
        
    def create_input(self, name:str, default:Any=DefaultSentinel) -> 'Signature':
        """
        Add an input to this signature, with optional default value
        """
        res = copy.deepcopy(self)
        res.external_input_names.add(name)
        internal_name = get_uid()
        res.input_mapping[name] = internal_name
        if default is not DefaultSentinel:
            res.defaults[name] = default
        return res

    def rename(self, new_name:str) -> 'Signature':
        """
        Change the external name 
        """
        res = copy.deepcopy(self)
        res.name = new_name
        return res

    def rename_input(self, name:str, new_name:str) -> 'Signature':
        """
        Change the external name of an input
        """
        res = copy.deepcopy(self)
        internal_name = self.input_mapping[name]
        res.external_input_names.remove(name)
        res.external_input_names.add(new_name)
        res.input_mapping[new_name] = internal_name
        return res

    @staticmethod
    def from_py(name:str, version:int, sig:inspect.Signature, is_super:bool=False) -> 'Signature':
        """
        Create a `Signature` from a Python function's signature and the other
        necessary metadata.
        """
        input_names = set([param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD])
        return_annotation = sig.return_annotation
        if hasattr(return_annotation, '__origin__') and return_annotation.__origin__ is tuple:
            n_outputs = len(return_annotation.__args__)
        elif return_annotation is inspect._empty:
            n_outputs = 0
        else:
            n_outputs = 1
        defaults = {param.name:param.default for param in sig.parameters.values() if param.default is not inspect.Parameter.empty}
        return Signature(name=name, external_input_names=input_names,
                         n_outputs=n_outputs, defaults=defaults, 
                         version=version, is_super=is_super) 