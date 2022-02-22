from abc import ABC, abstractmethod
from inspect import Parameter

from .config import CoreConfig
from .tps import Type, is_subtype, AnyType

from ..common_imports import *
from ..util.common_ut import rename_dict_keys, invert_dict
from ..session import sess

def is_typing_tuple(obj:TAny) -> bool:
    """
    Check if an object is a typing.Tuple[...] thing
    """
    try:
        return obj.__origin__ is tuple
    except AttributeError:
        return False
    
################################################################################
### signature
################################################################################
class BaseSignature(ABC):
    """
    Base class for custom, restricted signature objects

    Responsible for:
        - keeping track of signature information
        - binding types to given arguments
        - binding default values to given arguments
        - todo: binding concrete types to type variables 
    """
    ### positional-or-keyword
    @property
    @abstractmethod
    def poskw_names(self) -> TList[str]:
        """
        Ordered names of the positional-or-keyword arguments 
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def poskw(self) -> TDict[str, Type]:
        """
        Mapping of positional-or-keyword args to type
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def ord_poskw(self) -> TList[TTuple[str, Type]]:
        """
        Pairs of (positional-or-keyword arg, type) ordered according to
        `self.poskw_names`
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def vararg(self) -> TOption[TTuple[str, Type]]:
        """
        Name and type of the argument representing varying positional arguments,
        if any.
        """
        raise NotImplementedError()
    
    ### keyword-only
    @property
    @abstractmethod
    def kw(self) -> TDict[str, Type]:
        """
        Mapping of keyword-only arguments to their type
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def varkwarg(self) -> TOption[TTuple[str, Type]]:
        """
        Name and type of the argument representing varying keyword-only
        arguments, if any.
        """
        raise NotImplementedError()
    
    ### defaults
    @property
    @abstractmethod
    def defaults(self) -> TDict[str, TAny]:
        raise NotImplementedError()
    
    ### outputs
    @property
    @abstractmethod
    def output_names(self) -> TOption[TList[str]]:
        """
        Ordered list of fixed output names for this signature, if defined.
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def outputs(self) -> TOption[TDict[str, Type]]:
        """
        Dictionary of output name to its type
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def ord_outputs(self) -> TOption[TList[TTuple[str, Type]]]:
        raise NotImplementedError()

    ### 
    def inputs(self) -> TDict[str, Type]:
        return {**self.poskw, **self.kw}
    
    def as_table(self) -> pd.DataFrame:
        rows = []
        default_null = inspect._empty
        for i, (name, tp) in enumerate(self.ord_poskw):
            rows.append({'name': name, 'kind': 'poskw', 'type': tp, 'idx': i,
                         'default': self.defaults.get(name, default_null)})
        if self.vararg is not None:
            vararg_name, vararg_tp = self.vararg
            rows.append({'name': vararg_name, 'kind': 'vararg', 
                         'type': vararg_tp, 'idx': -1, 
                         'default': self.defaults.get(vararg_name, default_null)})
        for name, tp in self.kw.items():
            rows.append({'name': name, 'kind': 'kw', 'type': tp, 'idx': -1,
                         'default': self.defaults.get(name, default_null)})
        if self.varkwarg is not None:
            varkwarg_name, varkwarg_tp = self.varkwarg
            rows.append({'name': varkwarg_name, 'kind': 'varkwarg', 
                         'type': varkwarg_tp, 'idx': -1,
                         'default': self.defaults.get(varkwarg_name, default_null)})
        if self.has_fixed_outputs:
            for i, (name, tp) in enumerate(self.ord_outputs):
                rows.append({'name': name, 'kind': 'output', 'type': tp, 
                             'idx': i, 
                             'default': self.defaults.get(name, default_null)})
        return pd.DataFrame(rows, columns=['name', 'kind', 'type', 'default', 'idx'])
    
    ############################################################################ 
    @staticmethod
    @abstractmethod
    def from_callable(clbl:TCallable) -> 'BaseSignature':
        raise NotImplementedError()
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def check_instance(self, input_types:TDict[str, Type], output_types:TDict[str, Type]):
        raise NotImplementedError()
    
    @abstractmethod
    def copy(self) -> 'BaseSignature':
        raise NotImplementedError()

    @abstractmethod
    def rename_inputs(self, rename_dict:TDict[str, str]):
        raise NotImplementedError()

    ############################################################################ 
    ### default implementations
    ############################################################################ 
    @property
    def has_fixed_inputs(self) -> bool:
        return self.vararg is None and self.varkwarg is None
    
    @property
    def has_fixed_outputs(self) -> bool:
        return self.output_names is not None
    
    @property
    def is_fixed(self) -> bool:
        return self.has_fixed_inputs and self.has_fixed_outputs
    
    @property
    def fixed_input_names(self) -> TSet[str]:
        return {*self.poskw_names, *self.kw}
    
    def _bind(self, args:tuple, kwargs:dict, apply_defaults:bool=False, 
              bind_what:str='types') -> TDict[str, TAny]:
        """
        Convert args and kwargs into a single dict of either types or values.
        
        The two modes:
            - when `bind_what='types'`, it is expected that args, kwargs hold
            respective types for each argument
            - when `bind_what='values'`, `args` and `kwargs` hold arbitrary
            objects that are arranged into a dict
        """
        assert bind_what in ('types', 'values')
        bind_types = (bind_what == 'types')
        res = {}
        num_args = len(self.poskw_names)
        for i in range(len(args)):
            if i < num_args:
                res[self.poskw_names[i]] = (self.ord_poskw[i] if 
                                            bind_types else args[i])
            else:
                assert self.vararg is not None
                vararg_name, vararg_type = self.vararg
                res[f'{vararg_name}_{i-num_args}'] = (vararg_type if 
                                                      bind_types else args[i])
        for k, v in kwargs.items():
            assert k not in res
            if k in self.kw:
                res[k] = self.kw[k] if bind_types else kwargs[k]
            elif k in self.poskw_names:
                res[k] = self.poskw[k] if bind_types else kwargs[k]
            else:
                if self.varkwarg is None:
                    raise ValueError(f'Got kw name {k} not in signature!')
                res[k] = self.varkwarg[1] if bind_types else kwargs[k]
        if (bind_what == 'values' and apply_defaults):
            for k, v in self.defaults.items():
                if (k not in res) and (v is not inspect._empty):
                    res[k] = v
        return res
        
    def bind_types(self, args:tuple, kwargs:dict) -> TDict[str, Type]:
        """
        Return a dictionary mapping input names to their types.
        """
        return self._bind(args=args, kwargs=kwargs, bind_what='types')

    def bind_args(self, args:tuple, kwargs:dict, apply_defaults:bool) -> TDict[str, TAny]:
        """
        Parse a dictionary mapping input names to their given values.
        """
        return self._bind(args=args, kwargs=kwargs, bind_what='values', 
                          apply_defaults=apply_defaults)


class Signature(BaseSignature):

    def __init__(self, ord_poskw:TList[TTuple[str, Type]]=None,
                 kw:TDict[str, Type]=None, 
                 ord_outputs:TList[TTuple[str, Type]]=None,
                 vararg:TTuple[str, Type]=None,
                 varkwarg:TTuple[str, Type]=None,
                 defaults:TDict[str, TAny]=None, 
                 fixed_outputs:bool=True):
        self._ord_poskw = [] if ord_poskw is None else ord_poskw
        self._kw = {} if kw is None else kw
        self._vararg = vararg
        self._varkwarg = varkwarg
        if not fixed_outputs:
            assert ord_outputs is None
            self._ord_outputs = None
        else:
            self._ord_outputs = [] if ord_outputs is None else ord_outputs
        self._defaults = {} if defaults is None else defaults
    
    @property
    def poskw_names(self) -> TList[str]:
        return [name for name, _ in self.ord_poskw]
    
    @property
    def poskw(self) -> TDict[str, Type]:
        return {name: tp for name, tp in self.ord_poskw}

    @property
    def ord_poskw(self) -> TList[TTuple[str, Type]]:
        return self._ord_poskw
    
    @property
    def kw(self) -> TDict[str, Type]:
        return self._kw
    
    @property
    def vararg(self) -> TOption[TTuple[str, Type]]:
        return self._vararg

    @property
    def varkwarg(self) -> TOption[TTuple[str, Type]]:
        return self._varkwarg

    @property
    def defaults(self) -> TDict[str, TAny]:
        return self._defaults
    
    ### outputs
    @property
    def output_names(self) -> TOption[TList[str]]:
        if self.ord_outputs is None:
            return None
        return [name for name, _ in self.ord_outputs]
    
    @property
    def outputs(self) -> TOption[TDict[str, Type]]:
        if self.ord_outputs is None:
            return None
        return {name: tp for name, tp in self.ord_outputs}
    
    @property
    def ord_outputs(self) -> TOption[TList[TTuple[str, Type]]]:
        return self._ord_outputs

    ### 
    @staticmethod
    def from_callable(clbl:TCallable, output_names:TList[str]=None, 
                      strict:bool=None, excluded_names:TList[str]=None, 
                      fixed_outputs:bool=True) -> 'Signature':
        """
        Get a Signature object from a callable. 
            - strict: controls whether annotations are allowed to be generic 
            python annotations (False), or must be only custom 
            Type-related objects (True)
        """
        if strict is None:
            strict = CoreConfig.strict_signatures
        sig = inspect.signature(clbl)
        params = sig.parameters
        arg_annotations:TList[TTuple[str, TAny]] = []
        vararg_annotation:TOption[TTuple[str, TAny]] = None
        kwarg_annotations:TDict[str, TAny] = {}
        varkwarg_annotation:TOption[TTuple[str, TAny]]  = None
        defaults = {}
        excluded_names = [] if excluded_names is None else excluded_names
        included_params = {k: v for k, v in params.items() 
                           if k not in excluded_names}
        for name, p in included_params.items():
            if p.kind == Parameter.POSITIONAL_ONLY:
                raise NotImplementedError()
            elif p.kind == Parameter.POSITIONAL_OR_KEYWORD:
                arg_annotations.append((name, p.annotation))
                defaults[name] = p.default
            elif p.kind == Parameter.VAR_POSITIONAL:
                vararg_annotation = (name, p.annotation)
            elif p.kind == Parameter.KEYWORD_ONLY:
                kwarg_annotations[name] = p.annotation
                defaults[name] = p.default
            elif p.kind == Parameter.VAR_KEYWORD:
                varkwarg_annotation = (name, p.annotation)
        if not CoreConfig.enable_defaults and (len(defaults) > 0):
            raise ValueError()
        if fixed_outputs:
            return_annotation = sig.return_annotation
            if is_typing_tuple(return_annotation):
                # we must unpack the typing tuple in this weird way
                pre_output_annotations:TList[TAny] = list(return_annotation.__args__)
            elif return_annotation is inspect._empty:
                pre_output_annotations = []
            else:
                pre_output_annotations = [return_annotation]
            if output_names is None:
                output_names = [f'output_{i}' for 
                                i in range(len(pre_output_annotations))]
            output_annotations = [(name, annotation) for name, annotation
                                in zip(output_names, pre_output_annotations)]
        else:
            output_annotations = None
        ### convert to types 
        if strict:
            type_getter = Type.from_tp_or_wrapper
        else:
            type_getter = Type.from_annotation
        args = [(name, type_getter(annotation)) 
                for name, annotation in arg_annotations]
        if vararg_annotation is not None:
            vararg = (vararg_annotation[0], type_getter(vararg_annotation[1]))
        else:
            vararg = None
        kwargs = {name: type_getter(annotation) 
                  for name, annotation in kwarg_annotations.items()}
        if varkwarg_annotation is not None:
            varkwarg = (varkwarg_annotation[0], type_getter(varkwarg_annotation[1]))
        else:
            varkwarg = None
        ### deal with fixed/varying outputs
        if fixed_outputs:
            ord_outputs = [(name, type_getter(annotation)) 
                           for name, annotation in output_annotations]
        else:
            ord_outputs = None
        return Signature(ord_poskw=args, kw=kwargs, ord_outputs=ord_outputs, 
                         vararg=vararg, varkwarg=varkwarg, defaults=defaults, 
                         fixed_outputs=fixed_outputs)
        
    def copy(self) -> 'Signature':
        return copy.deepcopy(self)
    
    def check_instance(self, input_types:TDict[str, Type], 
                       output_types:TDict[str, Type]):
        # inputs must be subtypes
        sig_input_types = self.bind_types(args=(), kwargs=input_types)
        for k in input_types:
            assert is_subtype(s=input_types[k], t=sig_input_types[k])
        # outputs must be subtypes too
        if self.has_fixed_outputs:
            sig_output_types = self.outputs
            for k in output_types:
                assert is_subtype(s=output_types[k], t=sig_output_types[k])
        else:
            logging.warning('Cannot verify output types of signature with varying outputs')

    ############################################################################ 
    ### refactoring
    ############################################################################ 
    def rename_inputs(self, rename_dict:TDict[str, str]) -> BaseSignature:
        """
        Rename inputs according to the given dictionary, interpreted as 
            {current_name: new_name}. 
        Not all inputs must be present in the dictionary.
        """
        # verify what we are renaming is a subset
        assert set(rename_dict.keys()).issubset(self.fixed_input_names)
        new_names = [rename_dict.get(k, k) for k in self.fixed_input_names]
        # verify the new names are still distinct
        assert len(new_names) == len(set(new_names))
        ord_poskw = [(rename_dict.get(k, k), tp) for k, tp in self._ord_poskw]
        kw = {rename_dict.get(k, k): tp for k, tp in self._kw.items()}
        # rename defaults
        defaults = {rename_dict.get(k, k) : v for k, v in self.defaults.items()}
        return Signature(ord_poskw=ord_poskw, kw=kw, ord_outputs=self.ord_outputs,
                         vararg=self.vararg, varkwarg=self.varkwarg, 
                         defaults=defaults, fixed_outputs=(self.ord_outputs is not None))
                         
    ############################################################################ 
    ### magics
    ############################################################################ 
    def __eq__(self, other) -> bool:
        if not isinstance(other, Signature):
            return False
        return self.__dict__ == other.__dict__
    
    def _stringify_interface(self, names:TList[str], types:TDict[str, Type],
                             defaults:TDict[str, TAny]) -> str:
        parts = []
        for name in names:
            if defaults[name] is inspect._empty:
                parts.append(f'{name}: {types[name]}')
            else:
                parts.append(f'{name}: {types[name]} = {defaults[name]}')
        return f"({', '.join(parts)})"
    
    def __repr__(self) -> str:
        assert self.has_fixed_inputs
        assert self.has_fixed_outputs
        input_names = self.poskw_names + list(self.kw.keys())
        inputs_part = self._stringify_interface(names=input_names, 
                                                types=self.inputs(), 
                                                defaults=self.defaults)
        outputs_part = self._stringify_interface(
            names=self.output_names,
            types=self.outputs,
            defaults={k: inspect._empty for k in self.output_names}
        )
        return f'{inputs_part} |---> {outputs_part}'

################################################################################
### maps between signatures
################################################################################
class BaseSigMap(ABC):
    """
    Encode the data of a mapping between signatures
    """
    @property
    @abstractmethod
    def source(self) -> BaseSignature:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def target(self) -> BaseSignature:
        raise NotImplementedError()
    
    @abstractmethod
    def inverse(self) -> 'BaseSigMap':
        raise NotImplementedError()
    
    @abstractmethod
    def bind_dict_forward(self, args:tuple, kwargs:dict) -> TDict[str, TAny]:
        """
        Return dictionary of {target name: argument value} pairs
        """
        raise NotImplementedError()
    
    @abstractmethod
    def map_input_names(self, input_names:TIter[str]) -> TDict[str, str]:
        """
        Map names of inputs from the source signature to their respective names
        in the target signature. 
        
        The reason this is not a single static map is because there can be
        varargs/varkwargs.
        """
        raise NotImplementedError()

    @abstractmethod
    def fixed_inputs_map(self) -> TDict[str, str]:
        """
        Return a mapping of the fixed inputs of the source signature to those of
        the target.
        """
        raise NotImplementedError()

    @abstractmethod
    def fixed_outputs_map(self) -> TDict[str, str]:
        """
        Return a mapping of the fixed outputs of the source signature to those
        of the target.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def rename_source_inputs(self, rename_dict:TDict[str, str]) -> 'BaseSigMap':
        raise NotImplementedError()
    
    def fixed_io_map(self) -> TDict[str, str]:
        """
        Return combined mapping of input/output UI names to internal
        """
        return {**self.fixed_inputs_map(), **self.fixed_outputs_map()}
    

class SigMap(BaseSigMap):
    
    def __init__(self, source:BaseSignature, target:BaseSignature, 
                 kwarg_map:TDict[str, str]):
        """
        An object that keeps track of how names of inputs and outputs are mapped
        between two signatures.
        
        NOTE: only the names of `.poskw`, `.kw` and `.outputs` are mapped, since
        types and defaults are not changed from source to target.
        """
        self._source = source
        self._target = target
        self._kwarg_map = kwarg_map
        self._arg_map = {sname: tname for sname, tname in 
                         zip(self.source.poskw_names, self.target.poskw_names)}
    
    @property
    def source(self) -> BaseSignature:
        return self._source
    
    @property
    def target(self) -> BaseSignature:
        return self._target
    
    @property
    def kwarg_map(self) -> TDict[str, str]:
        return self._kwarg_map
    
    def inverse(self) -> 'SigMap':
        return SigMap(source=self.target, target=self.source, 
                      kwarg_map=invert_dict(self.kwarg_map))
    
    @property
    def arg_map(self) -> TDict[str, str]:
        return self._arg_map
    
    def map_input_names(self, input_names:TIter[str]) -> TDict[str, str]:
        res = {}
        argnames = set(self.source.poskw_names)
        kwargnames = set(self.source.kw)
        for name in input_names:
            if name in argnames:
                res[name] = self.arg_map[name]
            elif name in kwargnames:
                res[name] = self.kwarg_map[name]
            else:
                assert self.source.vararg is not None
                assert self.target.vararg is not None
                vararg_name = self.source.vararg[0]
                assert name.startswith(vararg_name)
                res[name] = name.replace(vararg_name, self.target.vararg[0])
        return res
    
    def bind_dict_forward(self, args:tuple, kwargs:dict, 
                          apply_defaults:bool=CoreConfig.enable_defaults) -> TDict[str, TAny]:
        renamed_kwargs = rename_dict_keys(dct=kwargs, mapping=self.kwarg_map)
        return self.target.bind_args(args=args, 
                                     kwargs=renamed_kwargs,
                                     apply_defaults=apply_defaults)
    
    def fixed_inputs_map(self) -> TDict[str, str]:
        return {**self.arg_map, **self.kwarg_map}
    
    def fixed_outputs_map(self) -> TDict[str, str]:
        return {k: v for k, v in 
                zip(self.source.output_names, self.target.output_names)}

    def rename_source_inputs(self, rename_dict: TDict[str, str]) -> BaseSigMap:
        source = self.source.rename_inputs(rename_dict=rename_dict)
        target = self.target
        kwarg_map = {rename_dict.get(k, k): v for k, v in self.kwarg_map.items()}
        return SigMap(source=source, target=target, kwarg_map=kwarg_map)
    
    def __eq__(self, other:TAny) -> bool:
        if not isinstance(other, SigMap):
            return False
        return self.__dict__ == other.__dict__


def relabel_sig(sig:BaseSignature, arg_map:TDict[str, str]=None, 
                new_vararg:str=None, kwarg_map:TDict[str, str]=None, 
                new_varkwarg:str=None, 
                output_map:TDict[str, str]=None) -> BaseSigMap:
    """
    Given maps along which to rename signature elements, generate a new
    signature and an associated signature mapping from the original signature to
    the new signature.
    """
    arg_map = {} if arg_map is None else arg_map
    kwarg_map = {} if kwarg_map is None else kwarg_map
    if output_map is not None and (not sig.has_fixed_outputs):
        raise ValueError()
    output_map = {} if output_map is None else output_map
    defaults_key_map = {**arg_map, **kwarg_map}
    ord_args = [(arg_map[name], tp) for name, tp in sig.ord_poskw]
    vararg = None if sig.vararg is None else (new_vararg, sig.vararg[1])
    kwargs = {kwarg_map[name]: tp for name, tp in sig.kw.items()}
    varkwarg = None if sig.varkwarg is None else (new_varkwarg, sig.varkwarg[1])
    if sig.has_fixed_outputs:
        ord_outputs = [(output_map[name], tp) for name, tp in sig.ord_outputs]
        fixed_outputs = True
    else:
        ord_outputs = None
        fixed_outputs = False
    defaults = {defaults_key_map[k]: v for k, v in sig.defaults.items()}
    renamed_sig = Signature(
        ord_poskw=ord_args, kw=kwargs,
        ord_outputs=ord_outputs, vararg=vararg,
        varkwarg=varkwarg, defaults=defaults,
        fixed_outputs=fixed_outputs
    )
    sig_map = SigMap(source=sig, target=renamed_sig, kwarg_map=kwarg_map)
    return sig_map