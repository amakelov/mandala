from common_imports import *
from functools import wraps, partial
from model import *

################################################################################
### interfaces
################################################################################
class OpDecorator:
    def __init__(self, output_names: Optional[List[str]] = None,
                 nout: Union[Literal['var', 'auto'], int] = 'auto', 
                 skip_inputs: Optional[List[str]] = None,
                 skip_outputs: Optional[List[str]] = None,
                 __structural__: bool = False
                 ) -> None:
        self.output_names = output_names
        self.skip_inputs = skip_inputs
        self.skip_outputs = skip_outputs
        self.nout = nout
        self.__structural__ = __structural__
    
    def __call__(self, f: Callable) -> 'f':
        return Op(f.__name__, f, output_names=self.output_names, nout=self.nout, __structural__=self.__structural__, skip_inputs=self.skip_inputs, skip_outputs=self.skip_outputs)
        # @wraps(f)
        # def wrapper(*args, **kwargs):
        #     return Op(f, output_names=self.output_names, nout=self.nout, __structural__=self.__structural__)(*args, **kwargs)
        # return wrapper

op = OpDecorator


class Context:

    current_context: Optional["Context"] = None

    def __init__(self, storage: "Storage") -> None:
        self.storage = storage
    
    def __enter__(self) -> "Storage":
        Context.current_context = self
        return self.storage
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        Context.current_context = None

################################################################################
### builtin collections
################################################################################

from cf import ComputationFrame
from storage import Storage