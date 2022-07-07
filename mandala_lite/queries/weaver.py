from ..common_imports import *
from ..core.model import FuncOp

class ValQuery:
    """
    Represents an input/output to an operation under the `query` interpretation
    of code.
    
    This is the equivalent of a `ValueRef` when in a query context. In SQL
    terms, it points to the table of objects. What matters more are the
    constraints it is connected to via the `creator` and `consumers` fields,
    which tell us which tables to join and how.
    
    There is always a unique `creator`, which is the operation that returned
    this `ValQuery` object, but potentially many `consumers`, which are
    subsequent calls to operations that consume this `ValQuery` object.
    """
    def __init__(self, creator:'OpQuery'=None, created_as:str=None, 
                 consumers:List['OpQuery']=None, consumed_as:List[str]=None):
        self.creator = creator
        self.created_as = created_as
        self.consumers = [] if consumers is None else consumers
        self.consumed_as = [] if consumed_as is None else consumed_as
        self.aliases = []


class OpQuery:
    """
    Represents a call to an operation under the `query` interpretation of code.

    This is the equivalent to a `Call` when in a query context. In SQL terms, it
    points to the memoization table of some function. The `inputs` and `outputs`
    connected to it are the `ValQuery` objects that represent the inputs and
    outputs of this call.
    """
    def __init__(self, inputs:Dict[str, ValQuery], outputs:List[ValQuery], 
                 op:FuncOp):
        self.inputs = inputs
        self.outputs = outputs
        self.op = op