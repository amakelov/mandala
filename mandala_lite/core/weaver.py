from ..common_imports import *
from .model import FuncOp


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

    def __init__(self, creator: Optional["FuncQuery"], created_as: Optional[int]):
        self.creator = creator
        self.created_as = created_as
        self.consumers: List["FuncQuery"] = []
        self.consumed_as: List[str] = []
        self.aliases = []
        self.column_name = None

    def add_consumer(self, consumer: "FuncQuery", consumed_as: str):
        self.consumers.append(consumer)
        self.consumed_as.append(consumed_as)

    def neighbors(self) -> List["FuncQuery"]:
        res = []
        if self.creator is not None:
            res.append(self.creator)
        for cons in self.consumers:
            res.append(cons)
        return res

    def named(self, name: str) -> "ValQuery":
        self.column_name = name
        return self


class FuncQuery:
    """
    Represents a call to an operation under the `query` interpretation of code.

    This is the equivalent to a `Call` when in a query context. In SQL terms, it
    points to the memoization table of some function. The `inputs` and `outputs`
    connected to it are the `ValQuery` objects that represent the inputs and
    outputs of this call. They are indexed by *internal* names. See `core.sig.Signature`
    for an explanation.
    """

    def __init__(self, inputs: Dict[str, ValQuery], op: FuncOp):
        self.inputs = inputs
        self.outputs = []
        self.op = op

    def set_outputs(self, outputs: List[ValQuery]):
        self.outputs = outputs

    def neighbors(self) -> List[ValQuery]:
        return [
            x
            for x in itertools.chain(
                self.inputs.values(), self.outputs if self.outputs is not None else []
            )
        ]
