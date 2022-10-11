from ..common_imports import *
from .weaver import *
from .model import ValueRef, FuncOp, Call
from .utils import Hashing
from ..utils import invert_dict


class CallStruct:
    def __init__(
        self, func_op: FuncOp, inputs: Dict[str, ValueRef], outputs: List[ValueRef]
    ):
        self.func_op = func_op
        self.inputs = inputs
        self.outputs = outputs


class Workflow:
    """
    An intermediate representation of a collection of calls, possibly not all of
    which have been executed, following a particular computational graph.

    Used to:
        - represent work to be done in a `batch` context as a data structure
        - encode an entire workflow for e.g. testing scenarios that simulate
          real-world workloads
    """

    def __init__(self):
        ### encoding the shape
        # in topological order
        self.var_nodes: List[ValQuery] = []
        # note that there may be many var nodes with the same causal hash (but
        # they must be among the inputs of the workflow)
        self.var_node_to_causal_hash: Dict[ValQuery, str] = {}
        # in topological order
        self.op_nodes: List[FuncQuery] = []
        # self.causal_hash_to_op_node: Dict[str, FuncQuery] = {}
        self.op_node_to_causal_hash: Dict[FuncQuery, str] = {}
        ### encoding the data
        self.value_to_var: Dict[ValueRef, ValQuery] = {}
        self.op_node_to_call_structs: Dict[FuncQuery, List[CallStruct]] = {}

    def check_invariants(self):
        assert set(self.var_node_to_causal_hash.keys()) == set(self.var_nodes)
        assert set(self.op_node_to_causal_hash.keys()) == set(self.op_nodes)
        assert set(self.op_node_to_call_structs.keys()) == set(self.op_nodes)
        assert set(self.value_to_var.values()) <= set(self.var_nodes)
        for op_node in self.op_nodes:
            for call_struct in self.op_node_to_call_structs[op_node]:
                input_locations = {
                    k: self.value_to_var[v] for k, v in call_struct.inputs.items()
                }
                assert input_locations == op_node.inputs
                output_locations = [self.value_to_var[v] for v in call_struct.outputs]
                assert output_locations == op_node.outputs

    def get_default_hash(self) -> str:
        return Hashing.get_content_hash(obj="null")

    @property
    def callable_op_nodes(self) -> List[FuncQuery]:
        # return op nodes that have non-empty inputs
        res = []
        var_to_values = self.var_to_values()
        for op_node in self.op_nodes:
            if all([len(var_to_values[var]) > 0 for var in op_node.inputs.values()]):
                res.append(op_node)
        return res

    @property
    def inputs(self) -> List[ValQuery]:
        return [var for var in self.var_nodes if var.creator is None]

    # @property
    # def op_to_causal_hash(self) -> Dict[FuncQuery, str]:
    #     return invert_dict(self.causal_hash_to_op_node)

    def var_to_values(self) -> Dict[ValQuery, List[ValueRef]]:
        res = defaultdict(list)
        for value, var in self.value_to_var.items():
            res[var].append(value)
        return res

    def add_var(self, val_query: Optional[ValQuery] = None) -> ValQuery:
        res = (
            val_query
            if val_query is not None
            else ValQuery(creator=None, created_as=None)
        )
        if res.creator is None:
            causal_hash = self.get_default_hash()
        else:
            creator_hash = self.op_node_to_causal_hash[res.creator]
            causal_hash = Hashing.get_content_hash(obj=[creator_hash, res.created_as])
        self.var_nodes.append(res)
        self.var_node_to_causal_hash[res] = causal_hash
        return res

    def add_op(
        self,
        inputs: Dict[str, ValQuery],
        func_op: FuncOp,
    ) -> Tuple[FuncQuery, List[ValQuery]]:
        res = FuncQuery(inputs=inputs, func_op=func_op)
        op_representation = [
            {name: self.var_node_to_causal_hash[inp] for name, inp in inputs.items()},
            func_op.sig.versioned_internal_name,
        ]
        causal_hash = Hashing.get_content_hash(obj=op_representation)
        self.op_nodes.append(res)
        self.op_node_to_causal_hash[res] = causal_hash
        # self.causal_hash_to_op_node[causal_hash] = res
        # create outputs
        outputs = []
        for i in range(func_op.sig.n_outputs):
            output = self.add_var(val_query=ValQuery(creator=res, created_as=i))
            outputs.append(output)
        # assign outputs to op
        res.set_outputs(outputs=outputs)
        self.op_node_to_call_structs[res] = []
        return res, outputs

    def add_value(self, value: ValueRef, var: ValQuery):
        assert var in self.var_nodes
        self.value_to_var[value] = var

    def add_call_struct(self, call_struct: CallStruct):
        # process inputs
        func_op, inputs, outputs = (
            call_struct.func_op,
            call_struct.inputs,
            call_struct.outputs,
        )
        if any([inp not in self.value_to_var.keys() for inp in inputs.values()]):
            raise NotImplementedError()
        # process call
        op_representation = [
            {
                name: self.var_node_to_causal_hash[self.value_to_var[inp]]
                for name, inp in inputs.items()
            },
            func_op.sig.versioned_internal_name,
        ]
        op_hash = Hashing.get_content_hash(obj=op_representation)
        if op_hash not in self.op_node_to_causal_hash.values():
            # create op
            op_node, output_nodes = self.add_op(
                inputs={name: self.value_to_var[inp] for name, inp in inputs.items()},
                func_op=func_op,
            )
        else:
            candidates = [
                op_node
                for op_node in self.op_nodes
                if self.op_node_to_causal_hash[op_node] == op_hash
                and op_node.inputs
                == {name: self.value_to_var[inp] for name, inp in inputs.items()}
            ]
            op_node = candidates[0]
            output_nodes = op_node.outputs
        # process outputs
        for output_node, output in zip(output_nodes, outputs):
            self.value_to_var[output] = output_node
        self.op_node_to_call_structs[op_node].append(call_struct)

    ############################################################################
    ###
    ############################################################################
    @staticmethod
    def from_call_structs(call_structs: List[CallStruct]) -> "Workflow":
        res = Workflow()
        input_var = res.add_var()
        for call_struct in call_structs:
            _, inputs, _ = call_struct.func_op, call_struct.inputs, call_struct.outputs
            for inp in inputs.values():
                if inp not in res.value_to_var.keys():
                    res.add_value(value=inp, var=input_var)
            res.add_call_struct(call_struct)
        return res

    ############################################################################
    ###
    ############################################################################
    @property
    def empty(self) -> bool:
        return len(self.value_to_var) == 0

    @property
    def shape_size(self) -> int:
        return len(self.op_nodes) + len(self.var_nodes)

    @property
    def num_calls(self) -> int:
        return sum(
            [
                len(call_structs)
                for call_structs in self.op_node_to_call_structs.values()
            ]
        )

    @property
    def is_saturated(self) -> bool:
        var_to_values = self.var_to_values()
        return all([len(var_to_values[var]) > 0 for var in self.var_nodes])

    @property
    def has_delayed(self) -> bool:
        return any(
            [ValueRef.is_delayed(vref=value) for value in self.value_to_var.keys()]
        )

    def print_shape(self):
        var_names = {var: f"var_{i}" for i, var in enumerate(self.var_nodes)}
        for var in self.inputs:
            print(f"{var_names[var]} = Q()")
        for op_node in self.op_nodes:
            lhs = ", ".join([var_names[var] for var in op_node.outputs])
            print(
                f"{lhs} = {op_node.func_op.sig.ui_name}("
                + ", ".join(
                    [f"{name}={var_names[var]}" for name, var in op_node.inputs.items()]
                )
                + ")"
            )
