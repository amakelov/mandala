from ..common_imports import *
from .weaver import *
from ..core.model import Ref, FuncOp
from ..core.utils import Hashing
from ..core.config import dump_output_name, parse_output_idx


class CallStruct:
    def __init__(self, func_op: FuncOp, inputs: Dict[str, Ref], outputs: List[Ref]):
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
        # note that there may be many var nodes with the same causal hash
        self.var_node_to_causal_hash: Dict[ValQuery, str] = {}
        # in topological order
        self.op_nodes: List[FuncQuery] = []
        # self.causal_hash_to_op_node: Dict[str, FuncQuery] = {}
        self.op_node_to_causal_hash: Dict[FuncQuery, str] = {}
        ### encoding instance data
        # multiple refs may map to the same query node
        self.value_to_var: Dict[Ref, ValQuery] = {}
        # for a given op node, there may be multiple call structs
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
                output_locations = {
                    dump_output_name(i): self.value_to_var[v]
                    for i, v in enumerate(call_struct.outputs)
                }
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
        # return [var for var in self.var_nodes if var.creator is None]
        return [var for var in self.var_nodes if len(var.creators) == 0]

    def var_to_values(self) -> Dict[ValQuery, List[Ref]]:
        res = defaultdict(list)
        for value, var in self.value_to_var.items():
            res[var].append(value)
        return res

    def add_var(self, val_query: Optional[ValQuery] = None) -> ValQuery:
        res = (
            val_query
            if val_query is not None
            # else ValQuery(creator=None, created_as=None)
            else ValQuery(creators=[], created_as=[], constraint=None, tp=AnyType())
        )
        # if res.creator is None:
        if len(res.creators) == 0:
            causal_hash = self.get_default_hash()
        else:
            # creator_hash = self.op_node_to_causal_hash[res.creator]
            creator_hash = self.op_node_to_causal_hash[res.creators[0]]
            # causal_hash = Hashing.get_content_hash(obj=[creator_hash,
            # res.created_as])
            causal_hash = Hashing.get_content_hash(
                obj=[creator_hash, res.created_as[0]]
            )
        self.var_nodes.append(res)
        self.var_node_to_causal_hash[res] = causal_hash
        return res

    def get_op_hash(
        self,
        func_op: FuncOp,
        node_inputs: Optional[Dict[str, ValQuery]] = None,
        val_inputs: Optional[Dict[str, Ref]] = None,
    ) -> str:
        assert (node_inputs is None) != (val_inputs is None)
        if val_inputs is not None:
            node_inputs = {
                name: self.value_to_var[val] for name, val in val_inputs.items()
            }
        assert node_inputs is not None
        input_causal_hashes = {
            name: self.var_node_to_causal_hash[val] for name, val in node_inputs.items()
        }
        input_causal_hashes = {
            k: v for k, v in input_causal_hashes.items() if v != self.get_default_hash()
        }
        input_causal_hashes = sorted(input_causal_hashes.items())
        op_representation = [
            input_causal_hashes,
            func_op.sig.versioned_internal_name,
        ]
        causal_hash = Hashing.get_content_hash(obj=op_representation)
        return causal_hash

    def add_op(
        self,
        inputs: Dict[str, ValQuery],
        func_op: FuncOp,
    ) -> Tuple[FuncQuery, Dict[str, ValQuery]]:
        # TODO: refactor the `FuncQuery` creation here
        res = FuncQuery(inputs=inputs, func_op=func_op, outputs={}, constraint=None)
        causal_hash = self.get_op_hash(node_inputs=inputs, func_op=func_op)
        self.op_nodes.append(res)
        self.op_node_to_causal_hash[res] = causal_hash
        # create outputs
        outputs = {}
        for i in range(func_op.sig.n_outputs):
            # output = self.add_var(val_query=ValQuery(creator=res,
            # created_as=i))
            output_name = dump_output_name(index=i)
            output = self.add_var(
                val_query=ValQuery(
                    creators=[res],
                    created_as=[output_name],
                    constraint=None,
                    tp=AnyType(),
                )
            )
            outputs[output_name] = output
        # assign outputs to op
        res.set_outputs(outputs=outputs)
        self.op_node_to_call_structs[res] = []
        return res, outputs

    def add_value(self, value: Ref, var: ValQuery):
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
        op_hash = self.get_op_hash(func_op=func_op, val_inputs=inputs)
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
        outputs_dict = {dump_output_name(i): output for i, output in enumerate(outputs)}
        for k in outputs_dict.keys():
            self.value_to_var[outputs_dict[k]] = output_nodes[k]
        self.op_node_to_call_structs[op_node].append(call_struct)

    ############################################################################
    ###
    ############################################################################
    @staticmethod
    def from_call_structs(call_structs: List[CallStruct]) -> "Workflow":
        """
        Assumes calls are given in topological order
        """
        res = Workflow()
        input_var = res.add_var()
        for call_struct in call_structs:
            inputs = call_struct.inputs
            for inp in inputs.values():
                if inp not in res.value_to_var.keys():
                    res.add_value(value=inp, var=input_var)
            res.add_call_struct(call_struct)
        return res

    @staticmethod
    def from_traversal(
        vqs: List[ValQuery],
    ) -> Tuple["Workflow", Dict[ValQuery, ValQuery]]:
        vqs, fqs = traverse_all(vqs, direction="backward")
        vqs_topsort = reversed(vqs)
        fqs_topsort = reversed(fqs)
        # input_vqs = [vq for vq in vqs_topsort if vq.creator is None]
        input_vqs = [vq for vq in vqs_topsort if len(vq.creators) == 0]
        res = Workflow()
        vq_to_new_vq = {}
        for vq in input_vqs:
            new_vq = res.add_var(val_query=vq)
            vq_to_new_vq[vq] = new_vq
        for fq in fqs_topsort:
            new_inputs = {name: vq_to_new_vq[vq] for name, vq in fq.inputs.items()}
            new_fq, new_outputs = res.add_op(inputs=new_inputs, func_op=fq.func_op)
            for k in new_outputs.keys():
                vq, new_vq = fq.outputs[k], new_outputs[k]
                vq_to_new_vq[vq] = new_vq
        return res, vq_to_new_vq

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
        return any([value.is_delayed() for value in self.value_to_var.keys()])

    def print_shape(self):
        var_names = {var: f"var_{i}" for i, var in enumerate(self.var_nodes)}
        for var in self.inputs:
            print(f"{var_names[var]} = Q()")
        for op_node in self.op_nodes:
            numbered_outputs = {
                parse_output_idx(k): v for k, v in op_node.outputs.items()
            }
            outputs_list = [numbered_outputs[i] for i in range(len(numbered_outputs))]
            lhs = ", ".join([var_names[var] for var in outputs_list])
            print(
                f"{lhs} = {op_node.func_op.sig.ui_name}("
                + ", ".join(
                    [f"{name}={var_names[var]}" for name, var in op_node.inputs.items()]
                )
                + ")"
            )


class History:
    def __init__(self, workflow: Workflow, node: ValQuery):
        self.workflow = workflow
        self.node = node
