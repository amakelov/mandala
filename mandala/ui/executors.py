from ..common_imports import *
from ..queries.workflow import Workflow
from abc import ABC, abstractmethod
from ..core.model import Call
from ..core.config import MODES
from ..core.wrapping import unwrap


class WorkflowExecutor(ABC):
    @abstractmethod
    def execute(self, workflow: Workflow, storage: "storage.Storage") -> List[Call]:
        pass


class SimpleWorkflowExecutor(WorkflowExecutor):
    def execute(self, workflow: Workflow, storage: "storage.Storage") -> List[Call]:
        result = []
        for op_node in workflow.op_nodes:
            call_structs = workflow.op_node_to_call_structs[op_node]
            for call_struct in call_structs:
                func_op, inputs, outputs = (
                    call_struct.func_op,
                    call_struct.inputs,
                    call_struct.outputs,
                )
                assert all([not inp.is_delayed() for inp in inputs.values()])
                inputs = unwrap(inputs)
                fi = funcs.FuncInterface(func_op=func_op)
                with storage.run():
                    _, call = fi.call(args=tuple(), kwargs=inputs)
                vref_outputs = call.outputs
                # vref_outputs, call, _ = r.call(args=tuple(), kwargs=inputs, conn=None)
                # vref_outputs, call, _ = storage.call_run(
                #     func_op=func_op,
                #     inputs=inputs,
                #     _call_depth=0,
                # )
                # overwrite things
                for output, vref_output in zip(outputs, vref_outputs):
                    output._obj = vref_output.obj
                    output.uid = vref_output.uid
                    output.in_memory = True
                result.append(call)
        # filter out repeated calls
        result = list({call.uid: call for call in result}.values())
        return result


from . import storage
from . import runner
from . import contexts
from . import funcs
