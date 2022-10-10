from hypothesis.stateful import (
    RuleBasedStateMachine,
    Bundle,
    rule,
    initialize,
    precondition,
    invariant,
    run_state_machine_as_test,
)
from hypothesis._settings import settings, Verbosity
from hypothesis import strategies as st
import string

from mandala_lite.common_imports import *
from mandala_lite.all import *
from mandala_lite.tests.utils import *
from mandala_lite.core.workflow import Workflow
from mandala_lite.core.utils import Hashing, get_uid
from mandala_lite.core.compiler import *
from mandala_lite.ui.main import SimpleWorkflowExecutor
from mandala_lite.ui.funcs import synchronize_op

import cloudpickle


def combine_inputs(*args, **kwargs) -> str:
    return Hashing.get_content_hash(obj=(args, kwargs))


def generate_deterministic(seed: str, n_outputs: int) -> List[str]:
    result = []
    current = seed
    for i in range(n_outputs):
        new = Hashing.get_content_hash(obj=current)
        result.append(new)
        current = new
    return result


def random_string(size: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(size))


def make_op(
    ui_name: str,
    input_names: List[str],
    n_outputs: int,
    defaults: Dict[str, Any],
    version: int = 0,
    deterministic: bool = True,
) -> FuncOp:
    """
    Generate a deterministic function with given interface
    """
    sig = Signature(
        ui_name=ui_name,
        input_names=set(input_names),
        n_outputs=n_outputs,
        version=version,
        defaults=defaults,
    )
    if n_outputs == 0:
        f = lambda *args, **kwargs: None
    elif n_outputs == 1:
        f = lambda *args, **kwargs: generate_deterministic(
            seed=combine_inputs(*args, **kwargs), n_outputs=1
        )[0]
    else:
        f = lambda *args, **kwargs: tuple(
            generate_deterministic(
                seed=combine_inputs(*args, **kwargs), n_outputs=n_outputs
            )
        )
    return FuncOp.from_data(f=f, sig=sig)


class Preconditions:
    @staticmethod
    def create_op(instance: "SingleClientSimulator") -> bool:
        return len(instance._ops) < 5

    @staticmethod
    def create_workflow(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) < 5

    @staticmethod
    def add_input_var_to_workflow(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) > 0 and any(
            wf.shape_size < 5 for wf in instance._workflows
        )

    @staticmethod
    def add_op_to_workflow(instance: "SingleClientSimulator") -> bool:
        return (
            len(instance._ops) > 0
            and any(wf.shape_size < 10 for wf in instance._workflows)
            and any(len(wf.var_nodes) > 0 for wf in instance._workflows)
        )

    @staticmethod
    def add_call_to_workflow(instance: "SingleClientSimulator") -> bool:
        return any(len(wf.callable_op_nodes) > 0 for wf in instance._workflows)

    @staticmethod
    def execute_workflow(instance: "SingleClientSimulator") -> bool:
        return any(wf.num_calls > 0 for wf in instance._workflows)

    @staticmethod
    def query_workflow(instance: "SingleClientSimulator") -> bool:
        return any(
            len(wf.var_nodes) > 0 and wf.is_saturated for wf in instance._workflows
        )


class SingleClientSimulator(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.storage = Storage()
        self._workflows: List[Workflow] = []
        self._ops: List[FuncOp] = []
        random.seed(0)

    ############################################################################
    ### schema modifications
    ############################################################################
    @precondition(Preconditions.create_op)
    @rule()
    def create_op(self):
        """
        Add a random op to the storage.
        """
        res = make_op(
            ui_name=random_string(size=10),
            input_names=[random_string(size=10) for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=0,
            deterministic=True,
        )
        synchronize_op(op=res, storage=self.storage)
        self._ops.append(res)

    ############################################################################
    ### generating workflows
    ############################################################################
    @precondition(Preconditions.create_workflow)
    @rule()
    def add_workflow(self):
        """
        Add a new (empty) workflow to the test.
        """
        res = Workflow()
        self._workflows.append(res)

    @precondition(Preconditions.add_input_var_to_workflow)
    @rule()
    def add_input_var_to_workflow(self):
        workflow = random.choice([w for w in self._workflows if w.shape_size < 5])
        # always add a value to make sampling proceed faster
        var = workflow.add_var()
        workflow.add_value(value=wrap(get_uid()), var=var)

    @precondition(Preconditions.add_op_to_workflow)
    @rule()
    def add_op_to_workflow(self):
        """
        Add an instance of some op to some workflow.
        """
        op = random.choice(self._ops)
        workflow = random.choice([w for w in self._workflows if len(w.var_nodes) > 0])
        # pick inputs randomly from workflow
        inputs = {
            name: random.choice(workflow.var_nodes) for name in op.sig.input_names
        }
        # add function over these inputs
        op_node, output_nodes = workflow.add_op(inputs=inputs, op=op)

    @precondition(Preconditions.add_call_to_workflow)
    @rule()
    def add_call_to_workflow(self):
        workflow = random.choice(
            [w for w in self._workflows if len(w.callable_op_nodes) > 0]
        )
        # pick random op in workflow
        op_node = random.choice(workflow.callable_op_nodes)
        input_vars = op_node.inputs
        # pick random values
        var_to_values = workflow.var_to_values()
        input_values = {
            name: random.choice(var_to_values[var]) for name, var in input_vars.items()
        }
        call_struct = tuple(
            [
                op_node.op,
                input_values,
                [
                    ValueRef(obj=None, in_memory=False, uid=None)
                    for _ in range(op_node.op.sig.n_outputs)
                ],
            ]
        )
        workflow.add_call_struct(call_struct=call_struct)

    @precondition(Preconditions.execute_workflow)
    @rule()
    def execute_workflow(self):
        workflow = random.choice([w for w in self._workflows if w.num_calls > 0])
        calls = SimpleWorkflowExecutor().execute(
            workflow=workflow, storage=self.storage
        )
        self.storage.commit(calls=calls)

    @invariant()
    def call_storage_funcs(self):
        # make sure that functions called on the storage work
        self.storage.rel_adapter.get_all_call_data()

    # @precondition(Preconditions.query_workflow)
    # @rule()
    # def query_workflow(self):
    #     workflow = random.choice([w for w in self._workflows if w.is_saturated])
    #     path = Path(__file__).parent / f"bug.cloudpickle"
    #     op_nodes = copy.deepcopy(workflow.op_nodes)
    #     db_dump_path = Path(__file__).parent.absolute() / 'db_dump/'
    #     self.storage.rel_storage.execute_no_results(query=f"EXPORT DATABASE '{db_dump_path}';")
    #     data = (self._ops)
    #     with open(path, "wb") as f:
    #         cloudpickle.dump(data, f)
    #     val_queries, op_queries = workflow.var_nodes, workflow.op_nodes
    #     workflow.print_shape()
    #     df = self.storage.execute_query(select_queries=val_queries)


TestCase = SingleClientSimulator.TestCase
TestCase.settings = settings(max_examples=50, deadline=None, stateful_step_count=200)
