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
from mandala_lite.core.workflow import Workflow, CallStruct
from mandala_lite.core.utils import Hashing, get_uid
from mandala_lite.core.compiler import *
from mandala_lite.ui.main import SimpleWorkflowExecutor
from mandala_lite.ui.funcs import synchronize_op


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
    return FuncOp._from_data(f=f, sig=sig)


class Preconditions:
    """
    A namespace for preconditions for the rules of the state machine.

    NOTE: Preconditions are defined as functions instead of lambdas to enable
    type introspection, autorefactoring, etc.
    """

    # control some of the transitions to avoid long chains, especially ones that
    # make the DB larger
    #! (does this actually optimize things? we need to benchmark)
    MAX_OPS = 10
    MAX_INPUTS_PER_OP = 5
    MAX_WORKFLOWS = 5
    MAX_WORKFLOW_SIZE_TO_ADD_VAR = 5
    MAX_WORKFLOW_SIZE_TO_ADD_OP = 10
    # prevent too many renames
    MAX_FUNC_RENAMES = 20
    MAX_INPUT_RENAMES = 20

    @staticmethod
    def create_op(instance: "SingleClientSimulator") -> bool:
        return len(instance._func_ops) < Preconditions.MAX_OPS

    ############################################################################
    ### refactoring
    ############################################################################
    @staticmethod
    def add_input(instance: "SingleClientSimulator") -> bool:
        return len(instance._func_ops) > 0 and all(
            len(op.sig.input_names) < Preconditions.MAX_INPUTS_PER_OP
            for op in instance._func_ops
        )

    @staticmethod
    def rename_func(instance: "SingleClientSimulator") -> bool:
        return (
            len(instance._func_ops) > 0
            and instance._num_func_renames < Preconditions.MAX_FUNC_RENAMES
        )

    @staticmethod
    def create_new_version(instance: "SingleClientSimulator") -> bool:
        return len(instance._func_ops) > 0 and Preconditions.create_op(
            instance=instance
        )

    @staticmethod
    def rename_input(instance: "SingleClientSimulator") -> bool:
        return (
            len(instance._func_ops) > 0
            and any(len(op.sig.input_names) > 0 for op in instance._func_ops)
            and instance._num_input_renames < Preconditions.MAX_INPUT_RENAMES
        )

    @staticmethod
    def create_workflow(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) < Preconditions.MAX_WORKFLOWS

    @staticmethod
    def add_input_var_to_workflow(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) > 0 and any(
            wf.shape_size < Preconditions.MAX_WORKFLOW_SIZE_TO_ADD_VAR
            for wf in instance._workflows
        )

    @staticmethod
    def add_op_to_workflow(instance: "SingleClientSimulator") -> bool:
        return (
            len(instance._func_ops) > 0
            and any(
                wf.shape_size < Preconditions.MAX_WORKFLOW_SIZE_TO_ADD_OP
                for wf in instance._workflows
            )
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
        self._func_ops: List[FuncOp] = []
        # to avoid using too many transitions on renaming
        self._num_func_renames = 0
        self._num_input_renames = 0
        #! keep everything deterministic. only use `random` for generating
        #! stuff
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
        new_func_op = make_op(
            ui_name=random_string(size=10),
            input_names=[random_string(size=10) for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=0,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=self.storage)
        self._func_ops.append(new_func_op)

    @precondition(Preconditions.add_input)
    @rule()
    def add_input(self):
        idx = random.randint(0, len(self._func_ops) - 1)
        func_op = self._func_ops[idx]
        f = func_op.func
        sig = func_op.sig
        # simulate update using low-level API
        new_sig = sig.create_input(name=random_string(size=10), default=23)
        # TODO: provide a new function with extra input as a user would
        new_func_op = FuncOp._from_data(f=f, sig=new_sig)
        synchronize_op(func_op=new_func_op, storage=self.storage)
        self._func_ops[idx] = new_func_op

    @precondition(Preconditions.rename_func)
    @rule()
    def rename_func(self):
        idx = random.randint(0, len(self._func_ops) - 1)
        func_op = self._func_ops[idx]
        new_name = random_string(size=10)
        # find and rename all versions
        all_versions = [
            (i, other_func_op)
            for i, other_func_op in enumerate(self._func_ops)
            if other_func_op.sig.internal_name == func_op.sig.internal_name
        ]
        rename_done = False
        for version_idx, func_op_version in all_versions:
            if not rename_done:
                # use the API the user would use to rename. This will rename all
                # versions.
                func_interface = FuncInterface(func_op=func_op_version)
                synchronize(func=func_interface, storage=self.storage)
                new_sig = rename_func(
                    storage=self.storage, func=func_interface, new_name=new_name
                )
                rename_done = True
            else:
                # after the rename, get the true signature from the storage
                new_sig = self.storage.sig_adapter.load_state()[
                    func_op_version.sig.internal_name, func_op_version.sig.version
                ]
            # now update the state of the simulator
            new_func_op_version = FuncOp._from_data(f=func_op_version.func, sig=new_sig)
            synchronize_op(func_op=new_func_op_version, storage=self.storage)
            self._func_ops[version_idx] = new_func_op_version
        self._num_func_renames += 1

    @precondition(Preconditions.rename_input)
    @rule()
    def rename_input(self):
        func_idx = random.randint(0, len(self._func_ops) - 1)
        func_op = self._func_ops[func_idx]
        input_to_rename = random.choice(sorted(list(func_op.sig.input_names)))
        new_name = random_string(size=10)
        # use the API the user would use to rename
        func_interface = FuncInterface(func_op=func_op)
        synchronize(func=func_interface, storage=self.storage)
        new_sig = rename_arg(
            storage=self.storage,
            func=func_interface,
            name=input_to_rename,
            new_name=new_name,
        )
        # now update the state of the simulator
        new_func_op = FuncOp._from_data(f=func_op.func, sig=new_sig)
        synchronize_op(func_op=new_func_op, storage=self.storage)
        self._func_ops[func_idx] = new_func_op
        self._num_input_renames += 1

    @precondition(Preconditions.create_new_version)
    @rule()
    def create_new_version(self):
        func_idx = random.randint(0, len(self._func_ops) - 1)
        func_op = self._func_ops[func_idx]
        latest_version = self.storage.sig_adapter.get_latest_version(sig=func_op.sig)
        new_version = latest_version.version + 1
        new_func_op = make_op(
            ui_name=func_op.sig.ui_name,
            input_names=[random_string(size=10) for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=new_version,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=self.storage)
        self._func_ops.append(new_func_op)

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
        func_op = random.choice(self._func_ops)
        workflow = random.choice([w for w in self._workflows if len(w.var_nodes) > 0])
        # pick inputs randomly from workflow
        inputs = {
            name: random.choice(workflow.var_nodes) for name in func_op.sig.input_names
        }
        # add function over these inputs
        op_node, output_nodes = workflow.add_op(inputs=inputs, func_op=func_op)

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
        call_struct = CallStruct(
            func_op=op_node.func_op,
            inputs=input_values,
            outputs=[
                ValueRef.make_delayed() for _ in range(op_node.func_op.sig.n_outputs)
            ],
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
    def verify_state(self):
        # make sure that functions called on the storage work
        self.storage.rel_adapter.get_all_call_data()
        # check storage invariants
        check_invariants(storage=self.storage)
        # check invariants on the workflows
        for w in self._workflows:
            w.check_invariants()

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
