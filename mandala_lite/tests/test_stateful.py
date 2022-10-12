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
from mandala_lite.tests.stateful_utils import *
from mandala_lite.core.workflow import Workflow, CallStruct
from mandala_lite.core.utils import Hashing, get_uid
from mandala_lite.core.compiler import *
from mandala_lite.storages.remote_storage import RemoteStorage
from mandala_lite.ui.main import SimpleWorkflowExecutor
from mandala_lite.ui.funcs import synchronize_op


class ClientState:
    def __init__(self, root: RemoteStorage):
        self.storage = Storage(root=root)
        self._workflows: List[Workflow] = []
        self._func_ops: List[FuncOp] = []
        self._num_func_renames = 0
        self._num_input_renames = 0


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
    def create_op(instance: "SingleClientSimulator") -> Tuple[bool, List[ClientState]]:
        # return clients for which an op can be created
        candidates = [
            c for c in instance.clients if len(c._func_ops) < Preconditions.MAX_OPS
        ]
        return len(candidates) > 0, candidates

    ############################################################################
    ### refactoring
    ############################################################################
    @staticmethod
    def add_input(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, op_idx) for which an input can be added
        candidates = []
        for c in instance.clients:
            for idx, func_op in enumerate(c._func_ops):
                if len(func_op.sig.input_names) < Preconditions.MAX_INPUTS_PER_OP:
                    candidates.append((c, func_op, idx))
        return len(candidates) > 0, candidates

    @staticmethod
    def rename_func(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, idx) for which the op can be renamed
        candidates = []
        for c in instance.clients:
            if c._num_func_renames < Preconditions.MAX_FUNC_RENAMES:
                for idx, func_op in enumerate(c._func_ops):
                    candidates.append((c, func_op, idx))
        return len(candidates) > 0, candidates

    @staticmethod
    def rename_input(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, idx) for which an input can be renamed
        candidates = []
        for c in instance.clients:
            if c._num_input_renames < Preconditions.MAX_INPUT_RENAMES:
                candidates += [(c, op, idx) for idx, op in enumerate(c._func_ops)]
        return len(candidates) > 0, candidates

    @staticmethod
    def create_new_version(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, idx of this op) for which a new version can be created
        candidates = []
        for c in instance.clients:
            num_ops = len(c._func_ops)
            if num_ops > 0 and num_ops < Preconditions.MAX_OPS:
                candidates += [(c, op, idx) for idx, op in enumerate(c._func_ops)]
        return len(candidates) > 0, candidates

    ############################################################################
    ### growing workflows
    ############################################################################
    @staticmethod
    def create_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[ClientState]]:
        # return clients for which a workflow can be created
        candidates = [
            c
            for c in instance.clients
            if len(c._workflows) < Preconditions.MAX_WORKFLOWS
        ]
        return len(candidates) > 0, candidates

    @staticmethod
    def add_input_var_to_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow]]]:
        # return tuples of (client, workflow) for which an input var can be added
        candidates = []
        for c in instance.clients:
            for w in c._workflows:
                if w.shape_size < Preconditions.MAX_WORKFLOW_SIZE_TO_ADD_VAR:
                    candidates.append((c, w))
        return len(candidates) > 0, candidates

    @staticmethod
    def add_op_to_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow, FuncOp]]]:
        # return tuples of (client, workflow, op) for which an op can be added
        candidates = []
        for c in instance.clients:
            for w in c._workflows:
                if (
                    w.shape_size < Preconditions.MAX_WORKFLOW_SIZE_TO_ADD_OP
                    and len(w.var_nodes) > 0
                ):
                    candidates += [(c, w, op) for op in c._func_ops]
        return len(candidates) > 0, candidates

    @staticmethod
    def add_call_to_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow, FuncQuery]]]:
        # return tuples of (client, workflow, op_node) for which a call can be added
        candidates = []
        for c in instance.clients:
            for w in c._workflows:
                candidates += [(c, w, op_node) for op_node in w.callable_op_nodes]
        return len(candidates) > 0, candidates

    @staticmethod
    def execute_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow]]]:
        # return tuples of (client, workflow) for which a workflow can be
        # executed
        candidates = []
        for c in instance.clients:
            for w in c._workflows:
                if w.num_calls > 0:
                    candidates.append((c, w))
        return len(candidates) > 0, candidates

    @staticmethod
    def query_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow]]]:
        # return tuples of (client, workflow) for which a workflow can be
        # queried
        candidates = []
        for c in instance.clients:
            for w in c._workflows:
                if len(w.var_nodes) > 0 and w.is_saturated:
                    candidates.append((c, w))
        return len(candidates) > 0, candidates


class SingleClientSimulator(RuleBasedStateMachine):
    def __init__(self, n_clients: int = 1):
        super().__init__()
        client = mongomock.MongoClient()
        root = MongoMockRemoteStorage(db_name="test", client=client)
        self.clients = [ClientState(root=root) for _ in range(n_clients)]
        # self.storage = Storage() if storage is None else storage
        # self._workflows: List[Workflow] = []
        # self._func_ops: List[FuncOp] = []
        # # to avoid using too many transitions on renaming
        # self._num_func_renames = 0
        # self._num_input_renames = 0
        #! keep everything deterministic. only use `random` for generating
        #! stuff
        random.seed(0)

    ############################################################################
    ### schema modifications
    ############################################################################
    @precondition(lambda machine: Preconditions.create_op(machine)[0])
    @rule()
    def create_op(self):
        """
        Add a random op to the storage.
        """
        client = random.choice(Preconditions.create_op(self)[1])
        new_func_op = make_op(
            ui_name=random_string(size=10),
            input_names=[random_string(size=10) for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=0,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client._func_ops.append(new_func_op)

    @precondition(lambda machine: Preconditions.add_input(machine)[0])
    @rule()
    def add_input(self):
        candidates = Preconditions.add_input(self)[1]
        client, func_op, idx = random.choice(candidates)
        # idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[idx]
        f = func_op.func
        sig = func_op.sig
        # simulate update using low-level API
        new_sig = sig.create_input(name=random_string(size=10), default=23)
        # TODO: provide a new function with extra input as a user would
        new_func_op = FuncOp._from_data(f=f, sig=new_sig)
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client._func_ops[idx] = new_func_op

    @precondition(lambda machine: Preconditions.rename_func(machine)[0])
    @rule()
    def rename_func(self):
        candidates = Preconditions.rename_func(self)[1]
        client, func_op, idx = random.choice(candidates)
        # idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[idx]
        new_name = random_string(size=10)
        # find and rename all versions
        all_versions = [
            (i, other_func_op)
            for i, other_func_op in enumerate(client._func_ops)
            if other_func_op.sig.internal_name == func_op.sig.internal_name
        ]
        rename_done = False
        for version_idx, func_op_version in all_versions:
            if not rename_done:
                # use the API the user would use to rename. This will rename all
                # versions.
                func_interface = FuncInterface(func_op=func_op_version)
                synchronize(func=func_interface, storage=client.storage)
                new_sig = rename_func(
                    storage=client.storage, func=func_interface, new_name=new_name
                )
                # sess.d = locals()
                # raise
                rename_done = True
            else:
                # after the rename, get the true signature from the storage
                new_sig = client.storage.sig_adapter.load_state()[
                    func_op_version.sig.internal_name, func_op_version.sig.version
                ]
            # now update the state of the simulator
            new_func_op_version = FuncOp._from_data(f=func_op_version.func, sig=new_sig)
            synchronize_op(func_op=new_func_op_version, storage=client.storage)
            client._func_ops[version_idx] = new_func_op_version
        client._num_func_renames += 1

    @precondition(lambda machine: Preconditions.rename_input(machine)[0])
    @rule()
    def rename_input(self):
        candidates = Preconditions.rename_input(self)[1]
        client, func_op, func_idx = random.choice(candidates)
        # func_idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[func_idx]
        input_to_rename = random.choice(sorted(list(func_op.sig.input_names)))
        new_name = random_string(size=10)
        # use the API the user would use to rename
        func_interface = FuncInterface(func_op=func_op)
        synchronize(func=func_interface, storage=client.storage)
        new_sig = rename_arg(
            storage=client.storage,
            func=func_interface,
            name=input_to_rename,
            new_name=new_name,
        )
        # now update the state of the simulator
        new_func_op = FuncOp._from_data(f=func_op.func, sig=new_sig)
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client._func_ops[func_idx] = new_func_op
        client._num_input_renames += 1

    @precondition(lambda machine: Preconditions.create_new_version(machine)[0])
    @rule()
    def create_new_version(self):
        candidates = Preconditions.create_new_version(self)[1]
        client, func_op, func_idx = random.choice(candidates)
        # func_idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[func_idx]
        latest_version = client.storage.sig_adapter.get_latest_version(sig=func_op.sig)
        new_version = latest_version.version + 1
        new_func_op = make_op(
            ui_name=func_op.sig.ui_name,
            input_names=[random_string(size=10) for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=new_version,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client._func_ops.append(new_func_op)

    ############################################################################
    ### generating workflows
    ############################################################################
    @precondition(lambda machine: Preconditions.create_workflow(machine)[0])
    @rule()
    def add_workflow(self):
        """
        Add a new (empty) workflow to the test.
        """
        candidates = Preconditions.create_workflow(self)[1]
        client = random.choice(candidates)
        res = Workflow()
        client._workflows.append(res)

    @precondition(lambda machine: Preconditions.add_input_var_to_workflow(machine)[0])
    @rule()
    def add_input_var_to_workflow(self):
        candidates = Preconditions.add_input_var_to_workflow(self)[1]
        client, workflow = random.choice(candidates)
        # workflow = random.choice([w for w in self._workflows if w.shape_size < 5])
        # always add a value to make sampling proceed faster
        var = workflow.add_var()
        workflow.add_value(value=wrap(get_uid()), var=var)

    @precondition(lambda machine: Preconditions.add_op_to_workflow(machine)[0])
    @rule()
    def add_op_to_workflow(self):
        """
        Add an instance of some op to some workflow.
        """
        candidates = Preconditions.add_op_to_workflow(self)[1]
        client, workflow, func_op = random.choice(candidates)
        # func_op = random.choice(self._func_ops)
        # workflow = random.choice([w for w in self._workflows if len(w.var_nodes) > 0])
        # pick inputs randomly from workflow
        inputs = {
            name: random.choice(workflow.var_nodes) for name in func_op.sig.input_names
        }
        # add function over these inputs
        op_node, output_nodes = workflow.add_op(inputs=inputs, func_op=func_op)

    @precondition(lambda machine: Preconditions.add_call_to_workflow(machine)[0])
    @rule()
    def add_call_to_workflow(self):
        candidates = Preconditions.add_call_to_workflow(self)[1]
        client, workflow, op_node = random.choice(candidates)
        # workflow = random.choice(
        #     [w for w in self._workflows if len(w.callable_op_nodes) > 0]
        # )
        # # pick random op in workflow
        # op_node = random.choice(workflow.callable_op_nodes)
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

    @precondition(lambda machine: Preconditions.execute_workflow(machine)[0])
    @rule()
    def execute_workflow(self):
        candidates = Preconditions.execute_workflow(self)[1]
        client, workflow = random.choice(candidates)
        # workflow = random.choice([w for w in self._workflows if w.num_calls > 0])
        calls = SimpleWorkflowExecutor().execute(
            workflow=workflow, storage=client.storage
        )
        client.storage.commit(calls=calls)

    @invariant()
    def verify_state(self):
        for client in self.clients:
            # make sure that functions called on the storage work
            client.storage.rel_adapter.get_all_call_data()
            # check storage invariants
            check_invariants(storage=client.storage)
            # check invariants on the workflows
            for w in client._workflows:
                w.check_invariants()

    def check_sig_synchronization(self):
        for client in self.clients:
            assert client.storage.root.sigs == client.storage.sig_adapter.load_state()

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
TestCase.settings = settings(max_examples=100, deadline=None, stateful_step_count=50)
