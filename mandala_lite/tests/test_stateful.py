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


class MockStorage:
    """
    A simple storage simulator that
    - stores all data in memory: calls as tables, vrefs as a dictionary
    - only uses internal names for signatures;
    - can be synced in a "naive" way with another storage: the state of the
    other storage is upserted into this storage.
    """

    def __init__(self):
        self.calls: Dict[str, pd.DataFrame] = {}
        self.values: Dict[str, Any] = {}
        # versioned internal op name -> (internal input name -> default uid)
        self.default_uids: Dict[str, Dict[str, str]] = {}

    def check_invariants(self):
        assert self.default_uids.keys() == self.calls.keys()
        # get all vref uids that appear in calls
        vref_uids_from_calls = []
        for k, df in self.calls.items():
            for col in df.columns:
                if col != Config.uid_col:
                    vref_uids_from_calls += df[col].values.tolist()
        assert set(vref_uids_from_calls) <= set(self.values.keys())
        for versioned_internal_name, defaults in self.default_uids.items():
            df = self.calls[versioned_internal_name]
            for internal_input_name, default_uid in defaults.items():
                assert internal_input_name in df.columns

    def create_op(self, func_op: FuncOp):
        sig = func_op.sig
        self.calls[sig.versioned_internal_name] = pd.DataFrame(
            columns=[Config.uid_col]
            + list(sig.ui_to_internal_input_map.values())
            + [dump_output_name(index=i) for i in range(sig.n_outputs)]
        )
        self.default_uids[sig.versioned_internal_name] = {}

    def add_input(
        self, func_op: FuncOp, internal_name: str, default_value: Any, default_uid: str
    ):
        sig = func_op.sig
        df = self.calls[sig.versioned_internal_name]
        df[internal_name] = [default_uid for _ in range(len(df))]
        self.values[default_uid] = default_value
        self.default_uids[sig.versioned_internal_name][internal_name] = default_uid

    def rename_func(self, func_op: FuncOp, new_name: str):
        pass

    def rename_input(self, func_op: FuncOp, old_name: str, new_name: str):
        pass

    def create_new_version(self, new_version: FuncOp):
        self.create_op(func_op=new_version)

    def add_call(self, call: Call):
        func_op, inputs, outputs = call.func_op, call.inputs, call.outputs
        sig = func_op.sig
        row = {
            Config.uid_col: call.uid,
            **{sig.ui_to_internal_input_map[k]: v.uid for k, v in inputs.items()},
            **{dump_output_name(index=i): v.uid for i, v in enumerate(outputs)},
        }
        df = self.calls[sig.versioned_internal_name]
        # handle stale calls
        for k in df.columns:
            if k not in row.keys():
                row[k] = self.default_uids[sig.versioned_internal_name][k]
        row_df = pd.DataFrame([row])
        if row[Config.uid_col] not in df[Config.uid_col].values:
            self.calls[sig.versioned_internal_name] = pd.concat(
                [df, row_df],
                ignore_index=True,
            )
        for vref in itertools.chain(inputs.values(), outputs):
            self.values[vref.uid] = unwrap(vref)

    def sync_from_other(self, other: "MockStorage"):
        for k, v in other.values.items():
            if k in self.values.keys():
                assert v == self.values[k]
            self.values[k] = v
        for versioned_internal_name, defaults in other.default_uids.items():
            if versioned_internal_name in self.calls.keys():
                df = self.calls[versioned_internal_name]
                for internal_input_name, default_uid in defaults.items():
                    if internal_input_name not in df.columns:
                        df[internal_input_name] = [default_uid for _ in range(len(df))]
                        self.default_uids[versioned_internal_name][
                            internal_input_name
                        ] = default_uid
            else:
                self.default_uids[versioned_internal_name] = defaults
        for versioned_internal_name, df in other.calls.items():
            if versioned_internal_name in self.calls.keys():
                current_df = self.calls[versioned_internal_name]
                new_df = df[~df[Config.uid_col].isin(current_df[Config.uid_col])]
                self.calls[versioned_internal_name] = pd.concat(
                    [current_df, new_df], ignore_index=True
                )
            else:
                self.calls[versioned_internal_name] = df

    def compare_with_real(self, real_storage: Storage):
        # extract values
        values_df = real_storage.rel_adapter.get_vrefs()
        values = {
            row[Config.uid_col]: row[Config.vref_value_col]
            for _, row in values_df.iterrows()
        }
        sigs = real_storage.sig_adapter.load_ui_sigs()
        # extract calls and defaults
        all_call_data = real_storage.rel_adapter.get_all_call_data()
        calls = {}
        default_uids = {}
        for versioned_ui_name, df in all_call_data.items():
            sig = sigs[Signature.parse_versioned_name(versioned_ui_name)]
            versioned_internal_name = sig.versioned_internal_name
            calls[versioned_internal_name] = df.rename(
                columns=sig.ui_to_internal_input_map
            )
            default_uids[versioned_internal_name] = sig._new_input_defaults_uids
            sess.d = locals()
        return (
            values == self.values
            and all(
                compare_dfs_as_relations(calls[k], self.calls[k]) for k in calls.keys()
            )
            and default_uids == self.default_uids
        )


class ClientState:
    def __init__(self, root: RemoteStorage):
        self.storage = Storage(root=root)
        self.mock_storage = MockStorage()
        self.workflows: List[Workflow] = []
        self.func_ops: List[FuncOp] = []
        self.num_func_renames = 0
        self.num_input_renames = 0


class Preconditions:
    """
    A namespace for preconditions for the rules of the state machine.

    NOTE: Preconditions are defined as functions instead of lambdas to enable
    type introspection, autorefactoring, etc.
    """

    # control some of the transitions to avoid long chains, especially ones that
    # make the DB larger
    #! (does this actually optimize things? we need to benchmark)
    MAX_OPS_PER_CLIENT = 10
    MAX_INPUTS_PER_OP = 5
    MAX_WORKFLOWS_PER_CLIENT = 5
    MAX_WORKFLOW_SIZE_TO_ADD_VAR = 5
    MAX_WORKFLOW_SIZE_TO_ADD_OP = 10
    # prevent too many renames
    MAX_FUNC_RENAMES_PER_CLIENT = 20
    MAX_INPUT_RENAMES_PER_CLIENT = 20

    @staticmethod
    def create_op(instance: "SingleClientSimulator") -> Tuple[bool, List[ClientState]]:
        # return clients for which an op can be created
        candidates = [
            c
            for c in instance.clients
            if len(c.func_ops) < Preconditions.MAX_OPS_PER_CLIENT
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
            for idx, func_op in enumerate(c.func_ops):
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
            if c.num_func_renames < Preconditions.MAX_FUNC_RENAMES_PER_CLIENT:
                for idx, func_op in enumerate(c.func_ops):
                    candidates.append((c, func_op, idx))
        return len(candidates) > 0, candidates

    @staticmethod
    def rename_input(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, idx) for which an input can be renamed
        candidates = []
        for c in instance.clients:
            if c.num_input_renames < Preconditions.MAX_INPUT_RENAMES_PER_CLIENT:
                candidates += [(c, op, idx) for idx, op in enumerate(c.func_ops)]
        return len(candidates) > 0, candidates

    @staticmethod
    def create_new_version(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, FuncOp, int]]]:
        # return tuples of (client, op, idx of this op) for which a new version can be created
        candidates = []
        for c in instance.clients:
            num_ops = len(c.func_ops)
            if num_ops > 0 and num_ops < Preconditions.MAX_OPS_PER_CLIENT:
                candidates += [(c, op, idx) for idx, op in enumerate(c.func_ops)]
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
            if len(c.workflows) < Preconditions.MAX_WORKFLOWS_PER_CLIENT
        ]
        return len(candidates) > 0, candidates

    @staticmethod
    def add_input_var_to_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow]]]:
        # return tuples of (client, workflow) for which an input var can be added
        candidates = []
        for c in instance.clients:
            for w in c.workflows:
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
            for w in c.workflows:
                if (
                    w.shape_size < Preconditions.MAX_WORKFLOW_SIZE_TO_ADD_OP
                    and len(w.var_nodes) > 0
                ):
                    candidates += [(c, w, op) for op in c.func_ops]
        return len(candidates) > 0, candidates

    @staticmethod
    def add_call_to_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow, FuncQuery]]]:
        # return tuples of (client, workflow, op_node) for which a call can be added
        candidates = []
        for c in instance.clients:
            for w in c.workflows:
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
            for w in c.workflows:
                if w.num_calls > 0:
                    candidates.append((c, w))
        return len(candidates) > 0, candidates

    @staticmethod
    def check_mock_storage_single(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[ClientState]]:
        candidates = [c for c in instance.clients if len(c.workflows) > 0]
        return len(candidates) > 0, candidates

    @staticmethod
    def query_workflow(
        instance: "SingleClientSimulator",
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow]]]:
        # return tuples of (client, workflow) for which a workflow can be
        # queried
        candidates = []
        for c in instance.clients:
            for w in c.workflows:
                if len(w.var_nodes) > 0 and w.is_saturated:
                    candidates.append((c, w))
        return len(candidates) > 0, candidates


class SingleClientSimulator(RuleBasedStateMachine):
    def __init__(self, n_clients: int = 1):
        super().__init__()
        client = mongomock.MongoClient()
        root = MongoMockRemoteStorage(db_name="test", client=client)
        self.clients = [ClientState(root=root) for _ in range(n_clients)]
        # a central storage on which all operations are performed
        self.mock_storage = MockStorage()
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
            ui_name=random_string(),
            input_names=[random_string() for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=0,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client.func_ops.append(new_func_op)
        for mock_storage in [client.mock_storage, self.mock_storage]:
            mock_storage.create_op(func_op=new_func_op)
            mock_storage.check_invariants()

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
        new_name = random_string()
        default_value = 23
        new_sig = sig.create_input(name=new_name, default=default_value)
        # TODO: provide a new function with extra input as a user would
        new_func_op = FuncOp._from_data(f=f, sig=new_sig)
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client.func_ops[idx] = new_func_op
        new_sig = new_func_op.sig
        for mock_storage in [client.mock_storage, self.mock_storage]:
            mock_storage.add_input(
                func_op=new_func_op,
                internal_name=new_sig.ui_to_internal_input_map[new_name],
                default_value=default_value,
                default_uid=new_sig.new_ui_input_default_uids[new_name],
            )
            mock_storage.check_invariants()

    @precondition(lambda machine: Preconditions.rename_func(machine)[0])
    @rule()
    def rename_func(self):
        candidates = Preconditions.rename_func(self)[1]
        client, func_op, idx = random.choice(candidates)
        # idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[idx]
        new_name = random_string()
        # find and rename all versions
        all_versions = [
            (i, other_func_op)
            for i, other_func_op in enumerate(client.func_ops)
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
                rename_done = True
            else:
                # after the rename, get the true signature from the storage
                new_sig = client.storage.sig_adapter.load_state()[
                    func_op_version.sig.internal_name, func_op_version.sig.version
                ]
            # now update the state of the simulator
            new_func_op_version = FuncOp._from_data(f=func_op_version.func, sig=new_sig)
            synchronize_op(func_op=new_func_op_version, storage=client.storage)
            client.func_ops[version_idx] = new_func_op_version
        client.num_func_renames += 1

    @precondition(lambda machine: Preconditions.rename_input(machine)[0])
    @rule()
    def rename_input(self):
        candidates = Preconditions.rename_input(self)[1]
        client, func_op, func_idx = random.choice(candidates)
        # func_idx = random.randint(0, len(self._func_ops) - 1)
        # func_op = self._func_ops[func_idx]
        input_to_rename = random.choice(sorted(list(func_op.sig.input_names)))
        new_name = random_string()
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
        client.func_ops[func_idx] = new_func_op
        client.num_input_renames += 1

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
            input_names=[random_string() for _ in range(random.randint(1, 3))],
            n_outputs=random.randint(0, 3),
            defaults={},
            version=new_version,
            deterministic=True,
        )
        synchronize_op(func_op=new_func_op, storage=client.storage)
        client.func_ops.append(new_func_op)
        for mock_storage in [client.mock_storage, self.mock_storage]:
            mock_storage.create_new_version(new_version=new_func_op)
            mock_storage.check_invariants()

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
        client.workflows.append(res)

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

    def _execute_workflow(self, client: ClientState, workflow: Workflow) -> List[Call]:
        client.storage.sync_from_remote()
        client.mock_storage.sync_from_other(other=self.mock_storage)
        calls = SimpleWorkflowExecutor().execute(
            workflow=workflow, storage=client.storage
        )
        client.storage.commit(calls=calls)
        client.storage.sync_to_remote()
        for mock_storage in [client.mock_storage, self.mock_storage]:
            for call in calls:
                mock_storage.add_call(call=call)
            mock_storage.check_invariants()
        return calls

    @precondition(lambda machine: Preconditions.execute_workflow(machine)[0])
    @rule()
    def execute_workflow(self):
        candidates = Preconditions.execute_workflow(self)[1]
        client, workflow = random.choice(candidates)
        calls = self._execute_workflow(client=client, workflow=workflow)
        # client.storage.sync_from_remote()
        # calls = SimpleWorkflowExecutor().execute(
        #     workflow=workflow, storage=client.storage
        # )
        # client.storage.commit(calls=calls)
        # client.storage.sync_to_remote()

    ############################################################################
    ### multi-client rules
    ############################################################################
    @rule()
    def sync(self):
        client = random.choice(self.clients)
        client.storage.sync_with_remote()

    @precondition(lambda machine: Preconditions.check_mock_storage_single(machine)[0])
    @rule()
    def check_mock_storage_single(self):
        # run all workflows for a given client and check that the state equals
        # the state of the mock storage
        candidates = Preconditions.check_mock_storage_single(self)[1]
        client = random.choice(candidates)
        for workflow in client.workflows:
            self._execute_workflow(client=client, workflow=workflow)
        assert client.mock_storage.compare_with_real(client.storage)

    @invariant()
    def verify_state(self):
        for client in self.clients:
            # make sure that functions called on the storage work
            client.storage.rel_adapter.get_all_call_data()
            # check storage invariants
            check_invariants(storage=client.storage)
            # check invariants on the workflows
            for w in client.workflows:
                w.check_invariants()
            for func_op in client.func_ops:
                func_op.sig.check_invariants()

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


class MultiClientSimulator(SingleClientSimulator):
    def __init__(self, n_clients: int = 3):
        super().__init__(n_clients=n_clients)


TestCaseSingle = SingleClientSimulator.TestCase
TestCaseSingle.settings = settings(
    max_examples=100, deadline=None, stateful_step_count=50
)

TestCaseMany = MultiClientSimulator.TestCase
TestCaseMany.settings = settings(
    max_examples=100, deadline=None, stateful_step_count=50
)
