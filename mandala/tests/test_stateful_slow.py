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

from mandala.common_imports import *
from mandala.all import *
from mandala.tests.utils import *
from mandala.tests.stateful_utils import *
from mandala.queries.workflow import Workflow, CallStruct
from mandala.core.utils import Hashing, get_uid, parse_full_uid
from mandala.queries.compiler import *
from mandala.core.model import Type, ListType
from mandala.core.builtins_ import Builtins
from mandala.core.sig import _get_return_annotations
from mandala.storages.remote_storage import RemoteStorage
from mandala.ui.executors import SimpleWorkflowExecutor
from mandala.ui.funcs import FuncInterface
from mandala.ui.storage import make_delayed


class MockStorage:
    """
    A simple storage simulator that
    - stores all data in memory: calls as tables, vrefs as a dictionary
    - only uses internal names for signatures;
    - can be synced in a "naive" way with another storage: the state of the
    other storage is upserted into this storage.
    - note: currently, it replicates only the memoization tables and the table
    of values (not the causal UIDs table)

    It is invariant-checked at the entry and exit of every method. The
    state of this object should only be changed through these methods. This
    makes it easy to track down when an inconsistent update happened.
    """

    def __init__(self):
        self.calls: Dict[str, pd.DataFrame] = {}
        self.values: Dict[str, Any] = {}
        # versioned internal op name -> (internal input name -> default uid)
        self.default_uids: Dict[str, Dict[str, str]] = {}
        for builtin_op in Builtins.OPS.values():
            name = builtin_op.sig.versioned_internal_name
            self.calls[name] = pd.DataFrame(
                columns=list(builtin_op.sig.input_names) + Config.special_call_cols
            )
            self.default_uids[name] = {}
        self.check_invariants()

    def __eq__(self, other: Any):
        if not isinstance(other, MockStorage):
            return False
        values_equal = self.values == other.values
        default_uids_equal = self.default_uids == other.default_uids
        calls_equal = self.calls.keys() == other.calls.keys() and all(
            compare_dfs_as_relations(self.calls[k], other.calls[k])
            for k in self.calls.keys()
        )
        return values_equal and default_uids_equal and calls_equal

    def check_invariants(self):
        assert self.default_uids.keys() == self.calls.keys()
        # get all vref uids that appear in calls
        vref_uids_from_calls = []
        for k, df in self.calls.items():
            for col in df.columns:
                if col not in Config.special_call_cols:
                    vref_uids_from_calls += df[col].values.tolist()
        vref_uids_from_calls = [parse_full_uid(x)[0] for x in vref_uids_from_calls]
        assert set(vref_uids_from_calls) <= set(self.values.keys())
        for versioned_internal_name, defaults in self.default_uids.items():
            df = self.calls[versioned_internal_name]
            for internal_input_name, default_uid in defaults.items():
                assert internal_input_name in df.columns

    def create_op(self, func_op: FuncOp):
        self.check_invariants()
        sig = func_op.sig
        if sig.versioned_internal_name in self.calls.keys():
            raise ValueError()
        if sig.versioned_internal_name in self.default_uids:
            raise ValueError()
        self.calls[sig.versioned_internal_name] = pd.DataFrame(
            columns=Config.special_call_cols
            + list(sig.ui_to_internal_input_map.values())
            + [dump_output_name(index=i) for i in range(sig.n_outputs)]
        )
        self.default_uids[sig.versioned_internal_name] = {}
        self.check_invariants()

    def add_input(
        self,
        func_op: FuncOp,
        internal_name: str,
        default_value: Any,
        default_full_uid: str,
    ):
        self.check_invariants()
        default_uid, default_causal_uid = parse_full_uid(default_full_uid)
        sig = func_op.sig
        df = self.calls[sig.versioned_internal_name]
        df[internal_name] = [default_full_uid for _ in range(len(df))]
        self.values[default_uid] = default_value
        self.default_uids[sig.versioned_internal_name][internal_name] = default_full_uid
        self.check_invariants()

    def rename_func(self, func_op: FuncOp, new_name: str):
        pass

    def rename_input(self, func_op: FuncOp, old_name: str, new_name: str):
        pass

    def create_new_version(self, new_version: FuncOp):
        self.check_invariants()
        self.create_op(func_op=new_version)
        self.check_invariants()

    def add_call(self, call: Call):
        self.check_invariants()
        func_op, inputs, outputs = call.func_op, call.inputs, call.outputs
        sig = func_op.sig
        row = {
            Config.uid_col: call.uid,
            Config.causal_uid_col: call.causal_uid,
            Config.content_version_col: call.content_version,
            Config.semantic_version_col: call.semantic_version,
            Config.transient_col: call.transient,
            **{sig.ui_to_internal_input_map[k]: v.full_uid for k, v in inputs.items()},
            **{dump_output_name(index=i): v.full_uid for i, v in enumerate(outputs)},
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
        self.check_invariants()

    def sync_from_other(self, other: "MockStorage"):
        """
        Update this storage with the new data from another storage.

        NOTE: always copy the other storage's state into this storage; never use
        shared objects.
        """
        self.check_invariants()
        # update values
        for k, v in other.values.items():
            if k in self.values.keys():
                assert v == self.values[k]
            # avoid shared objects
            self.values[k] = copy.deepcopy(v)
        # update defaults
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
                # use a copy to avoid a shared mutable object
                self.default_uids[versioned_internal_name] = copy.deepcopy(defaults)
        # update calls
        for versioned_internal_name, df in other.calls.items():
            if versioned_internal_name in self.calls.keys():
                current_df = self.calls[versioned_internal_name]
                new_df = df[~df[Config.uid_col].isin(current_df[Config.uid_col])].copy()
                self.calls[versioned_internal_name] = pd.concat(
                    [current_df, new_df], ignore_index=True
                )
            else:
                self.calls[versioned_internal_name] = df.copy()
        self.check_invariants()

    def compare_with_real(self, real_storage: Storage):
        self.check_invariants()
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
        sess.d()
        return (
            values == self.values
            and all(
                compare_dfs_as_relations(calls[k], self.calls[k]) for k in calls.keys()
            )
            and default_uids == self.default_uids
        )


class ClientState:
    def __init__(self, root: Optional[RemoteStorage]):
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
    def add_workflow(
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
    ) -> Tuple[bool, List[Tuple[ClientState, Workflow, CallNode]]]:
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
    def sync_all(
        instance: "SingleClientSimulator",
    ) -> bool:
        # require at least some calls to be executed
        for client in instance.clients:
            if any(
                len(df) > 0
                for df in client.storage.rel_adapter.get_all_call_data().values()
            ):
                return True
        return False

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
        # client = mongomock.MongoClient()
        # root = MongoMockRemoteStorage(db_name="test", client=client)
        root = None
        self.clients = [ClientState(root=root) for _ in range(n_clients)]
        # a central storage on which all operations are performed
        self.mock_storage = MockStorage()
        #! keep everything deterministic. only use `random` for generating
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
        client.storage.synchronize_op(func_op=new_func_op)
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
        new_sig = sig.create_input(name=new_name, default=default_value, annotation=Any)
        # TODO: provide a new function with extra input as a user would
        new_func_op = FuncOp._from_data(func=make_func_from_sig(new_sig), sig=new_sig)
        client.storage.synchronize_op(func_op=new_func_op)
        client.func_ops[idx] = new_func_op
        new_sig = new_func_op.sig
        for mock_storage in [client.mock_storage, self.mock_storage]:
            mock_storage.add_input(
                func_op=new_func_op,
                internal_name=new_sig.ui_to_internal_input_map[new_name],
                default_value=default_value,
                default_full_uid=new_sig.new_ui_input_default_uids[new_name],
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
                client.storage.synchronize(f=func_interface)
                new_sig = client.storage.rename_func(
                    func=func_interface, new_name=new_name
                )
                rename_done = True
            else:
                # after the rename, get the true signature from the storage
                new_sig = client.storage.sig_adapter.load_state()[
                    func_op_version.sig.internal_name, func_op_version.sig.version
                ]
            # now update the state of the simulator
            new_func_op_version = FuncOp._from_data(
                func=func_op_version.func, sig=new_sig
            )
            client.storage.synchronize_op(func_op=new_func_op_version)
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
        client.storage.synchronize(f=func_interface)
        new_sig = client.storage.rename_arg(
            func=func_interface,
            name=input_to_rename,
            new_name=new_name,
        )
        # now update the state of the simulator
        new_func = make_func_from_sig(sig=new_sig)
        new_func_op = FuncOp._from_data(func=new_func, sig=new_sig)
        client.storage.synchronize_op(func_op=new_func_op)
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
        client.storage.synchronize_op(func_op=new_func_op)
        client.func_ops.append(new_func_op)
        for mock_storage in [client.mock_storage, self.mock_storage]:
            mock_storage.create_new_version(new_version=new_func_op)
            mock_storage.check_invariants()

    ############################################################################
    ### generating workflows
    ############################################################################
    @precondition(lambda machine: Preconditions.add_workflow(machine)[0])
    @rule()
    def add_workflow(self):
        """
        Add a new (empty) workflow to the test.
        """
        candidates = Preconditions.add_workflow(self)[1]
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
        workflow.add_value(value=wrap_atom(get_uid()), var=var)

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
        _, _ = workflow.add_op(inputs=inputs, func_op=func_op)

    @precondition(lambda machine: Preconditions.add_call_to_workflow(machine)[0])
    @rule()
    def add_call_to_workflow(self):
        candidates = Preconditions.add_call_to_workflow(self)[1]
        client, workflow, op_node = random.choice(candidates)
        input_vars = op_node.inputs
        # pick random values
        var_to_values = workflow.var_to_values()
        input_values = {
            name: random.choice(var_to_values[var]) for name, var in input_vars.items()
        }
        func_op = op_node.func_op
        output_types = [Type.from_annotation(a) for a in func_op.output_annotations]
        outputs = [make_delayed(tp=tp) for tp in output_types]
        call_struct = CallStruct(
            func_op=op_node.func_op, inputs=input_values, outputs=outputs
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
    def sync_one(self):
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

    @precondition(lambda machine: Preconditions.sync_all(machine))
    @rule()
    def sync_all(self):
        for client in self.clients:
            client.storage.sync_with_remote()
            client.mock_storage.sync_from_other(other=self.mock_storage)
        for client in self.clients:
            # have to do it again!
            client.storage.sync_with_remote()
            client.mock_storage.sync_from_other(other=self.mock_storage)
        for client in self.clients:
            assert client.mock_storage == self.mock_storage
            assert self.mock_storage.compare_with_real(real_storage=client.storage)

    @invariant()
    def verify_state(self):
        for client in self.clients:
            # make sure that functions called on the storage work
            client.storage.rel_adapter.get_all_call_data()
            for table in client.storage.rel_storage.get_tables():
                client.storage.rel_storage.get_count(table=table)
            # check storage invariants
            check_invariants(storage=client.storage)
            # check invariants on the workflows
            for w in client.workflows:
                w.check_invariants()
                # w.print_shape()
            # check the individual signatures
            for func_op in client.func_ops:
                func_op.sig.check_invariants()
            # check the set of signatures
            client.storage.sig_adapter.check_invariants()

    def check_sig_synchronization(self):
        for client in self.clients:
            assert client.storage.root.sigs == client.storage.sig_adapter.load_state()

    # @precondition(lambda machine: Preconditions.query_workflow(machine)[0])
    # @rule()
    # def query_workflow(self):
    #     candidates = Preconditions.query_workflow(self)[1]
    #     client, workflow = random.choice(candidates)
    #     # workflow = random.choice([w for w in client.workflows if w.is_saturated])
    #     # path = Path(__file__).parent / f"bug.cloudpickle"
    #     # op_nodes = copy.deepcopy(workflow.op_nodes)
    #     # db_dump_path = Path(__file__).parent.absolute() / 'db_dump/'
    #     # self.storage.rel_storage.execute_no_results(query=f"EXPORT DATABASE '{db_dump_path}';")
    #     # data = (self._ops)
    #     # with open(path, "wb") as f:
    #     #     cloudpickle.dump(data, f)
    #     val_queries, op_queries = workflow.var_nodes, workflow.op_nodes
    #     # workflow.print_shape()
    #     df = client.storage.execute_query(select_queries=val_queries, engine='naive')


class MultiClientSimulator(SingleClientSimulator):
    def __init__(self, n_clients: int = 3):
        super().__init__(n_clients=n_clients)


MAX_EXAMPLES = 100
STEP_COUNT = 25

TestCaseSingle = SingleClientSimulator.TestCase
TestCaseSingle.settings = settings(
    max_examples=MAX_EXAMPLES, deadline=None, stateful_step_count=STEP_COUNT
)

TestCaseMany = MultiClientSimulator.TestCase
TestCaseMany.settings = settings(
    max_examples=MAX_EXAMPLES, deadline=None, stateful_step_count=STEP_COUNT
)
