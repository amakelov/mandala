class Something(RuleBasedStateMachine):
    things_bundle = Bundle("things")

    def __init__(self):
        super().__init__()
        self.things = []

    @rule(target=things_bundle, size=st.integers(0, 2))
    def make_thing(self, size: int) -> list:
        print(size)
        thing = [get_uid() for _ in range(size)]
        self.things.append(thing)
        return thing

    @rule()
    def do_something_with_thing(self):
        for thing in self.things:
            if len(thing) > 1:
                thing.append("something")

    @rule()
    def break_everything(self):
        assert all(len(thing) < 3 for thing in self.things)


class SingleClientSimulator(RuleBasedStateMachine):
    workflows = Bundle(name="workflows")
    ops = Bundle(name="ops")

    def __init__(self):
        super().__init__()
        self.storage = Storage()
        self._workflows: List[Workflow] = []
        self._ops: List[FuncOp] = []

    ############################################################################
    ### schema modifications
    ############################################################################
    @precondition(Preconditions.not_too_many_ops)
    @rule(target=ops)
    def create_op(self) -> FuncOp:
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
        return res

    ############################################################################
    ### generating workflows
    ############################################################################
    @precondition(Preconditions.not_too_many_workflows)
    @rule(target=workflows)
    def add_workflow(self) -> Workflow:
        res = Workflow()
        self._workflows.append(res)
        return res

    # @precondition(Preconditions.workflows_with_few_ops_exist)
    @precondition(Preconditions.vars_exist)
    @rule(workflow=workflows.filter(Filters.workflow_has_vars), op=ops)
    def add_op_to_workflow(self, op: FuncOp, workflow: Workflow):
        # pick inputs randomly from workflow
        inputs = {
            name: random.choice(workflow.var_nodes) for name in op.sig.input_names
        }
        # add function over these inputs
        op_node, output_nodes = workflow.add_op(inputs=inputs, op=op)

    @precondition(Preconditions.workflows_exist)
    @precondition(Preconditions.small_workflows_exist)
    @rule(workflow=workflows)
    def add_input_var_to_workflow(self, workflow: Workflow):
        # always add a value to make sampling proceed faster
        var = workflow.add_var()
        workflow.add_value(value=wrap(get_uid()), var=var)

    @precondition(Preconditions.callable_ops_exist)
    @rule(workflow=workflows)
    def add_call_to_workflow(self, workflow: Workflow):
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
                [Delayed() for _ in range(op_node.op.sig.n_outputs)],
            ]
        )
        workflow.add_call_struct(call_struct=call_struct)

    # @precondition(Preconditions.op)
    # @rule(workflow=workflows)
    # def execute_workflow(self, workflow):
    #     pass

    # @precondition(Preconditions.workflows_saturated)
    # @rule(workflow=workflows)
    # def query_workflow(self, workflow:Workflow):
    #     val_queries, op_queries = workflow.var_nodes, workflow.op_nodes

    ############################################################################
    @staticmethod
    def not_too_many_workflows(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) < 3

    @staticmethod
    def not_too_many_ops(instance: "SingleClientSimulator") -> bool:
        return len(instance._ops) < 3

    @staticmethod
    def workflows_exist(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) > 0

    @staticmethod
    def workflows_saturated(instance: "SingleClientSimulator") -> bool:
        return all(wf.is_saturated for wf in instance._workflows)

    @staticmethod
    def delayed_values_exist(instance: "SingleClientSimulator") -> bool:
        pass

    @staticmethod
    def small_workflows_exist(instance: "SingleClientSimulator") -> bool:
        return any([w.shape_size < 5 for w in instance._workflows])

    @staticmethod
    def workflows_with_few_ops_exist(instance: "SingleClientSimulator") -> bool:
        return any([len(w.op_nodes) < 5 for w in instance._workflows])

    @staticmethod
    def vars_exist(instance: "SingleClientSimulator") -> bool:
        return any(len(wf.var_nodes) > 0 for wf in instance._workflows)

    @staticmethod
    def callable_ops_exist(instance: "SingleClientSimulator") -> bool:
        return len(instance._workflows) > 0 and all(
            len(wf.callable_op_nodes) > 0 for wf in instance._workflows
        )
