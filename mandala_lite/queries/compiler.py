from ..common_imports import *
from ..core.config import Config
from .weaver import ValQuery, FuncQuery
from pypika import Query, Table, Field, Column, Criterion


def concat_lists(lists: List[list]) -> list:
    return [x for lst in lists for x in lst]


def traverse_all(val_queries: List[ValQuery]) -> Tuple[List[ValQuery], List[FuncQuery]]:
    """
    Extend the given `ValQuery` objects to all objects connected to them through
    function inputs/outputs.
    """
    val_queries_ = [_ for _ in val_queries]
    op_queries_: List[FuncQuery] = []
    found_new = True
    while found_new:
        found_new = False
        val_neighbors = concat_lists([v.neighbors() for v in val_queries_])
        op_neighbors = concat_lists([o.neighbors() for o in op_queries_])
        if any(k not in op_queries_ for k in val_neighbors):
            found_new = True
            for neigh in val_neighbors:
                if neigh not in op_queries_:
                    op_queries_.append(neigh)
        if any(k not in val_queries_ for k in op_neighbors):
            found_new = True
            for neigh in op_neighbors:
                if neigh not in val_queries_:
                    val_queries_.append(neigh)
    return val_queries_, op_queries_


def solve_query(
    data: Dict[str, pd.DataFrame],
    selection: List[ValQuery],
    val_queries: List[ValQuery],
    op_queries: List[FuncQuery],
) -> pd.DataFrame:
    """
    Given the relational storage (i.e., a dictionary of {internal function name:
    memoization table}, solve the conjunctive query imposed by the given query
    objects.

    Algorithm
    =========

    Suppose we have func queries F_1, ..., F_m that are connected
    to val queries V_1, ..., V_n in a bipartite graph, where each edge is
    labeled
        F_i ---name_ij--> V_j
    by the name of the input/output of the function corresponding to this value
    query. Also let S_1, ..., S_k be the queries in the SELECT clause of the
    query if you will.

    For example, with the following code:
    ```python
    with query(storage) as q:
        i = Query()
        j = inc(x=i)
        final = add(x=i, y=j)
        q.get_table(i, final)
    ```
    the graph would have edges
    inc ---x--> i
    inc ---output_0--> j
    add ---x--> i
    add ---y--> j
    add ---output_0--> final

    with S_1 = i, S_2 = final.

    Then you can iteratively shrink this graph by
        - picking two function nodes F_i, F_j,
        - joining their tables along the shared edges in the obvious manner,
        - replacing them with a single node corresponding to the new table, with
          edges to the union of their columns.

    Since joins are associative, the order in which you do this does not change
    the result.

    Finally, you return the restriction of this table to the columns S_i that
    are in the given selection.

    Optimizations
    =============
        - One obvious one is to prune away columns that won't matter
    """
    if len(op_queries) == 1:
        op_query = op_queries[0]
        df = data[op_query.op.sig.internal_name]
        column_names = []
        for select_query in selection:
            for k, v in op_query.inputs.items():
                if v is select_query:
                    column_names.append(k)
            for i, output in enumerate(op_query.outputs):
                if output is select_query:
                    column_names.append(f"output_{i}")
        return df[column_names]
    else:
        raise NotImplementedError()


class Compiler:
    def __init__(self, val_queries: List[ValQuery], func_queries: List[FuncQuery]):
        self.val_queries = val_queries
        self.func_queries = func_queries
        # self._generate_aliases()
        self.val_aliases, self.func_aliases = self._generate_aliases()

    def _generate_aliases(self) -> Tuple[Dict[ValQuery, Table], Dict[FuncQuery, Table]]:
        func_aliases = {}
        for func_query in self.func_queries:
            op_table = Table(func_query.op.sig.internal_name)
            if Config.debug:
                name = func_query.op.sig.external_name
            else:
                name = f"_{id(func_query)}"
            func_aliases[func_query] = op_table.as_(name)
        val_aliases = {}
        for val_query in self.val_queries:
            val_table = Table(Config.vref_table)
            if Config.debug:
                assert val_query.column_name is not None
                name = val_query.column_name
            else:
                name = f"_{id(val_query)}"
            val_aliases[val_query] = val_table.as_(name)
        return val_aliases, func_aliases

    def compile_func(self, op_query: FuncQuery) -> Tuple[list, list]:
        """
        Compile the query corresponding to an op, including built-in ops
        """
        constraints = []
        select_fields = []
        func_alias = self.func_aliases[op_query]
        for input_name, val_query in op_query.inputs.items():
            val_alias = self.val_aliases[val_query]
            constraints.append(val_alias[Config.uid_col] == func_alias[input_name])
            select_fields.append(val_alias[Config.uid_col])
        for output_idx, val_query in enumerate(op_query.outputs):
            val_alias = self.val_aliases[val_query]
            constraints.append(
                val_alias[Config.uid_col] == func_alias[f"output_{output_idx}"]
            )
            select_fields.append(val_alias[Config.uid_col])
        return constraints, select_fields

    def compile(self, select_queries: List[ValQuery]):
        """
        Compile the query induced by the data of this compiler instance to
        an SQL select query.

        NOTE:
            - for each value query, we select both columns of the variable
            table: the index and the partition. This is to be able to convert
            the query result directly into locations.
            - The list of columns, partitioned into sublists per value query, is
            also returned.
        """
        # if select_queries is None:
        #     select_queries = tuple(self.val_queries)
        # assert all([vq in self.val_queries for vq in select_queries])
        from_tables = []
        all_constraints = []
        select_cols = [self.val_aliases[vq][Config.uid_col] for vq in select_queries]
        for func_query in self.func_queries:
            constraints, select_fields = self.compile_func(func_query)
            func_alias = self.func_aliases[func_query]
            from_tables.append(func_alias)
            all_constraints += constraints
        for val_query in self.val_queries:
            val_alias = self.val_aliases[val_query]
            from_tables.append(val_alias)
        query = Query
        for table in from_tables:
            query = query.from_(table)
        query = query.select(*select_cols)
        query = query.where(Criterion.all(all_constraints))
        print(query)
        return query
