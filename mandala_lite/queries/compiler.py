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


class Compiler:
    def __init__(self, val_queries: List[ValQuery], func_queries: List[FuncQuery]):
        self.val_queries = val_queries
        self.func_queries = func_queries
        # self._generate_aliases()
        self.val_aliases, self.func_aliases = self._generate_aliases()

    def _generate_aliases(self) -> Tuple[Dict[ValQuery, Table], Dict[FuncQuery, Table]]:
        func_aliases = {}
        for func_query in self.func_queries:
            op_table = Table(func_query.op.sig.versioned_ui_name)
            func_aliases[func_query] = op_table.as_(f"_{id(func_query)}")
        val_aliases = {}
        for val_query in self.val_queries:
            val_table = Table(Config.vref_table)
            val_aliases[val_query] = val_table.as_(f"_{id(val_query)}")
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
