from ..common_imports import *
from ..core.model import FuncOp
from ..core.sig import Signature
from ..core.config import Config, dump_output_name
from .weaver import ValNode, CallNode, traverse_all
from .viz import visualize_graph, get_names
from ..core.utils import invert_dict, OpKey
from pypika import Query, Table, Criterion


class Compiler:
    def __init__(self, vqs: List[ValNode], fqs: List[CallNode]):
        if not len(vqs) == len({id(x) for x in vqs}):
            raise InternalError
        if not len(fqs) == len({id(x) for x in fqs}):
            raise InternalError
        self.vqs = vqs
        self.fqs = fqs
        self.val_aliases, self.func_aliases = self._generate_aliases()

    def _generate_aliases(self) -> Tuple[Dict[ValNode, Table], Dict[CallNode, Table]]:
        func_aliases = {}
        func_counter = 0
        for i, func_query in enumerate(self.fqs):
            op_table = Table(func_query.func_op.sig.versioned_ui_name)
            func_aliases[func_query] = op_table.as_(f"_func_{func_counter}")
            func_counter += 1
        val_aliases = {}
        val_counter = 0
        for val_query in self.vqs:
            val_table = Table(Config.causal_vref_table)
            val_aliases[val_query] = val_table.as_(f"_var_{val_counter}")
            val_counter += 1
        return val_aliases, func_aliases

    def compile_func(
        self, fq: CallNode, semantic_versions: Optional[Set[str]] = None
    ) -> Tuple[list, list]:
        """
        Compile the query corresponding to an op, including built-in ops
        """
        constraints = []
        select_fields = []
        func_alias = self.func_aliases[fq]
        for input_name, val_query in fq.inputs.items():
            val_alias = self.val_aliases[val_query]
            constraints.append(val_alias[Config.full_uid_col] == func_alias[input_name])
        for output_name, val_query in fq.outputs.items():
            val_alias = self.val_aliases[val_query]
            constraints.append(
                val_alias[Config.full_uid_col] == func_alias[output_name]
            )
        if semantic_versions is not None:
            constraints.append(
                func_alias[Config.semantic_version_col].isin(semantic_versions)
            )
        if fq.constraint is not None:
            constraints.append(func_alias[Config.causal_uid_col].isin(fq.constraint))
        return constraints, select_fields

    def compile_val(self, val_query: ValNode) -> Tuple[list, list]:
        """
        Compile the query corresponding to a variable
        """
        constraints = []
        select_fields = []
        val_alias = self.val_aliases[val_query]
        if val_query.constraint is not None:
            constraints.append(
                val_alias[Config.full_uid_col].isin(val_query.constraint)
            )
        select_fields.append(val_alias[Config.full_uid_col])
        return constraints, select_fields

    def compile(
        self,
        select_queries: List[ValNode],
        semantic_version_constraints: Optional[Dict[OpKey, Optional[Set[str]]]] = None,
        filter_duplicates: bool = False,
    ):
        """
        Compile the query induced by the data of this compiler instance to
        an SQL select query. If `semantic_version_constraints` is not provided,
        no constraints are placed.

        NOTE:
            - for each value query, we select both columns of the variable
            table: the index and the partition. This is to be able to convert
            the query result directly into locations.
            - The list of columns, partitioned into sublists per value query, is
            also returned.
        """
        if not len(select_queries) == len({id(x) for x in select_queries}):
            raise InternalError
        assert all([vq in self.vqs for vq in select_queries])
        from_tables = []
        all_constraints = []
        select_cols = [
            self.val_aliases[vq][Config.full_uid_col] for vq in select_queries
        ]
        if semantic_version_constraints is None:
            semantic_version_constraints = {
                (op_query.func_op.sig.internal_name, op_query.func_op.sig.version): None
                for op_query in self.fqs
            }
        for func_query in self.fqs:
            op_key = (
                func_query.func_op.sig.internal_name,
                func_query.func_op.sig.version,
            )
            constraints, select_fields = self.compile_func(
                func_query, semantic_versions=semantic_version_constraints[op_key]
            )
            func_alias = self.func_aliases[func_query]
            from_tables.append(func_alias)
            all_constraints.extend(constraints)
        for val_query in self.vqs:
            val_alias = self.val_aliases[val_query]
            constraints, select_fields = self.compile_val(val_query)
            from_tables.append(val_alias)
            all_constraints.extend(constraints)
        query = Query
        for table in from_tables:
            query = query.from_(table)
        query = query.select(*select_cols)
        if filter_duplicates:
            query = query.distinct()
        query = query.where(Criterion.all(all_constraints))
        return query
