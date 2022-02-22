from abc import ABC, abstractmethod
from sqlalchemy import Column, sql
from sqlalchemy.sql.expression import Select, BooleanClauseList, CompoundSelect
from sqlalchemy.sql import alias, Alias

from .rel_weaver import (
    OpQuery, ValQuery, FuncQuery, ConstructListQuery, 
    ConstructDictQuery, DeconstructListQuery
)
from .rel_weaver import ListQuery, DictQuery

from ..common_imports import *
from ..util.common_ut import concat_lists
from ..core.config import CoreConsts
from ..storages.rel_impl.utils import get_in_clause_rhs
from ..adapters.rels import BaseRelAdapter
from ..adapters.vals import BaseValAdapter


UID_COL = CoreConsts.UID_COL
PARTITION_COL = CoreConsts.PARTITION_COL
ListConsts = CoreConsts.List
DictConsts = CoreConsts.Dict


class BasePreSelect(ABC): # todo - remove
    """
    An interface to specify the elements of a select-from-where query
    """
    @abstractmethod
    def select_cols(self) -> TList[Column]:
        raise NotImplementedError()
    
    @abstractmethod
    def constraint(self) -> BooleanClauseList:
        raise NotImplementedError()


class BaseCompiler(ABC):
    """
    Compiles a web of query objects to SQL.
    
    This object needs interfaces to both relations (to get the tables
    corresponding to query elements) and to values (to initiate queries for
    evaluating predicates and ranges) 
    """
    @property
    @abstractmethod
    def rel_adapter(self) -> BaseRelAdapter:
        raise NotImplementedError()

    @property
    @abstractmethod
    def val_adapter(self) -> BaseValAdapter:
        raise NotImplementedError()

    @property
    @abstractmethod
    def op_queries(self) -> TList[OpQuery]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def val_queries(self) -> TList[ValQuery]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def op_aliases(self) -> TDict[OpQuery, Alias]:
        raise NotImplementedError() 
    
    @property
    @abstractmethod
    def val_aliases(self) -> TDict[ValQuery, Alias]:
        raise NotImplementedError()

    @abstractmethod
    def compile_op(self, op_query:OpQuery) -> BasePreSelect:
        raise NotImplementedError()
    
    @abstractmethod
    def compile_val(self, val_query:ValQuery) -> BasePreSelect:
        raise NotImplementedError()
    
    @abstractmethod
    def compile_calls(self,
                      select_queries:TTuple[OpQuery,...]=None,
                      ) -> CompoundSelect:
        raise NotImplementedError()
    
    @abstractmethod
    def get_calls(self, select_queries:TTuple[OpQuery,...]=None) -> TList[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def compile(self, 
                select_queries:TTuple[ValQuery,...],
                ) -> TTuple[Select, TList[list]]:
        raise NotImplementedError()

    @abstractmethod    
    def get_locations(self, 
                      select_queries:TTuple[ValQuery,...]) -> pd.DataFrame:
        raise NotImplementedError()

################################################################################
### implementation
################################################################################
class PreSelect(BasePreSelect):
    
    def __init__(self, select_cols:TList[Column]=None,
                 constraint:BooleanClauseList=None):
        self._select_clause = [] if select_cols is None else select_cols
        self._constraint = sql.and_(True) if constraint is None else constraint
    
    def select_cols(self) -> TList[Column]:
        return self._select_clause
    
    def constraint(self) -> BooleanClauseList:
        return self._constraint
    
    
class Compiler(BaseCompiler):

    def __init__(self, rels_adapter:BaseRelAdapter, 
                 vals_adapter:BaseValAdapter,
                 val_queries:TTuple[ValQuery,...], 
                 op_queries:TTuple[OpQuery,...], 
                 verbose:bool=False):
        self._rels_adapter = rels_adapter
        self._vals_adapter = vals_adapter
        self._val_queries = val_queries
        self._op_queries = op_queries
        self._op_aliases = {}
        self._val_aliases = {}
        self._verbose = verbose
        self._generate_aliases()
    
    @property
    def rel_adapter(self) -> BaseRelAdapter:
        return self._rels_adapter
    
    @property
    def val_adapter(self) -> BaseValAdapter:
        return self._vals_adapter
    
    def _generate_aliases(self):
        """
        Attach dictionaries of 
            {ValQuery: query's table alias}
            {OpQuery: query's table alias}
        to this instance for the instance's value and op queries
        """
        adpt = self.rel_adapter
        op_aliases = {}
        for op_query in self.op_queries:
            op_relname = adpt.get_op_relname(op=op_query.op)
            op_tableobj = adpt.get_tableobj(name=op_relname)
            op_aliases[op_query] = alias(op_tableobj, name=op_query.sql_name)
        self._op_aliases = op_aliases
        val_aliases = {}
        for val_query in self.val_queries:
            val_relname = adpt.get_type_relname(tp=val_query.tp)
            val_tableobj = adpt.get_tableobj(name=val_relname)
            val_aliases[val_query] = alias(val_tableobj, name=val_query.sql_name)
        self._val_aliases = val_aliases
    
    @property
    def op_queries(self) -> TList[OpQuery]:
        return self._op_queries
    
    @property
    def val_queries(self) -> TList[ValQuery]:
        return self._val_queries
    
    @property
    def op_aliases(self) -> TDict[OpQuery, Alias]:
        return self._op_aliases
    
    @property
    def val_aliases(self) -> TDict[ValQuery, Alias]:
        return self._val_aliases
    
    def _compile_list_bidirectional(self, 
                                    op_query:TUnion[ConstructListQuery, 
                                                    DeconstructListQuery],
                                    elt_query:ValQuery, list_query:ListQuery):
        constraints = []
        elt_tableobj = self.val_aliases[elt_query]
        list_tableobj = self.val_aliases[list_query]
        # deal with foreign key constraints
        constraints.append(
            self.op_aliases[op_query].c[ListConsts.ELT] ==
            elt_tableobj.c[UID_COL]
        )
        constraints.append(
            self.op_aliases[op_query].c[ListConsts.LIST] ==
            list_tableobj.c[UID_COL]
        )
        # deal with constraint over index
        if op_query.idx_constraint is not None:
            idx_constraint = op_query.idx_constraint
            if isinstance(idx_constraint, int):
                idx_range = [str(idx_constraint)]
            elif isinstance(idx_constraint, list):
                idx_range = [str(x) for x in idx_constraint]
            else:
                raise NotImplementedError()
            constraints.append(
                self.op_aliases[op_query].c[ListConsts.IDX].
                in_(get_in_clause_rhs(idx_range))
            )
        select_cols = [
            elt_tableobj.c[UID_COL], 
            list_tableobj.c[UID_COL]
        ]
        constraint = sql.and_(*constraints)
        return PreSelect(select_cols=select_cols, constraint=constraint)
    
    def _compile_construct_list(self, cl_query:ConstructListQuery):
        elt_query = cl_query.inputs[ListConsts.ELT]
        list_query = cl_query.outputs[ListConsts.LIST]
        return self._compile_list_bidirectional(op_query=cl_query, 
                                                elt_query=elt_query, 
                                                list_query=list_query)
    
    def _compile_deconstruct_list(self, dl_query:DeconstructListQuery):
        elt_query = dl_query.outputs[ListConsts.ELT]
        list_query = dl_query.inputs[ListConsts.LIST]
        return self._compile_list_bidirectional(op_query=dl_query, 
                                                elt_query=elt_query, 
                                                list_query=list_query)
    
    def _compile_construct_dict(self, cd_query:ConstructDictQuery):
        constraints = []
        # explicit construction
        value_query = cd_query.inputs[DictConsts.VALUE]
        dict_query = cd_query.outputs[DictConsts.DICT]
        value_tableobj = self.val_aliases[value_query]
        dict_tableobj = self.val_aliases[dict_query]
        # deal with foreign key constraints
        constraints.append(
            self.op_aliases[cd_query].c[DictConsts.VALUE] ==
            value_tableobj.c[UID_COL]
        )
        constraints.append(
            self.op_aliases[cd_query].c[DictConsts.DICT] ==
            dict_tableobj.c[UID_COL]
        )
        # deal with constraint over index
        if cd_query.key_constraint is not None:
            key_constraint = cd_query.key_constraint
            if isinstance(key_constraint, str):
                key_range = [key_constraint]
            elif isinstance(key_constraint, list):
                key_range = key_constraint
            else:
                raise NotImplementedError()
            constraints.append(
                self.op_aliases[cd_query].c[DictConsts.KEY].
                in_(get_in_clause_rhs(key_range))
            )
        select_cols = [
            value_tableobj.c[UID_COL], 
            dict_tableobj.c[UID_COL]
        ]
        constraint = sql.and_(*constraints)
        return PreSelect(select_cols=select_cols, constraint=constraint)
    
    def _compile_func(self, op_query:FuncQuery) -> BasePreSelect:
        """
        Compile the query corresponding to an op, including built-in ops
        """
        constraints = []
        io_cols = []
        for io_name, val_query in itertools.chain(
            op_query.inputs.items(),
            op_query.outputs.items()
        ):
            val_tableobj = self.val_aliases[val_query]
            constraints.append(
                self.op_aliases[op_query].c[io_name] ==
                val_tableobj.c[UID_COL]
            )
            io_cols.append(val_tableobj.c[UID_COL])
        constraint = sql.and_(*constraints)
        return PreSelect(select_cols=io_cols, constraint=constraint)
    
    def compile_op(self, op_query:OpQuery) -> BasePreSelect:
        if isinstance(op_query, ConstructListQuery):
            return self._compile_construct_list(cl_query=op_query)
        elif isinstance(op_query, DeconstructListQuery):
            return self._compile_deconstruct_list(dl_query=op_query)
        elif isinstance(op_query, ConstructDictQuery):
            return self._compile_construct_dict(cd_query=op_query)
        elif isinstance(op_query, FuncQuery):
            res = self._compile_func(op_query=op_query)
            return res
        else:
            raise NotImplementedError()
    
    def resolve_constraints(self, val_query:ValQuery) -> TOption[TList[str]]:
        vals_adapter = self.val_adapter
        query_tp = val_query.tp
        constraints = val_query.constraints
        current_locs = None
        for constraint_id, constraint_data in constraints:
            if constraint_id == 'isin':
                current_locs = vals_adapter.isin(rng=constraint_data,
                                                 tp=query_tp, locs=current_locs)
            elif constraint_id == 'where':
                current_locs = vals_adapter.where(pred=constraint_data,
                                                  tp=query_tp, locs=current_locs)
            elif constraint_id == 'equals':
                current_locs = vals_adapter.isin(rng=[constraint_data],
                                                 tp=query_tp, locs=current_locs)
            elif constraint_id == 'identical':
                current_locs = [vals_adapter.get_vref_location(vref=vref)
                                for vref in constraint_data]
            else:
                raise NotImplementedError(f'Got {constraint_id}!!!! :O')
        if current_locs is None:
            return None
        else:
            return [loc.uid for loc in current_locs]
    
    def compile_val(self, val_query:ValQuery) -> BasePreSelect:
        table_obj = self.val_aliases[val_query]
        columns = [table_obj.c[col] for col in self.rel_adapter.vref_schema]
        # figure out extra consraints
        keys = self.resolve_constraints(val_query=val_query)
        # figure out partitions
        partitions = self.val_adapter.get_partitions_matching_tp(tp=val_query.tp)
        constraint = (table_obj.c[PARTITION_COL].
                      in_(get_in_clause_rhs(partitions)))
        if keys is not None:
            if not keys:
                constraint = False
            else:
                constraint = sql.and_(
                    constraint,
                    table_obj.c[UID_COL].in_(get_in_clause_rhs(keys))
                )
        return PreSelect(select_cols=columns, constraint=constraint)
    
    def compile_calls(self, select_queries:TTuple[OpQuery,...]=None,
                      ) -> CompoundSelect:
        """
        Return a union query over UID columns of the given op queries that
        satisfies the constraints of the given query.
        """
        if select_queries is None:
            select_queries = tuple(self.op_queries)
        assert all([vq in self.op_queries for vq in select_queries])
        op_preselects = {op_query: self.compile_op(op_query=op_query) 
                         for op_query in self.op_queries}
        val_preselects = {val_query: self.compile_val(val_query=val_query)
                          for val_query in self.val_queries}
        select_cols = [self._op_aliases[op_query].c[UID_COL]
                       for op_query in select_queries]
        constraints = sql.and_(
            *[preselect.constraint()
              for preselect in
              itertools.chain(op_preselects.values(), val_preselects.values())]
        )
        sql_query = sql.union(*[sql.select(col).where(constraints)
                                for col in select_cols])
        return sql_query
    
    def get_calls(self, select_queries:TTuple[OpQuery,...]=None) -> TList[str]:
        query = self.compile_calls(select_queries=select_queries)
        df = self.rel_adapter.rel_storage.fast_select(query=query)
        return df[UID_COL].values.tolist()
    
    def compile(self, select_queries:TTuple[ValQuery,...]=None,
                ) -> TTuple[Select, TList[list]]:
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
        if select_queries is None:
            select_queries = tuple(self.val_queries)
        assert all([vq in self.val_queries for vq in select_queries])
        op_preselects = {op_query: self.compile_op(op_query=op_query) 
                         for op_query in self.op_queries}
        val_preselects = {val_query: self.compile_val(val_query=val_query)
                          for val_query in self.val_queries}
        partitioned_select_cols = [val_preselects[val_query].select_cols()
                                   for val_query in select_queries]
        select_cols = concat_lists(partitioned_select_cols)
        constraints = sql.and_(
            *[preselect.constraint()
              for preselect in
              itertools.chain(op_preselects.values(), val_preselects.values())]
        )
        sql_query = sql.select(select_cols).where(constraints)
        return sql_query, partitioned_select_cols
    
    def get_locations(self, select_queries:TTuple[ValQuery,...],
                      ) -> pd.DataFrame:
        q, partition_pattern = self.compile(select_queries=select_queries)
        if self._verbose:
            print(q)
        df = self.rel_adapter.rel_storage.fast_select(query=q)
        parts = self._split_dataframe(df=df, partition_pattern=partition_pattern)
        columns = []
        for part in parts:
            part.columns = self.rel_adapter.vref_schema
            columns.append(
                self.rel_adapter.get_vref_df_locations(vref_relations=part)
            )
        cols_dict = {q.sql_name: col for q, col in zip(select_queries, columns)}
        return pd.DataFrame(data=cols_dict)

    @staticmethod
    def _split_dataframe(df:pd.DataFrame, 
                         partition_pattern:TList[list]) -> TList[pd.DataFrame]:
        lengths = [len(x) for x in partition_pattern]
        split_points = [0] + list(np.cumsum(lengths))
        assert sum(lengths) == df.shape[1]
        return [pd.DataFrame(
            df.iloc[:, range(split_points[i], split_points[i+1])]
        )
            for i in range(len(split_points) - 1)]
