from ..common_imports import *
from ..core.model import FuncOp
from ..core.sig import Signature
from ..core.config import Config
from .weaver import ValQuery, FuncQuery, traverse_all
from .graphs import InducedSubgraph
from .viz import visualize_graph, get_names
from ..core.utils import invert_dict


class NaiveQueryEngine:
    """
    Represents the graph expressing a query in a form that is suitable for
    incrementally computing the join of all the tables.

    Used as an alternative to a RDBMS engine for computing queries.

    Should work with induced subgraphs
    """

    def __init__(
        self,
        vqs: Set[ValQuery],
        fqs: Set[FuncQuery],
        selection: List[ValQuery],
        tables: Dict[FuncQuery, pd.DataFrame],
        _table_evaluator: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        _visualize_steps_at: Optional[Path] = None,
    ):
        self.vqs = vqs
        self.fqs = list(fqs)
        self.g = InducedSubgraph(vqs=vqs, fqs=fqs)
        self.selection = selection
        # {func query: table of data}. Note that there may be multiple func
        # queries with the same table, but we keep separate references to each
        # in order to enable recursively joining nodes in the graph.

        # pass to induced tables wrt the graph
        self.induce_tables(tables=tables)
        self.tables = tables
        for k, v in self.tables.items():
            for col in Config.special_call_cols:
                if col in v.columns:
                    v.drop(columns=[col], inplace=True)

        # for visualization
        self._visualize_intermediate_states = _visualize_steps_at is not None
        self._table_evaluator = _table_evaluator
        self._visualize_steps_at = _visualize_steps_at
        if self._visualize_intermediate_states:
            assert self._table_evaluator is not None

    def induce_tables(self, tables: Dict[FuncQuery, pd.DataFrame]):
        for fq, df in tables.items():
            inps, outps = self.get_fq_inputs(fq), self.get_fq_outputs(fq)
            induced_keys = set(inps.keys()) | set(outps.keys())
            df.drop(
                columns=[c for c in df.columns if c not in induced_keys], inplace=True
            )

    def get_fq_inputs(self, fq: FuncQuery) -> Dict[str, ValQuery]:
        if fq in self.g.fqs:
            return self.g.fq_inputs(fq=fq)
        else:
            return fq.inputs

    def get_fq_outputs(self, fq: FuncQuery) -> Dict[str, ValQuery]:
        if fq in self.g.fqs:
            return self.g.fq_outputs(fq=fq)
        else:
            return fq.outputs

    def _get_col_to_vq_mappings(
        self, func_query: FuncQuery
    ) -> Tuple[Dict[str, ValQuery], Dict[ValQuery, List[str]]]:
        """
        Given a FuncQuery, returns:
            - a mapping from column names to the ValQuery that they point to
            - a mapping from ValQuery objects to the list of column names that point to it
        """
        df = self.tables[func_query]
        col_to_vq = {}
        vq_to_cols = defaultdict(list)
        for name, val_query in self.get_fq_inputs(func_query).items():
            assert name in df.columns
            col_to_vq[name] = val_query
            vq_to_cols[val_query].append(name)
        for output_name, val_query in self.get_fq_outputs(func_query).items():
            assert output_name in df.columns
            col_to_vq[output_name] = val_query
            vq_to_cols[val_query].append(output_name)
        return col_to_vq, vq_to_cols

    def _drop_self_constraints(
        self, df: pd.DataFrame, vq_to_cols: Dict[ValQuery, List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, ValQuery], Dict[ValQuery, str]]:
        new_col_to_vq = {}
        df = df.copy()
        for vq, cols in vq_to_cols.items():
            if len(cols) > 1:
                representative = cols[0]
                for col in cols[1:]:
                    df = df[df[representative] == df[col]]
                    df.drop(columns=col, inplace=True)
            new_col_to_vq[cols[0]] = vq
        new_vq_to_col = {vq: col for col, vq in new_col_to_vq.items()}
        return df, new_col_to_vq, new_vq_to_col

    def _join_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        left_on: List[str],
        right_on: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
        """
        Join two dataframes along the specified dimensions.

        Returns
            - the result,
            - together with coprojections from the columns of each dataframe to
            the columns of the result.
        """
        assert len(left_on) == len(right_on)
        # rename the dataframe columns to avoid conflicts
        mapping1 = {col: f"input_{i}" for i, col in enumerate(df1.columns)}
        ncols1 = len(df1.columns)
        mapping2 = {col: f"input_{i + ncols1}" for i, col in enumerate(df2.columns)}
        renamed_df1 = df1.rename(columns=mapping1)
        renamed_df2 = df2.rename(columns=mapping2)
        # join the dataframes
        logger.info(f"Joining tables of shapes {df1.shape} and {df2.shape}...")
        start = time.time()
        if len(left_on) == 0:
            df = renamed_df1.merge(renamed_df2, how="cross")
        else:
            df = renamed_df1.merge(
                renamed_df2,
                left_on=[mapping1[col] for col in left_on],
                right_on=[mapping2[col] for col in right_on],
            )
        end = time.time()
        logger.info(f"Join took {round(end - start, 3)} seconds")
        # drop duplicate columns from the *right* dataframe
        df.drop(columns=[mapping2[col] for col in right_on], inplace=True)
        # construct the mapping functions from the columns of each dataframe to
        # the columns of the result
        from1 = mapping1
        from2 = {}
        for col in df2.columns:
            if mapping2[col] in df.columns:
                # first, assign the columns that we did not drop
                from2[col] = mapping2[col]
        for left_col, right_col in zip(left_on, right_on):
            # fill in the remaining ones
            from2[right_col] = mapping1[left_col]
        return df, from1, from2

    def merge(self, f1: FuncQuery, f2: FuncQuery):
        """
        Merge two func query nodes in the graph by joining their tables along
        the columns that correspond to the shared inputs/outputs
        """
        # get the data
        df1, df2 = self.tables[f1], self.tables[f2]
        # compute correspondence between columns and vqs
        col_to_vq1, vq_to_cols1 = self._get_col_to_vq_mappings(f1)
        col_to_vq2, vq_to_cols2 = self._get_col_to_vq_mappings(f2)
        # apply self-join constraints
        df1, col_to_vq1, vq_to_col1 = self._drop_self_constraints(df1, vq_to_cols1)
        df2, col_to_vq2, vq_to_col2 = self._drop_self_constraints(df2, vq_to_cols2)
        # compute the pairs of columns along which we need to join
        # {shared value query: (col1, col2)}
        intersection_vq_to_col_pairs = OrderedDict({})
        for col, vq in col_to_vq1.items():
            if vq in col_to_vq2.values():
                intersection_vq_to_col_pairs[vq] = (col, vq_to_col2[vq])
        left_on = [col for _, (col, _) in intersection_vq_to_col_pairs.items()]
        right_on = [col for _, (_, col) in intersection_vq_to_col_pairs.items()]
        df, from1, from2 = self._join_dataframes(
            df1=df1, df2=df2, left_on=left_on, right_on=right_on
        )
        # get the correspondence between columns and vqs for the new table
        inputs = {}
        for col in df.columns:
            if col in from1.values():
                col_1 = invert_dict(from1)[col]
                inputs[col] = col_to_vq1[col_1]
            elif col in from2.values():
                col_2 = invert_dict(from2)[col]
                inputs[col] = col_to_vq2[col_2]
            else:
                raise ValueError()
        # insert new func query
        sig = Signature(
            ui_name="internal_node",
            input_names=set(inputs.keys()),
            n_outputs=0,
            version=0,
            defaults={},
            input_annotations={k: Any for k in inputs.keys()},
            output_annotations=[Any for _ in range(0)],
        )
        func_op = FuncOp._from_sig(sig=sig)
        f = FuncQuery(inputs=inputs, func_op=func_op, outputs={}, constraint=None)
        for k, v in inputs.items():
            v.add_consumer(consumer=f, consumed_as=k)
        self.tables[f] = df
        self.fqs.append(f)
        # remove the old func queries from the graph
        f1.unlink()
        f2.unlink()
        self.fqs = [f for f in self.fqs if f not in (f1, f2)]
        del self.tables[f1], self.tables[f2]

    ### solver and solver utils
    def compute_intersection_size(self, f1: FuncQuery, f2: FuncQuery) -> int:
        col_to_vq1, vq_to_cols1 = self._get_col_to_vq_mappings(f1)
        col_to_vq2, vq_to_cols2 = self._get_col_to_vq_mappings(f2)
        return len(set(col_to_vq1.values()) & set(col_to_vq2.values()))

    def _visualize_state(self, step_num: int):
        assert self._visualize_intermediate_states
        val_queries, func_queries = traverse_all(vqs=self.selection)
        memoization_tables = {
            k: self._table_evaluator(v) for k, v in self.tables.items()
        }
        visualize_graph(
            vqs=val_queries,
            fqs=func_queries,
            names=get_names(hints={}, canonical_order=list(val_queries)),
            output_path=self._visualize_steps_at / f"{step_num}.svg",
            layout="bipartite",
            memoization_tables=memoization_tables,
        )

    def solve(self, verbose: bool = False) -> pd.DataFrame:
        step_num = 0
        while len(self.fqs) > 1:
            if verbose:
                logger.info(f"step {step_num}")
            if self._visualize_intermediate_states:
                self._visualize_state(step_num=step_num)
            intersections = {}
            # compute pairwise intersections
            for f1 in self.fqs:
                for f2 in self.fqs:
                    if f1 == f2:
                        continue
                    intersections[(f1, f2)] = self.compute_intersection_size(f1, f2)
            # pick the pair with the largest intersection
            f1, f2 = max(intersections, key=lambda x: intersections.get(x, 0))
            # merge the pair
            self.merge(f1, f2)
            step_num += 1
        if self._visualize_intermediate_states:
            self._visualize_state(step_num=step_num)
        assert len(self.tables) == 1
        df = self.tables[self.fqs[0]]
        f = self.fqs[0]
        # figure out which columns to select
        col_to_vq, vq_to_cols = self._get_col_to_vq_mappings(f)
        cols = [vq_to_cols[vq][0] for vq in self.selection]
        return df[cols]
