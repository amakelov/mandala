import textwrap
from ..common_imports import *
from ..core.config import *
from ..core.tps import Type
from ..core.model import Ref, FuncOp, Call
from ..core.builtins_ import StructOrientations, Builtins
from ..storages.rel_impls.utils import Transactable, transaction, Connection
from ..queries.graphs import copy_subgraph
from ..core.prov import propagate_struct_provenance
from ..queries.weaver import CallNode, ValNode
from ..queries.viz import GraphPrinter, visualize_graph
from .funcs import FuncInterface
from .storage import Storage, ValueLoader
from .cfs_utils import estimate_uid_storage, convert_bytes_to


class ComputationFrame(Transactable):
    """
    In-memory, dynamic representation of a slice of storage representing some
    computation, with methods for indexing, evaluating (i.e. loading from
    storage), and navigating back/forward along computation paths. These methods
    turn a `ComputationFrame` into a generalized dataframe over a computational
    graph, with columns corresponding to variables in the computation, and rows
    corresponding to values of those variables for a single instance of the
    computation.

    This is a simple declarative interface for exploring the storage, in
    contrast with using an imperative computational context (i.e., manipulating
    some memoized piece of code to interface with storage). It has several
    limitations, the main one being that it can only represent a single
    computation in a given instance (i.e., a single composition of functions).
    This comes at the benefit of simplicity and ease of use.

    The main differences between a `ComputationFrame` and a `DataFrame` are as
    follows:
        - by default, the `ComputationFrame` is lazy, i.e. it does not load the
        values of the variables it represents from the storage, but only their
        metadata in the form of `Ref` objects;
        - the `eval(vars)` method allows for loading the values of chosen
        variables from the storage, and returns an ordinary `pandas` dataframe
        with columns corresponding to the variables;
        - the `forward(vars)` and `back(vars)` methods allow for navigation
        along the computation graph, by expanding the graph to include the
        operation(s) that created/used given variables represented in the
        `ComputationFrame`;
        - the `creators(var)` and `consumers(var)` methods allow for inspecting
        the operations that created/used a variable (including ones not
        currently represented in the `ComputationFrame`);

    """

    def __init__(
        self,
        call_nodes: Dict[str, CallNode],
        val_nodes: Dict[str, ValNode],
        storage: Storage,
        prov_df: Optional[pd.DataFrame] = None,
    ):
        self.op_nodes = call_nodes
        self.var_nodes = val_nodes

        self.storage = storage
        if prov_df is None:
            prov_df = storage.rel_storage.get_data(table=Config.provenance_table)
            prov_df = propagate_struct_provenance(prov_df)
        self.prov_df = prov_df

    def __len__(self) -> int:
        if len(self.op_nodes) > 0:
            return len(self.op_nodes[list(self.op_nodes.keys())[0]].calls)
        if len(self.var_nodes) > 0:
            return len(self.var_nodes[list(self.var_nodes.keys())[0]].refs)
        return 0

    def to_pandas(
        self,
        columns: Optional[Union[str, Iterable[str]]] = None,
        values: Literal["refs", "lazy", "objs"] = "lazy",
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Extract a series/dataframe of refs from the RefFunctor, analogous to
        indexing into a pandas dataframe.
        """
        if columns is None:
            columns = (
                list(self.var_nodes.keys())
                if len(self.var_nodes) > 1
                else list(self.var_nodes.keys())
            )
        if isinstance(columns, str):
            refs = pd.Series(self.var_nodes[columns].refs, name=columns)
            full_uids_df = refs.apply(lambda x: x.full_uid)
            res_df = self.storage.eval_df(
                full_uids_df=full_uids_df.to_frame(), values=values
            )
            # turn into a series
            return res_df[columns]
        elif isinstance(columns, list) and all(isinstance(x, str) for x in columns):
            refs = pd.DataFrame(
                {col: self.var_nodes[col].refs for col in columns}
            )
            return self.storage.eval_df(
                full_uids_df=refs.applymap(
                    lambda x: x.full_uid if x is not None else None
                ),
                values=values,
            )
        else:
            raise ValueError(f"Invalid columns type: {type(columns)}")

    def __getitem__(
        self, indexer: Union[str, Iterable[str], np.ndarray]
    ) -> "ComputationFrame":
        """
        Analogous to pandas __getitem__, but tries to return a `ComputationFrame`
        """
        if isinstance(indexer, str):
            if indexer in self.var_nodes:
                return self.copy_subgraph(
                    val_nodes=[self.var_nodes[indexer]], call_nodes=[]
                )
            elif indexer in self.op_nodes:
                return self.copy_subgraph(
                    val_nodes=[], call_nodes=[self.op_nodes[indexer]]
                )
            else:
                raise ValueError(
                    f"Column {indexer} not found in variables or operations"
                )
        elif isinstance(indexer, list) and all(isinstance(x, str) for x in indexer):
            var_keys = [x for x in self.var_nodes.keys() if x in indexer]
            call_keys = [x for x in self.op_nodes.keys() if x in indexer]
            return self.copy_subgraph(
                val_nodes={self.var_nodes[k] for k in var_keys},
                call_nodes={self.op_nodes[k] for k in call_keys},
            )
        elif isinstance(indexer, (np.ndarray, pd.Series)):
            if isinstance(indexer, pd.Series):
                indexer = indexer.values
            # boolean mask
            if indexer.dtype == bool:
                res = self.copy_subgraph()
                for k, v in res.var_nodes.items():
                    v.inplace_mask(indexer)
                for k, v in res.op_nodes.items():
                    v.inplace_mask(indexer)
                return res
            else:
                raise NotImplementedError(
                    "Indexing with a non-boolean mask is not supported"
                )
        else:
            raise ValueError(
                f"Invalid indexer type into {self.__class__}: {type(indexer)}"
            )

    @transaction()
    def eval(
        self,
        indexer: Optional[Union[str, List[str]]] = None,
        conn: Optional[Connection] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        if indexer is None:
            indexer = list(self.var_nodes.keys())
        if isinstance(indexer, str):
            full_uids_df = (
                self[indexer]
                .to_pandas()
                .applymap(lambda x: x.full_uid if x is not None else None)
            )
            res_df = self.storage.eval_df(full_uids_df=full_uids_df, values="objs")
            res = res_df[indexer]
            return res
        else:
            full_uids_df = (
                self[indexer]
                .to_pandas()
                .applymap(lambda x: x.full_uid if x is not None else None)
            )
            return self.storage.eval_df(
                full_uids_df=full_uids_df, values="objs", conn=conn
            )

    @transaction()
    def creators(self, col: str, conn: Optional[Connection] = None) -> np.ndarray:
        calls, output_names = self.storage.get_creators(
            refs=self.var_nodes[col].refs, prov_df=self.prov_df, conn=conn
        )
        return np.array(
            [
                call.func_op.sig.versioned_ui_name if call is not None else None
                for call in calls
            ]
        )

    @transaction()
    def consumers(self, col: str, conn: Optional[Connection] = None) -> np.ndarray:
        calls_list, input_names_list = self.storage.get_consumers(
            refs=self.var_nodes[col].refs, prov_df=self.prov_df, conn=conn
        )
        res = np.empty(len(calls_list), dtype=object)
        res[:] = [
            tuple(
                [
                    call.func_op.sig.versioned_ui_name if call is not None else None
                    for call in calls
                ]
            )
            for calls in calls_list
        ]
        return res

    @transaction()
    def get_adjacent_calls(
        self,
        col: str,
        direction: Literal["back", "forward"],
        conn: Optional[Connection] = None,
    ) -> Dict[Tuple[str, str], List[Optional[Call]]]:
        """
        Given a column and a direction to traverse the graph (back or forward),
        return the calls that created/used the values in the column, along with
        the output/input names under which the values appear in the calls.

        The calls are grouped by the operation and the output/input name under
        which the values in this column appear in the calls.
        """
        refs: List[Ref] = self.var_nodes[col].refs
        if direction == "back":
            calls_list, names_list = self.storage.get_creators(
                refs=refs, prov_df=self.prov_df, conn=conn
            )
            calls_list = [[c] if c is not None else [] for c in calls_list]
            names_list = [[n] if n is not None else [] for n in names_list]
        elif direction == "forward":
            calls_list, names_list = self.storage.get_consumers(
                refs=refs, prov_df=self.prov_df, conn=conn
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")
        index_dict = defaultdict(list)  # (op_id, input/output name) -> (idx, call)
        # for i, (calls, names) in enumerate(zip(calls_list, names_list)):
        for i in range(len(refs)):
            calls = calls_list[i]
            names = names_list[i]
            for call, name in zip(calls, names):
                index_dict[(call.func_op.sig.versioned_ui_name, name)].append((i, call))
        res = {}
        for (op_id, name), indices_and_calls in index_dict.items():
            calls_list = [None for _ in range(len(refs))]
            for i, call in indices_and_calls:
                calls_list[i] = call
            res[(op_id, name)] = calls_list
        return res

    @transaction()
    def _back_all(
        self,
        res: "ComputationFrame",
        inplace: bool = False,
        verbose: bool = False,
        conn: Optional[Connection] = None,
    ) -> "ComputationFrame":
        # this means we want to expand the entire graph
        node_frontier = res.var_nodes.keys()
        visited = set()
        while True:
            res = res.back(
                cols=list(node_frontier),
                inplace=inplace,
                skip_failures=True,
                verbose=verbose,
                conn=conn,
            )
            visited |= node_frontier
            nodes_after = set(res.var_nodes.keys())
            node_frontier = nodes_after - visited
            if not node_frontier:
                break
        return res

    def join_var_node(
        self,
        refs: List[Ref],
        tp: Type,
        name_hint: Optional[str] = None,
    ) -> Tuple[str, ValNode]:
        refs_hash = ValNode.get_refs_hash(refs=refs)
        for var_name, var_node in self.var_nodes.items():
            if var_node.refs_hash == refs_hash:
                return var_name, var_node
        else:
            res = ValNode(
                tp=tp,
                refs=refs,
                constraint=None,
            )
            res_name = self.get_new_vname(hint=name_hint)
            self.var_nodes[res_name] = res
            return res_name, res

    def join_op_node(
        self,
        calls: List[Call],
        out_map: Optional[Dict[str, str]] = None,  # output name -> var name
        in_map: Optional[Dict[str, str]] = None,  # input name -> var name
    ) -> Tuple[str, CallNode]:
        """
        Join an op node to the graph. If a node with this hash already exists,
        only connect any not yet connected inputs/outputs; otherwise, create a
        new node and then connect inputs/outputs.
        """
        out_map = {} if out_map is None else out_map
        in_map = {} if in_map is None else in_map
        calls_hash = CallNode.get_calls_hash(calls=calls)
        for op_name, op_node in self.op_nodes.items():
            if op_node.calls_hash == calls_hash:
                res = op_node
                res_name = op_name
                break
        else:
            call_representative = calls[0]
            op = call_representative.func_op

            #! for struct calls, figure out the orientation; currently ad-hoc
            if op.is_builtin:
                logging.warning(
                    f"Found a ref data structure: {op.sig.ui_name}; ComputationFrame support for this is experimental and may not work as expected."
                )
                output_names = list(out_map.keys())
                input_names = list(in_map.keys())
                if len(output_names) == 0:
                    if any(x in input_names for x in ("elt", "value")):
                        orientation = StructOrientations.construct
                    elif any(x in input_names for x in ("lst", "dct", "st")):
                        orientation = StructOrientations.destruct
                    else:
                        raise NotImplementedError
                else:
                    if any(x in output_names for x in ("lst", "dct", "st")):
                        orientation = StructOrientations.construct
                    elif any(x in output_names for x in ("idx", "key")):
                        orientation = StructOrientations.destruct
                    else:
                        raise NotImplementedError
                in_map, out_map = Builtins.reassign_io_using_orientation(
                    in_dict=in_map,
                    out_dict=out_map,
                    orientation=orientation,
                    builtin_id=op.sig.ui_name,
                )
                in_map_for_linking = {**in_map, **out_map}
                out_map_for_linking = {}
            else:
                orientation = None
                in_map_for_linking = in_map
                out_map_for_linking = out_map

            res = CallNode.link(
                calls=calls,
                inputs={k: self.var_nodes[v] for k, v in in_map_for_linking.items()},
                outputs={k: self.var_nodes[v] for k, v in out_map_for_linking.items()},
                constraint=None,
                func_op=call_representative.func_op,
                orientation=orientation,
            )
            res_name = self.get_new_cname(op)
            self.op_nodes[res_name] = res
        for k, v in out_map.items():
            if k not in res.outputs.keys():
                # connect manually
                res.outputs[k] = self.var_nodes[v]
                self.var_nodes[v].add_creator(creator=res, created_as=k)
        for k, v in in_map.items():
            if k not in res.inputs.keys():
                # connect manually
                res.inputs[k] = self.var_nodes[v]
                self.var_nodes[v].add_consumer(consumer=res, consumed_as=k)
        return res_name, res

    @transaction()
    def back(
        self,
        cols: Optional[Union[str, List[str]]] = None,
        inplace: bool = False,
        skip_failures: bool = False,
        verbose: bool = False,
        conn: Optional[Connection] = None,
    ) -> "ComputationFrame":
        res = self if inplace else self.copy_subgraph()
        if cols is None:
            # this means we want to expand the entire graph
            return res._back_all(res, inplace=inplace, verbose=verbose, conn=conn)
        if verbose:
            logger.info(f"Expanding graph to include the provenance of columns {cols}")
        if isinstance(cols, str):
            cols = [cols]

        adjacent_calls_data = {
            col: res.get_adjacent_calls(col=col, direction="back", conn=conn)
            for col in cols
        }

        filtered_cols = []
        for col, calls_dict in adjacent_calls_data.items():
            if len(calls_dict) > 1:
                reason = f"Values in column {col} were created by multiple ops and/or as different outputs: {[f'{op}::{out}' for op, out in calls_dict.keys()]}"
                if skip_failures:
                    if verbose:
                        logger.info(f"{reason}; skipping column {col}")
                    continue
                else:
                    raise ValueError(reason)
            if len(calls_dict) == 0 or any(call is None for call in calls_dict[list(calls_dict.keys())[0]]):
                reason = f"Some refs in column {col} were not created by any op"
                if skip_failures:
                    if verbose:
                        logger.info(f"{reason}; skipping column {col}")
                    continue
                else:
                    raise ValueError(reason)
            filtered_cols.append(col)
        cols = filtered_cols

        for col in cols:
            calls_dict = adjacent_calls_data[col]
            op_id, output_name = list(calls_dict.keys())[0]
            calls = calls_dict[(op_id, output_name)]
            # produce input nodes
            representative_call: Call = calls[0] 
            op = representative_call.func_op
            input_node_names = {}
            for input_name, input_tp in op.input_types.items():
                input_node_names[input_name], _ = res.join_var_node(
                    refs=[call.inputs[input_name] for call in calls],
                    tp=input_tp,
                    name_hint=input_name,)
            res.join_op_node(
                calls=calls,
                out_map={output_name: col},
                in_map={k: v for k, v in input_node_names.items()},
            )
            
        return res

    @transaction()
    def delete(
        self,
        delete_dependents: bool,
        verbose: bool = True,
        ask: bool = True,
        conn: Optional[Connection] = None,
    ):
        """
        ! this is a powerful method that can delete a lot of data, use with caution

        Delete the calls referenced by this ComputationFrame from the storage, and
        clean up any orphaned refs.

        Warning: You probably want to apply this only on RefFunctors that are
        "forward-closed", i.e., that have been expanded to include all the calls
        that use their values. Otherwise, you may end up with obscure refs for
        which you have no provenance, i.e. "zombie" refs that have no meaning in
        the context of the rest of the storage. Alternatively, you can set
        `delete_dependents` to True, which will delete all the calls that depend
        on the calls in this ComputationFrame, and then clean up the orphaned refs.
        """
        # gather all the calls to be deleted
        call_uids_to_delete = defaultdict(list)
        call_outputs = {}
        for x in self.op_nodes.values():
            # process the call uids
            for call in x.calls:
                call_uids_to_delete[x.func_op.sig.versioned_ui_name].append(
                    call.causal_uid
                )
            # process the call outputs
            if delete_dependents:
                for vnode in x.outputs.values():
                    for ref in vnode.refs:
                        call_outputs[ref.causal_uid] = ref
        if delete_dependents:
            dependent_calls = self.storage.get_dependent_calls(
                refs=list(call_outputs.values()), prov_df=self.prov_df, conn=conn
            )
            for call in dependent_calls:
                call_uids_to_delete[call.func_op.sig.versioned_ui_name].append(
                    call.causal_uid
                )
        if verbose:
            # summarize the number of calls per op to be deleted
            for op, uids in call_uids_to_delete.items():
                print(f"Op {op} has {len(uids)} calls to be deleted")
            if ask:
                if input("Proceed? (y/n) ").strip().lower() != "y":
                    logging.info("Aborting deletion")
                    return
        for versioned_ui_name, call_uids in call_uids_to_delete.items():
            self.storage.rel_adapter.delete_calls(
                versioned_ui_name=versioned_ui_name,
                causal_uids=call_uids,
                conn=conn,
            )
        self.storage.rel_adapter.cleanup_vrefs(conn=conn, verbose=verbose)

    ############################################################################
    ### creating new RefFunctors
    ############################################################################
    def copy_subgraph(
        self,
        val_nodes: Optional[Iterable[ValNode]] = None,
        call_nodes: Optional[Iterable[CallNode]] = None,
    ) -> "ComputationFrame":
        """
        Get a copy of the ComputationFrame supported on the given nodes.
        """
        # must copy the graph
        val_nodes = (
            set(self.var_nodes.values()) if val_nodes is None else set(val_nodes)
        )
        call_nodes = (
            set(self.op_nodes.values()) if call_nodes is None else set(call_nodes)
        )
        val_map, call_map = copy_subgraph(
            vqs=val_nodes,
            fqs=call_nodes,
        )
        return ComputationFrame(
            call_nodes={
                k: call_map[v] for k, v in self.op_nodes.items() if v in call_map
            },
            val_nodes={
                k: val_map[v] for k, v in self.var_nodes.items() if v in val_map
            },
            storage=self.storage,
            prov_df=self.prov_df,
        )

    @staticmethod
    def from_refs(
        refs: Iterable[Ref],
        storage: Storage,
        prov_df: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> "ComputationFrame":
        val_node = ValNode(
            constraint=None,
            tp=None,
            # refs=list(refs),
            refs=refs,
        )
        name = "v0" if name is None else name
        return ComputationFrame(
            call_nodes={},
            val_nodes={name: val_node},
            storage=storage,
            prov_df=prov_df,
        )

    @staticmethod
    def from_op(
        func: FuncInterface,
        storage: Storage,
        prov_df: Optional[pd.DataFrame] = None,
    ) -> "ComputationFrame":
        """
        Get a ComputationFrame expressing the memoization table for a single function
        """
        storage.synchronize(f=func)
        reftable = storage.get_table(func, values="lazy", meta=True)
        op = func.func_op
        if op.is_builtin:
            raise ValueError("Cannot create a ComputationFrame from a builtin op")
        input_nodes = {
            input_name: ValNode(
                constraint=None,
                tp=op.input_types[input_name],
                refs=reftable[input_name].values.tolist(),
            )
            for input_name in op.input_types.keys()
        }
        output_nodes = {
            dump_output_name(i): ValNode(
                constraint=None,
                tp=tp,
                refs=reftable[dump_output_name(i)].values.tolist(),
            )
            for i, tp in enumerate(op.output_types)
        }
        call_uids = reftable[Config.causal_uid_col].values.tolist()
        calls = storage.cache.call_mget(
            uids=call_uids,
            versioned_ui_name=op.sig.versioned_ui_name,
            by_causal=True,
        )
        call_node = CallNode.link(
            inputs=input_nodes,
            func_op=op,
            outputs=output_nodes,
            constraint=None,
            calls=calls,
            orientation=None,
        )
        return ComputationFrame(
            call_nodes={func.func_op.sig.versioned_ui_name: call_node},
            val_nodes={
                k: v
                for k, v in itertools.chain(input_nodes.items(), output_nodes.items())
            },
            storage=storage,
            prov_df=prov_df,
        )

    ############################################################################
    ### visualization
    ############################################################################
    def get_printer(self) -> GraphPrinter:
        printer = GraphPrinter(
            vqs=set(self.var_nodes.values()),
            fqs=set(self.op_nodes.values()),
            names={v: k for k, v in self.var_nodes.items()},
            fnames={v: k for k, v in self.op_nodes.items()},
            value_loader=ValueLoader(storage=self.storage),
        )
        return printer

    def _get_string_representation(self) -> str:
        printer = self.get_printer()
        graph_description = printer.print_computational_graph(
            show_sources_as="name_only"
        )
        # indent the graph description
        graph_description = textwrap.indent(graph_description, "  ")
        return f"{self.__class__.__name__} with {self.num_vars} variable(s), {self.num_ops} operation(s) and {len(self)} row(s), representing the computation:\n{graph_description}"

    def __repr__(self) -> str:
        return self._get_string_representation()
        # return self.to_pandas().head(5).to_string()

    def print(self):
        print(self._get_string_representation())

    def show(self, how: Literal["inline", "browser"] = "browser"):
        visualize_graph(
            vqs=set(self.var_nodes.values()),
            fqs=set(self.op_nodes.values()),
            layout="computational",
            names={v: k for k, v in self.var_nodes.items()},
            show_how=how,
        )

    def get_new_vname(self, hint: Optional[str] = None) -> str:
        """
        Return the first name of the form `v{i}` that is not in self.val_nodes
        """
        if hint is not None and hint not in self.var_nodes:
            return hint
        i = 0
        prefix = "v" if hint is None else hint
        while f"{prefix}{i}" in self.var_nodes:
            i += 1
        return f"{prefix}{i}"

    def get_new_cname(self, op: FuncOp) -> str:
        if op.sig.versioned_ui_name not in self.op_nodes:
            return op.sig.versioned_ui_name
        i = 0
        while f"{op.sig.versioned_ui_name}_{i}" in self.op_nodes:
            i += 1
        return f"{op.sig.versioned_ui_name}_{i}"

    def rename(self, columns: Dict[str, str], inplace: bool = False):
        for old_name, new_name in columns.items():
            if old_name not in self.var_nodes:
                raise ValueError(f"Column {old_name} does not exist")
            if new_name in self.var_nodes:
                raise ValueError(f"Column {new_name} already exists")
        if inplace:
            res = self
        else:
            res = self.copy_subgraph()
        for old_name, new_name in columns.items():
            res.var_nodes[new_name] = res.var_nodes.pop(old_name)
        return res

    def r(
        self,
        inplace: bool = False,
        **kwargs,
    ) -> "ComputationFrame":
        """
        Fast alias for rename
        """
        return self.rename(columns=kwargs, inplace=inplace)

    @property
    def num_vars(self) -> int:
        return len(self.var_nodes)

    @property
    def num_ops(self) -> int:
        return len(self.op_nodes)

    def get_var_info(
        self,
        include_uniques: bool = False,
        small_threshold_bytes: int = 4096,
        units: Literal["bytes", "KB", "MB", "GB"] = "MB",
        sample_size: int = 20,
    ) -> pd.DataFrame:
        var_rows = []
        for k, v in self.var_nodes.items():
            if len(v.refs) == 0:
                avg_size, std = 0, 0
            else:
                avg_size_bytes, std_bytes = estimate_uid_storage(
                    uids=[ref.uid for ref in v.refs],
                    storage=self.storage,
                    units="bytes",
                    sample_size=sample_size,
                )
                avg_size, std = convert_bytes_to(
                    num_bytes=avg_size_bytes, units=units
                ), convert_bytes_to(num_bytes=std_bytes, units=units)
            # round to 2 decimal places
            avg_size, std = round(avg_size, 2), round(std, 2)
            var_data = {
                "name": k,
                "size": f"{avg_size}±{std} {units}",
                "nunique": len(set(ref.uid for ref in v.refs)),
            }
            if include_uniques:
                if avg_size_bytes < small_threshold_bytes:
                    uniques = {ref.uid: ref for ref in v.refs}
                    uniques_values = self.storage.unwrap(list(uniques.values()))
                    try:
                        uniques_values = sorted(uniques_values)
                    except:
                        pass
                    var_data["unique_values"] = uniques_values
                else:
                    var_data["unique_values"] = "<too large>"
            var_rows.append(var_data)
        var_df = pd.DataFrame(var_rows)
        var_df.set_index("name", inplace=True)
        var_df = var_df.sort_values(by="size", ascending=False)
        return var_df

    def get_op_info(self) -> pd.DataFrame:
        rows = []
        for k, op_node in self.op_nodes.items():
            input_types = op_node.func_op.input_types
            output_types = {
                dump_output_name(index=i): op_node.func_op.output_types[i]
                for i in range(len(op_node.func_op.output_types))
            }
            input_types_dict = {
                k: input_types.get(k, output_types.get(k))
                for k in op_node.inputs.keys()
            }
            output_types_dict = {
                k: output_types.get(k, input_types.get(k))
                for k in op_node.outputs.keys()
            }
            signature = f'{op_node.func_op.sig.ui_name}({", ".join([f"{k}: {v}" for k, v in input_types_dict.items()])}) -> {", ".join([f"{k}: {v}" for k, v in output_types_dict.items()])}'
            rows.append(
                {
                    "name": k,
                    "function": op_node.func_op.sig.ui_name,
                    "version": op_node.func_op.sig.version,
                    "num_calls": len(op_node.calls),
                    "num_unique_calls": len(
                        set(call.causal_uid for call in op_node.calls)
                    ),
                    "signature": signature,
                }
            )
        op_df = pd.DataFrame(rows)
        op_df.set_index("name", inplace=True)
        return op_df

    def info(
        self,
        units: Literal["bytes", "KB", "MB", "GB"] = "MB",
        sample_size: int = 20,
        show_uniques: bool = False,
        small_threshold_bytes: int = 4096,
    ):
        """
        Print some basic info about the ComputationFrame
        """
        # print(self.__class__)
        var_df = self.get_var_info(
            include_uniques=show_uniques,
            small_threshold_bytes=small_threshold_bytes,
            units=units,
            sample_size=sample_size,
        )
        op_df = self.get_op_info()
        print(
            f"{self.__class__.__name__} with {self.num_vars} variable(s), {self.num_ops} operation(s), {len(self)} row(s)"
        )
        printer = self.get_printer()
        print("Computation graph:")
        print(
            textwrap.indent(
                printer.print_computational_graph(show_sources_as="name_only"), "  "
            )
        )
        try:
            print("Variables:")
            import prettytable
            from io import StringIO

            output = StringIO()
            var_df.to_csv(output)
            output.seek(0)
            pt = prettytable.from_csv(output)
            print(textwrap.indent(pt.get_string(), "  "))
            print("Operations:")
            output = StringIO()
            op_df.to_csv(output)
            output.seek(0)
            pt = prettytable.from_csv(output)
            print(textwrap.indent(pt.get_string(), "  "))
        except ImportError:
            print("Variables:")
            print(textwrap.indent(var_df.to_string(), "  "))
            print("Operations:")
            print(textwrap.indent(op_df.to_string(), "  "))
        # representative_node = self.val_nodes[list(self.val_nodes.keys())[0]]
        # num_rows = len(representative_node.refs)
        # print(f"ComputationFrame with {self.num_vars} variable(s) and {self.num_ops} operations(s), representing {num_rows} computations")

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.storage.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.storage.rel_storage._end_transaction(conn=conn)
