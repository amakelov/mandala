from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from pypika import Query, Table, Parameter
from pypika.terms import LiteralValue
from duckdb import DuckDBPyConnection as Connection

from ..common_imports import *
from ..core.config import Config, Prov, dump_output_name
from ..core.model import Call, unwrap, FuncOp, ValueRef
from ..core.sig import Signature
from ..utils import serialize, deserialize, _rename_cols
from .rel_impls.duckdb_impl import DuckDBRelStorage
from .rel_impls.utils import Transactable, transaction


# {internal table name -> serialized (internally named) table}
RemoteEventLogEntry = dict[str, bytes]


class SigAdapter(Transactable):
    """
    Responsible for state transitions of the schema that update the
    signature objects *and* the relational tables in a transactional way.
    """

    def __init__(
        self,
        rel_adapter: "RelAdapter",
    ):
        self.rel_adapter = rel_adapter
        self.rel_storage = self.rel_adapter.rel_storage

    @transaction()
    def dump_state(
        self, state: Dict[Tuple[str, int], Signature], conn: Optional[Connection] = None
    ):
        """
        Dump the given state of the signatures to the database. Should always
        call this after the signatures have been updated.
        """
        # delete existing, if any
        index_col = "index"
        query = f"DELETE FROM {self.rel_adapter.SIGNATURES_TABLE} WHERE {index_col} = 0"
        conn.execute(query)
        # insert new
        serialized = serialize(obj=state)
        df = pd.DataFrame(
            {
                index_col: [0],
                "signatures": [serialized],
            }
        )
        ta = pa.Table.from_pandas(df)
        self.rel_storage.insert(
            relation=self.rel_adapter.SIGNATURES_TABLE, ta=ta, conn=conn
        )

    @transaction()
    def load_state(
        self, conn: Optional[Connection] = None
    ) -> Dict[Tuple[str, int], Signature]:
        """
        Load the state of the signatures from the database. All interactions
        with the state of the signatures are done transactionally through this
        method.
        """
        query = f"SELECT * FROM {self.rel_adapter.SIGNATURES_TABLE} WHERE index = 0"
        df = self.rel_storage.execute_df(query=query, conn=conn)
        if len(df) == 0:
            return {}
        else:
            return deserialize(df["signatures"][0])

    def check_invariants(
        self,
        sigs: Optional[Dict[Tuple[str, int], Signature]] = None,
        conn: Optional[Connection] = None,
    ):
        """
        This checks that the invariants of the *set* of signatures for the storage
        hold. This means that:
            - all versions of a signature are consecutive integers starting from
              0
            - signatures have the same UI name iff they have the same internal
            name

        Invariants for individual signatures are not checked by this (they are
        checked by the `Signature` class).
        """
        # check version numbering
        if sigs is None:
            sigs = self.load_state(conn=conn)
        internal_names = {internal_name for internal_name, _ in sigs.keys()}
        for internal_name in internal_names:
            versions = [version for _, version in sigs.keys() if _ == internal_name]
            assert sorted(versions) == list(range(len(versions)))
        # check exactly 1 UI name per internal name
        internal_to_ui_names = defaultdict(set)
        for (internal_name, _), sig in sigs.items():
            internal_to_ui_names[internal_name].add(sig.ui_name)
        for internal_name, ui_names in internal_to_ui_names.items():
            assert (
                len(ui_names) == 1
            ), f"Internal name {internal_name} has multiple UI names: {ui_names}"
        # check exactly 1 internal name per UI name
        ui_to_internal_names = defaultdict(set)
        for (internal_name, _), sig in sigs.items():
            ui_to_internal_names[sig.ui_name].add(internal_name)
        for ui_name, internal_names in ui_to_internal_names.items():
            assert (
                len(internal_names) == 1
            ), f"UI name {ui_name} has multiple internal names: {internal_names}"

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
    ###
    ############################################################################
    @transaction()
    def load_ui_sigs(
        self, conn: Optional[Connection] = None
    ) -> Dict[Tuple[str, int], Signature]:
        """
        Get the signatures indexed by (ui_name, version)
        """
        sigs = self.load_state(conn=conn)
        res = {(sig.ui_name, sig.version): sig for sig in sigs.values()}
        assert len(res) == len(sigs)
        return res

    @transaction()
    def exists_versioned_ui(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> bool:
        """
        Check if the signature exists based on its UI name *and* version
        """
        return (sig.ui_name, sig.version) in self.load_ui_sigs(conn=conn)

    @transaction()
    def exists_any_version(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> bool:
        """
        Check using internal name (or UI name, if it has no internal data) if
        there exists any version for this signature.
        """
        if sig.has_internal_data:
            return any(
                sig.internal_name == k[0] for k in self.load_state(conn=conn).keys()
            )
        else:
            return any(sig.ui_name == k[0] for k in self.load_ui_sigs(conn=conn).keys())

    @transaction()
    def get_latest_version(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        """
        Get the latest version of the signature, based on internal name or UI
        name if it has no internal data.
        """
        sigs = self.load_state(conn=conn)
        if sig.has_internal_data:
            versions = [k[1] for k in sigs.keys() if k[0] == sig.internal_name]
            if len(versions) == 0:
                raise ValueError(f"No versions for signature {sig}")
            version = max(versions)
            return sigs[(sig.internal_name, version)]
        else:
            versions = [
                k[1] for k in self.load_ui_sigs(conn=conn).keys() if k[0] == sig.ui_name
            ]
            if len(versions) == 0:
                raise ValueError(f"No versions for signature {sig}")
            version = max(versions)
            return self.load_ui_sigs(conn=conn)[(sig.ui_name, version)]

    @transaction()
    def get_versions(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> List[int]:
        """
        Get all versions of the signature, based on internal name or UI name if
        it has no internal data.
        """
        sigs = self.load_state(conn=conn)
        if sig.has_internal_data:
            return [k[1] for k in sigs.keys() if k[0] == sig.internal_name]
        else:
            ui_sigs = self.load_ui_sigs(conn=conn)
            return [k[1] for k in ui_sigs.keys() if k[0] == sig.ui_name]

    @transaction()
    def exists_internal(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> bool:
        """
        Check if the signature exists based on its *internal* name
        """
        return (sig.internal_name, sig.version) in self.load_state(conn=conn)

    @transaction()
    def internal_to_ui(self, conn: Optional[Connection] = None) -> Dict[str, str]:
        """
        Get a mapping from internal names to UI names
        """
        return {k[0]: v.ui_name for k, v in self.load_state(conn=conn).items()}

    @transaction()
    def ui_to_internal(self, conn: Optional[Connection] = None) -> Dict[str, str]:
        """
        Get a mapping from UI names to internal names
        """
        return {v.ui_name: k[0] for k, v in self.load_state(conn=conn).items()}

    @transaction()
    def ui_names(self, conn: Optional[Connection] = None) -> Set[str]:
        # return the set of ui names
        return set(self.ui_to_internal(conn=conn).keys())

    @transaction()
    def internal_names(self, conn: Optional[Connection] = None) -> Set[str]:
        # return the set of internal names
        return set(self.internal_to_ui(conn=conn).keys())

    @transaction()
    def is_sig_table_name(
        self, name: str, use_internal: bool, conn: Optional[Connection] = None
    ) -> bool:
        """
        Check if the name is a valid name for a table corresponding to a
        signature.
        """
        parts = name.split("_", 1)
        return (
            parts[0]
            in (
                self.internal_names(conn=conn)
                if use_internal
                else self.ui_names(conn=conn)
            )
            and parts[1].isdigit()
        )

    ############################################################################
    ### elementary transitions for local state
    ############################################################################
    @transaction()
    def create_sig(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Create a new signature `sig`. `sig` must have internal data, and not be
        present in storage at any version.
        """
        assert sig.has_internal_data
        sigs = self.load_state(conn=conn)
        assert sig.internal_name not in self.internal_names(conn=conn)
        # assert (sig.internal_name, sig.version) not in sigs.keys()
        sigs[(sig.internal_name, sig.version)] = sig
        # write signatures
        self.dump_state(state=sigs, conn=conn)
        # create relation
        columns = list(sig.input_names) + [
            dump_output_name(index=i) for i in range(sig.n_outputs)
        ]
        columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
        self.rel_storage.create_relation(
            name=sig.versioned_ui_name,
            columns=columns,
            primary_key=Config.uid_col,
            defaults=sig.new_ui_input_default_uids,
            conn=conn,
        )
        logger.debug(f"Created signature:\n{sig}")

    @transaction()
    def create_new_version(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Create a new version of an already existing function using the `sig`
        object. `sig` must have internal data, and the internal name must
        already be present in some version.
        """
        assert sig.has_internal_data
        latest_sig = self.get_latest_version(sig=sig, conn=conn)
        assert sig.version == latest_sig.version + 1
        # update signatures
        sigs = self.load_state(conn=conn)
        sigs[(sig.internal_name, sig.version)] = sig
        self.dump_state(state=sigs, conn=conn)
        # create relation
        columns = list(sig.input_names) + [
            dump_output_name(index=i) for i in range(sig.n_outputs)
        ]
        columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
        self.rel_storage.create_relation(
            name=sig.versioned_ui_name,
            columns=columns,
            primary_key=Config.uid_col,
            defaults=sig.new_ui_input_default_uids,
            conn=conn,
        )
        logger.debug(f"Created new version:\n{sig}")

    @transaction()
    def update_sig(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Update an existing signature. `sig` must have internal data, and
        must already exist in storage.
        """
        assert sig.has_internal_data
        sigs = self.load_state(conn=conn)
        assert (sig.internal_name, sig.version) in sigs.keys()
        current = sigs[(sig.internal_name, sig.version)]
        # the `update` method also ensures that the signature is compatible
        new_sig, updates = current.update(new=sig)
        # update the signature data
        sigs[(sig.internal_name, sig.version)] = new_sig
        self.dump_state(state=sigs, conn=conn)
        # create new inputs in the database, if any
        for new_input, default_value in updates.items():
            internal_input_name = new_sig.ui_to_internal_input_map[new_input]
            default_uid = new_sig._new_input_defaults_uids[internal_input_name]
            self.rel_storage.create_column(
                relation=new_sig.versioned_ui_name,
                name=new_input,
                default_value=default_uid,
                conn=conn,
            )
            # insert the default in the objects *in the database*, if it's
            # not there already
            default_vref = ValueRef(uid=default_uid, obj=default_value, in_memory=True)
            self.rel_adapter.obj_set(uid=default_uid, value=default_vref, conn=conn)
        if len(updates) > 0:
            logger.debug(
                f"Updated signature:\n    new inputs:{updates} new signature:\n    {sig}"
            )

    @transaction()
    def update_ui_name(
        self,
        sig: Signature,
        conn: Optional[Connection] = None,
        validate_only: bool = False,
    ) -> Dict[Tuple[str, int], Signature]:
        """
        Update a signature's UI name using the given `Signature` object to get
        the new name. `sig` must have internal data, and must carry the new UI
        name.

        NOTE: the `sig` may have the same UI name as the current signature, in
        which case this method does nothing but return the current state of the
        signatures.

        This method has the option of only generating the new state of the
        signatures without performing the update.
        """
        assert sig.has_internal_data
        assert self.exists_internal(sig=sig, conn=conn)
        sigs = self.load_state(conn=conn)
        all_versions = self.get_versions(sig=sig, conn=conn)
        current_ui_name = sigs[(sig.internal_name, all_versions[0])].ui_name
        new_ui_name = sig.ui_name
        if current_ui_name == new_ui_name:
            # nothing to do
            return sigs
        # make sure there are no conflicts
        internal_to_ui = self.internal_to_ui(conn=conn)
        if new_ui_name in internal_to_ui.values():
            raise ValueError(
                f"UI name {new_ui_name} already exists for another signature."
            )
        for version in all_versions:
            current = sigs[(sig.internal_name, version)]
            if current.ui_name != sig.ui_name:
                new_sig = current.rename(new_name=sig.ui_name)
                # update signature object in memory
                sigs[(sig.internal_name, version)] = new_sig
                if not validate_only:
                    # update table
                    self.rel_storage.rename_relation(
                        name=current.versioned_ui_name,
                        new_name=new_sig.versioned_ui_name,
                        conn=conn,
                    )
        if not validate_only:
            # update signatures state
            self.dump_state(state=sigs, conn=conn)
            if current_ui_name != new_ui_name:
                logger.debug(
                    f"Updated UI name of signature: from {current_ui_name} to {new_ui_name}"
                )
        return sigs

    @transaction()
    def update_input_ui_names(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        """
        Update a signature's input UI names from the given `Signature` object.
        `sig` must have internal data, and must carry the new UI input names.
        """
        assert sig.has_internal_data
        sigs = self.load_state(conn=conn)
        current = sigs[(sig.internal_name, sig.version)]
        current_internal_to_ui = current.internal_to_ui_input_map
        new_internal_to_ui = sig.internal_to_ui_input_map
        renaming_map = {
            current_internal_to_ui[k]: new_internal_to_ui[k]
            for k in current_internal_to_ui
            if current_internal_to_ui[k] != new_internal_to_ui[k]
        }
        # update signature object
        new_sig = current.rename_inputs(mapping=renaming_map)
        sigs[(sig.internal_name, sig.version)] = new_sig
        self.dump_state(state=sigs, conn=conn)
        # update table columns
        self.rel_storage.rename_columns(
            relation=new_sig.versioned_ui_name, mapping=renaming_map, conn=conn
        )
        if len(renaming_map) > 0:
            logger.debug(
                f"Updated input UI names of signature named {sig.ui_name}: via mapping {renaming_map}"
            )
        return new_sig

    ############################################################################
    ###
    ############################################################################
    # @transaction()
    # def is_synced(
    #     self, sig: Signature, conn: Optional[Connection] = None
    # ) -> Tuple[bool, str]:
    #     """
    #     Given a signature with internal data, check if it matches what's
    #     in the databse. Returns a tuple of (is_synced, reason if false).
    #     """
    #     assert sig.has_internal_data
    #     current_sigs = self.load_state(conn=conn)
    #     if (sig.internal_name, sig.version) in current_sigs:
    #         current = current_sigs[(sig.internal_name, sig.version)]
    #         if sig != current:
    #             return (
    #                 False,
    #                 f"Signatures are different.\nThis signature:\n    {sig}\nCurrent signature:\n    {current}",
    #             )
    #     else:
    #         return (
    #             False,
    #             f"Signature with this internal name and version not found in database.",
    #         )
    #     return True, "Signature is equal to a signature in the database."

    @transaction()
    def rename_tables(
        self,
        tables: Dict[str, TableType],
        to: str = "internal",
        conn: Optional[Connection] = None,
    ) -> Dict[str, TableType]:
        """
        Rename a dictionary of {versioned name: table} pairs and the tables'
        columns to either internal or UI names.
        """
        result = {}
        assert to in ["internal", "ui"]
        for table_name, table in tables.items():
            if self.is_sig_table_name(
                name=table_name, use_internal=(to != "internal"), conn=conn
            ):
                if to == "internal":
                    ui_name, version = Signature.parse_versioned_name(table_name)
                    sig = self.load_ui_sigs(conn=conn)[ui_name, version]
                    new_table_name = sig.versioned_internal_name
                    mapping = sig.ui_to_internal_input_map
                else:
                    internal_name, version = Signature.parse_versioned_name(table_name)
                    sig = self.load_state(conn=conn)[internal_name, version]
                    new_table_name = sig.versioned_ui_name
                    mapping = sig.internal_to_ui_input_map
                result[new_table_name] = _rename_cols(table=table, mapping=mapping)
            else:
                result[table_name] = table
        return result


class RelAdapter(Transactable):
    """
    Responsible for high-level RDBMS interactions, such as
        - taking a bunch of calls and putting their data inside the database;
        - keeping track of data provenance (i.e., putting calls in a table that
          makes it easier to derive the history of a value reference)

    Uses `RelStorage` to do the actual work.
    """

    EVENT_LOG_TABLE = Config.event_log_table
    VREF_TABLE = Config.vref_table
    SIGNATURES_TABLE = Config.schema_table
    # tables to be excluded from certain operations
    SPECIAL_TABLES = (
        EVENT_LOG_TABLE,
        DuckDBRelStorage.TEMP_ARROW_TABLE,
        SIGNATURES_TABLE,
    )

    def __init__(self, rel_storage: DuckDBRelStorage):
        self.rel_storage = rel_storage
        self.sig_adapter = SigAdapter(rel_adapter=self)
        self.init()
        self.sig_adapter.dump_state(state={})

    @transaction()
    def init(self, conn: Optional[Connection] = None):
        self.rel_storage.create_relation(
            name=self.VREF_TABLE,
            columns=[(Config.uid_col, None), (Config.vref_value_col, "blob")],
            primary_key=Config.uid_col,
            defaults={},
            conn=conn,
        )
        # Initialize the event log.
        # The event log is just a list of UIDs that changed, for now.
        # the UID column stores the vref/call uid, the `table` column stores the
        # table in which this UID is to be found.
        self.rel_storage.create_relation(
            name=self.EVENT_LOG_TABLE,
            columns=[(Config.uid_col, None), ("table", "varchar")],
            primary_key=Config.uid_col,
            defaults={},
            conn=conn,
        )
        # The signatures table is a binary dump of the signatures
        self.rel_storage.create_relation(
            name=self.SIGNATURES_TABLE,
            columns=[
                ("index", "int"),
                ("signatures", "blob"),
            ],
            primary_key="index",
            defaults={},
            conn=conn,
        )

    @transaction()
    def get_call_tables(self, conn: Optional[Connection] = None) -> List[str]:
        tables = self.rel_storage.get_tables(conn=conn)
        return [
            t for t in tables if t not in self.SPECIAL_TABLES and t != self.VREF_TABLE
        ]

    @transaction()
    def evaluate_call_table(
        self, ta: TableType, conn: Optional[Connection] = None
    ) -> pd.DataFrame:
        # if isinstance(ta, pa.Table):
        #     ta = pa.Table.to_pandas(ta)
        ta = ta.copy()
        for col in ta.columns:
            if col != Config.uid_col:
                ta[col] = ta[col].apply(
                    lambda uid: unwrap(self.obj_get(uid, conn=conn))
                )
        return ta

    @transaction()
    def get_vrefs(self, conn: Optional[Connection] = None) -> pd.DataFrame:
        """
        Returns a dataframe of the deserialized values of the value references
        in the storage.
        """
        data = self.rel_storage.get_data(table=self.VREF_TABLE, conn=conn)
        data["value"] = data["value"].apply(lambda vref: unwrap(deserialize(vref)))
        return data

    @transaction()
    def get_all_call_data(
        self, conn: Optional[Connection] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a dictionary of all the memoization tables (labeled by function
        and version name)
        """
        result = {}
        for table in self.get_call_tables(conn=conn):
            result[table] = self.rel_storage.get_data(table=table, conn=conn)
        for k, v in result.items():
            v = self.evaluate_call_table(ta=v, conn=conn)
        return result

    ############################################################################
    ### event log stuff
    ############################################################################
    @transaction()
    def get_event_log(self, conn: Optional[Connection] = None) -> pd.DataFrame:
        return self.rel_storage.get_data(table=self.EVENT_LOG_TABLE, conn=conn)

    # @transaction()
    # def event_log_is_clean(self, conn: Optional[Connection] = None) -> bool:
    #     return self.rel_storage.get_count(table=self.EVENT_LOG_TABLE, conn=conn) == 0

    @transaction()
    def clear_event_log(self, conn: Optional[Connection] = None):
        event_log_table = Table(self.EVENT_LOG_TABLE)
        self.rel_storage.execute_no_results(
            query=Query.from_(event_log_table).delete(), conn=conn
        )

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
    ### call methods
    ############################################################################
    @transaction()
    def _get_current_names(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Given a possibly stale signature `sig`, return
            - the current ui name for this signature
            - a mapping of stale input names to their current values
        """
        current_sigs = self.sig_adapter.load_state()
        current_sig = current_sigs[sig.internal_name, sig.version]
        true_versioned_ui_name = current_sig.versioned_ui_name
        stale_to_true_input_mapping = {
            k: current_sig.internal_to_ui_input_map[v]
            for k, v in sig.ui_to_internal_input_map.items()
            # for k, v in current_sig.ui_to_internal_input_map.items()
        }
        return true_versioned_ui_name, stale_to_true_input_mapping

    @transaction()
    def tabulate_calls(
        self, calls: List[Call], conn: Optional[Connection] = None
    ) -> Dict[str, pa.Table]:
        """
        Converts call objects to a dictionary of {relation name: table
        to upsert} pairs.

        Note that the calls can involve functions in different stages of
        staleness. This method can handle calls to many different variants of
        the same function (adding inputs, renaming the function or its inputs).
        To handle calls to stale functions, this passes through internal names
        to get the current UI names.
        """
        if not len(calls) == len(set([call.uid for call in calls])):
            # something fishy may be going on
            raise InternalError("Calls must have unique UIDs")
        # split by operation *internal* name to group calls to the same op in
        # the same group, even if UI names are different.
        calls_by_op = defaultdict(list)
        for call in calls:
            calls_by_op[call.func_op.sig.versioned_internal_name].append(call)
        res = {}
        for versioned_internal_name, calls in calls_by_op.items():
            rows = []
            true_sig = self.sig_adapter.load_state()[
                Signature.parse_versioned_name(versioned_internal_name)
            ]
            true_versioned_ui_name = None
            for call in calls:
                # it is necessary to process each call individually to properly
                # handle multiple stale variants of this op
                sig = call.func_op.sig
                # get the current state of this signature
                (
                    true_versioned_ui_name,
                    # stale UI input -> current UI input. This could vary across calls
                    stale_to_true_input_mapping,
                ) = self._get_current_names(sig, conn=conn)
                # form the input UIDs
                input_uids = {
                    stale_to_true_input_mapping[k]: v.uid
                    for k, v in call.inputs.items()
                }
                # patch the input uids using the true signature. This is
                # necesary to do here because it seems duckdb has issues with
                # interpreting NaNs from pyarrow as nulls
                for k, v in true_sig.new_ui_input_default_uids.items():
                    if k not in input_uids:
                        input_uids[k] = v
                row = {
                    Config.uid_col: call.uid,
                    **input_uids,
                    **{
                        dump_output_name(index=i): v.uid
                        for i, v in enumerate(call.outputs)
                    },
                }
                rows.append(row)
            assert true_versioned_ui_name is not None
            res[true_versioned_ui_name] = pa.Table.from_pylist(rows)
        return res

    @transaction()
    def upsert_calls(self, calls: List[Call], conn: Optional[Connection] = None):
        """
        Upserts calls in the relational storage so that they will show up in
        declarative queries.
        """
        if len(calls) > 0:  # avoid dealing with empty dataframes
            for table_name, ta in self.tabulate_calls(calls).items():
                self.rel_storage.upsert(relation=table_name, ta=ta, conn=conn)
                # Write changes to the event log table
                self.rel_storage.upsert(
                    relation=self.EVENT_LOG_TABLE,
                    ta=pa.Table.from_pydict(
                        {
                            Config.uid_col: ta[Config.uid_col],
                            "table": [table_name] * len(ta),
                        }
                    ),
                    conn=conn,
                )

    @transaction()
    def _query_call(self, call_uid: str, conn: Optional[Connection] = None) -> pa.Table:
        all_tables = [
            Query.from_(table_name)
            .where(Table(table_name)[Config.uid_col] == Parameter("$1"))
            .select(Table(table_name)[Config.uid_col], LiteralValue(f"'{table_name}'"))
            for table_name in self.get_call_tables()
        ]
        query = sum(all_tables[1:], start=all_tables[0])
        return self.rel_storage.execute_arrow(query, [call_uid], conn=conn)

    @transaction()
    def call_exists(self, call_uid: str, conn: Optional[Connection] = None) -> bool:
        return len(self._query_call(call_uid, conn=conn)) > 0

    @transaction()
    def call_get_lazy(self, call_uid: str, conn: Optional[Connection] = None) -> Call:
        """
        Return the call with the inputs/outputs as lazy value references.
        """
        row = self._query_call(call_uid, conn=conn).take([0])
        table_name = row.column(1)[0]
        table = Table(table_name)
        query = (
            Query.from_(table)
            .where(table[Config.uid_col] == Parameter("$1"))
            .select(table.star)
        )
        results = self.rel_storage.execute_arrow(query, [call_uid], conn=conn)
        # determine the signature for this call
        ui_name, version = Signature.parse_versioned_name(
            versioned_name=str(table_name)
        )
        sig = self.sig_adapter.load_ui_sigs(conn=conn)[ui_name, version]
        return Call.from_row(results, func_op=FuncOp._from_data(f=None, sig=sig))

    ############################################################################
    ### object methods
    ############################################################################
    @transaction()
    def obj_gets(
        self, uids: list[str], conn: Optional[Connection] = None
    ) -> pd.DataFrame:
        if len(uids) == 0:
            return pd.DataFrame()

        table = Table(Config.vref_table)
        query = (
            Query.from_(table)
            .where(table[Config.uid_col].isin(uids))
            .select(table[Config.uid_col], table.value)
        )
        output = self.rel_storage.execute_arrow(query, conn=conn).to_pandas()
        output["value"] = output["value"].map(lambda x: deserialize(bytes(x)))
        return output

    @transaction()
    def obj_exists(
        self, uids: List[str], conn: Optional[Connection] = None
    ) -> List[bool]:
        assert isinstance(uids, list)
        df = self.obj_gets(uids, conn=conn)
        results = [len(df[df[Config.uid_col] == uid]) > 0 for uid in uids]
        return results

    @transaction()
    def obj_get(self, uid: str, conn: Optional[Connection] = None) -> ValueRef:
        df = self.obj_gets(uids=[uid], conn=conn)
        return df.loc[0, "value"]

    @transaction()
    def obj_set(
        self, uid: str, value: ValueRef, conn: Optional[Connection] = None
    ) -> None:
        if self.obj_exists(uids=[uid], conn=conn)[0]:
            return  # guarantees no overwriting of values
        serialized_value = serialize(value)
        query = Query.into(Config.vref_table).insert(Parameter("$1"), Parameter("$2"))
        self.rel_storage.execute_arrow(
            query=query, parameters=[uid, serialized_value], conn=conn
        )
        log_query = Query.into(self.EVENT_LOG_TABLE).insert(
            Parameter("$1"), Parameter("$2")
        )
        self.rel_storage.execute_arrow(
            log_query, parameters=[uid, Config.vref_table], conn=conn
        )

    @transaction()
    def obj_sets(self, kvs: dict[str, Any], conn: Optional[Connection] = None) -> None:
        uids = list(kvs.keys())
        indicators = self.obj_exists(uids, conn=conn)
        new_uids = [uid for uid, indicator in zip(uids, indicators) if not indicator]
        ta = pa.Table.from_pylist(
            [
                {Config.uid_col: new_uid, "value": serialize(kvs[new_uid])}
                for new_uid in new_uids
            ]
        )
        self.rel_storage.upsert(relation=Config.vref_table, ta=ta, conn=conn)
        log_ta = pa.Table.from_pylist(
            [
                {Config.uid_col: new_uid, "table": Config.vref_table}
                for new_uid in new_uids
            ]
        )
        self.rel_storage.upsert(relation=self.EVENT_LOG_TABLE, ta=log_ta, conn=conn)
