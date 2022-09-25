import pyarrow as pa
from duckdb import DuckDBPyConnection as Connection

from ..common_imports import *
from ..core.config import Config, dump_output_name
from ..core.sig import Signature
from .rel_impls.utils import Transactable, transaction
from .rels import RelAdapter, serialize, deserialize
from .remote_storage import RemoteStorage


def _rename_cols_pandas(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping, inplace=False)


def _rename_cols_arrow(table: pa.Table, mapping: Dict[str, str]) -> pa.Table:
    columns = table.column_names
    new_columns = [mapping.get(col, col) for col in columns]
    table = table.rename_columns(new_columns)
    return table


TableType = TypeVar("TableType", pa.Table, pd.DataFrame)


def _rename_cols(table: TableType, mapping: Dict[str, str]) -> TableType:
    if isinstance(table, pd.DataFrame):
        return _rename_cols_pandas(df=table, mapping=mapping)
    elif isinstance(table, pa.Table):
        return _rename_cols_arrow(table=table, mapping=mapping)
    else:
        raise NotImplementedError(f"rename_cols not implemented for {type(table)}")


class SigAdapter(Transactable):
    """
    Responsible for state transitions of the schema that update the
    signature objects *and* the relational tables in a transactional way.

    There are two kinds of updates:
        - updates from the server: these come in bulk when you pull the current
          true state of the signatures. The signature objects come with their
          internal data.
        - updates from the client: these could happen piece by piece, or in bulk
          if using a context manager. The signature objects may not have
          internal data if they come straight from the client's code, so need to
          be matched to the existing signatures.

    This class ensures that all updates are valid against the current state of
    the signatures on the server, and that only successful updates against this
    copy go through to the local storage.
    """

    def __init__(
        self,
        rel_adapter: RelAdapter,
        sigs: Dict[Tuple[str, int], Signature],
        root: Optional[Union[Path, RemoteStorage]] = None,
    ):
        self.rel_adapter = rel_adapter
        self.root = root
        self.rel_storage = self.rel_adapter.rel_storage
        # {(internal name, version): Signature}
        self.sigs = sigs

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
    @property
    def ui_sigs(self) -> Dict[Tuple[str, int], Signature]:
        """
        Get the signatures indexed by UI names
        """
        return {(sig.ui_name, sig.version): sig for sig in self.sigs.values()}

    def exists_ui(self, sig: Signature) -> bool:
        """
        Check if the signature exists based on its UI name
        """
        return (sig.ui_name, sig.version) in self.ui_sigs

    def exists_any_version(self, sig: Signature) -> bool:
        if sig.has_internal_data:
            return any(sig.internal_name == k[0] for k in self.sigs.keys())
        else:
            return any(sig.ui_name == k[0] for k in self.ui_sigs.keys())

    def get_latest_version(self, sig: Signature) -> Signature:
        """
        Get the latest version of the signature, based on its internal name.
        """
        if sig.has_internal_data:
            version = max([k[1] for k in self.sigs.keys() if k[0] == sig.internal_name])
            return self.sigs[(sig.internal_name, version)]
        else:
            version = max([k[1] for k in self.ui_sigs.keys() if k[0] == sig.ui_name])
            return self.ui_sigs[(sig.ui_name, version)]

    def exists_internal(self, sig: Signature) -> bool:
        """
        Check if the signature exists based on its internal name
        """
        return (sig.internal_name, sig.version) in self.sigs

    @property
    def internal_to_ui(self) -> Dict[str, str]:
        """
        Get a mapping from internal names to UI names
        """
        return {k[0]: v.ui_name for k, v in self.sigs.items()}

    @property
    def ui_to_internal(self) -> Dict[str, str]:
        """
        Get a mapping from UI names to internal names
        """
        return {v.ui_name: k[0] for k, v in self.sigs.items()}

    @property
    def ui_names(self) -> Set[str]:
        return set(self.ui_to_internal.keys())

    @property
    def internal_names(self) -> Set[str]:
        return set(self.internal_to_ui.keys())

    def is_sig_table_name(self, name: str) -> bool:
        """
        Check if the name is a valid name for a table corresponding to a
        signature.
        """
        parts = name.split("_", 1)
        return parts[0] in (self.internal_names | self.ui_names) and parts[1].isdigit()

    ############################################################################
    ### elementary transitions
    ############################################################################
    @transaction()
    def dump_state(self, conn: Optional[Connection] = None):
        """
        Dump the current state of the signatures to the database
        """
        # delete existing, if any
        index_col = "index"
        query = f"DELETE FROM {self.rel_adapter.SIGNATURES_TABLE} WHERE {index_col} = 0"
        conn.execute(query)
        # insert new
        serialized = serialize(obj=self.sigs)
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
        Load the state of the signatures from the database
        """
        query = f"SELECT * FROM {self.rel_adapter.SIGNATURES_TABLE} WHERE index = 0"
        df = self.rel_storage.execute_df(query=query, conn=conn)
        if len(df) == 0:
            return {}
        else:
            return deserialize(df["signatures"][0])

    @transaction()
    def create_sig(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Create a new signature. `sig` must have internal data and not be present
        in storage.
        """
        assert sig.has_internal_data
        assert (sig.internal_name, sig.version) not in self.sigs.keys()
        self.sigs[(sig.internal_name, sig.version)] = sig
        # write signatures
        self.dump_state(conn=conn)
        # create relation
        columns = list(sig.input_names) + [
            dump_output_name(index=i) for i in range(sig.n_outputs)
        ]
        columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
        self.rel_storage.create_relation(
            name=sig.versioned_ui_name,
            columns=columns,
            primary_key=Config.uid_col,
            conn=conn,
        )

    @transaction()
    def create_new_version(self, sig: Signature, conn: Optional[Connection] = None):
        latest_sig = self.get_latest_version(sig=sig)
        assert sig.version == latest_sig.version + 1
        # update signatures
        self.sigs[(sig.internal_name, sig.version)] = sig
        self.dump_state(conn=conn)
        # create relation
        columns = list(sig.input_names) + [
            dump_output_name(index=i) for i in range(sig.n_outputs)
        ]
        columns = [(Config.uid_col, None)] + [(column, None) for column in columns]
        self.rel_storage.create_relation(
            name=sig.versioned_ui_name,
            columns=columns,
            primary_key=Config.uid_col,
            conn=conn,
        )

    @transaction()
    def update_sig(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Update an existing signature. `sig` must have internal data, and
        must already exist in storage.
        """
        assert sig.has_internal_data
        assert (sig.internal_name, sig.version) in self.sigs.keys()
        current = self.sigs[(sig.internal_name, sig.version)]
        # the `update` method also ensures that the signature is compatible
        new_sig, updates = current.update(new=sig)
        # update the signature data
        self.sigs[(sig.internal_name, sig.version)] = new_sig
        self.dump_state(conn=conn)
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
            self.rel_adapter.obj_set(uid=default_uid, value=default_value, conn=conn)

    @transaction()
    def update_ui_name(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Update a signature's UI name from the given `Signature` object.
        `sig` must have internal data, and must carry the new UI name.
        """
        assert sig.has_internal_data
        assert self.exists_internal(sig=sig)
        current = self.sigs[(sig.internal_name, sig.version)]
        if current.ui_name != sig.ui_name:
            new_sig = current.rename(new_name=sig.ui_name)
            # update signature object
            self.sigs[(sig.internal_name, sig.version)] = new_sig
            self.dump_state(conn=conn)
            # update table
            self.rel_storage.rename_relation(
                name=current.versioned_ui_name,
                new_name=new_sig.versioned_ui_name,
                conn=conn,
            )

    @transaction()
    def update_input_ui_names(self, sig: Signature, conn: Optional[Connection] = None):
        """
        Update a signature's input UI names from the given `Signature` object.
        `sig` must have internal data, and must carry the new UI input names.
        """
        assert sig.has_internal_data
        current = self.sigs[(sig.internal_name, sig.version)]
        current_internal_to_ui = current.internal_to_ui_input_map
        new_internal_to_ui = sig.internal_to_ui_input_map
        renaming_map = {
            current_internal_to_ui[k]: new_internal_to_ui[k]
            for k in current_internal_to_ui
            if current_internal_to_ui[k] != new_internal_to_ui[k]
        }
        # update signature object
        new_sig = current.rename_inputs(mapping=renaming_map)
        self.sigs[(sig.internal_name, sig.version)] = new_sig
        self.dump_state(conn=conn)
        # update table columns
        self.rel_storage.rename_columns(
            relation=new_sig.versioned_ui_name, mapping=renaming_map, conn=conn
        )

    ############################################################################
    ### sync with server
    ############################################################################
    @property
    def has_remote(self) -> bool:
        return self.root is not None

    def pull_signatures(self) -> List[Signature]:
        """
        Pull the current state of the signatures from the remote, make sure that
        they are compatible with the current ones, and then update or create
        according to the new signatures.
        """
        if isinstance(self.root, RemoteStorage):
            new_sigs = self.root.pull_signatures()
            return new_sigs
        else:
            raise ValueError("No remote storage to pull from.")

    def push_signatures(self, sigs: List[Signature]):
        if isinstance(self.root, RemoteStorage):
            assert isinstance(self.root, RemoteStorage)
            self.root.push_signatures(new_sigs=sigs)

    @transaction()
    def sync_from_remote(self, conn: Optional[Connection] = None):
        """
        Update state from signatures *with internal data* (coming from the
        server). This includes:
            - creating new signatures
            - updating existing signatures
            - renaming functions and inputs
        """
        if self.has_remote:
            sigs = self.pull_signatures()
            for sig in sigs:
                if self.exists_internal(sig=sig):
                    self.update_ui_name(sig=sig, conn=conn)
                    self.update_input_ui_names(sig=sig, conn=conn)
                    self.update_sig(sig=sig, conn=conn)
                else:
                    self.create_sig(sig=sig, conn=conn)

    ############################################################################
    ### sync with client
    ############################################################################
    @transaction()
    def sync_create(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        new_sig = sig._generate_internal()
        all_sigs = list(copy.deepcopy(self.sigs).values())
        all_sigs.append(new_sig)
        self.push_signatures(sigs=all_sigs)
        self.create_sig(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_update(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        current = self.ui_sigs[sig.ui_name, sig.version]
        new_sig, _ = current.update(new=sig)
        all_sigs = copy.deepcopy(self.sigs)
        all_sigs[(current.internal_name, current.version)] = new_sig
        self.push_signatures(sigs=list(all_sigs.values()))
        self.update_sig(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_new_version(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        if not self.exists_any_version(sig=sig):
            raise ValueError()
        latest_sig = self.get_latest_version(sig=sig)
        new_version = latest_sig.version + 1
        if not sig.version == new_version:
            raise ValueError()
        new_sig = sig._generate_internal(internal_name=latest_sig.internal_name)
        all_sigs = copy.deepcopy(self.sigs)
        all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=list(all_sigs.values()))
        self.create_new_version(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_rename_sig(
        self, sig: Signature, new_name: str, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        new_sig = sig.rename(new_name=new_name)
        all_sigs = copy.deepcopy(self.sigs)
        all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=list(all_sigs.values()))
        self.update_ui_name(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_rename_input(
        self,
        sig: Signature,
        input_name: str,
        new_input_name: str,
        conn: Optional[Connection] = None,
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        new_sig = sig.rename_inputs(mapping={input_name: new_input_name})
        all_sigs = copy.deepcopy(self.sigs)
        all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=list(all_sigs.values()))
        self.update_input_ui_names(sig=new_sig, conn=conn)
        return new_sig

    def sync_from_local(self, sig: Signature) -> Signature:
        """
        Create a new signature, create a new version, or update an existing one.
        """
        # todo: versioning
        if self.exists_ui(sig=sig):
            res = self.sync_update(sig=sig)
        elif self.exists_any_version(sig=sig):
            res = self.sync_new_version(sig=sig)
        else:
            res = self.sync_create(sig=sig)
        return res

    ############################################################################
    ### helpers
    ############################################################################
    def rename_tables(
        self, tables: Dict[str, TableType], to: str = "internal"
    ) -> Dict[str, TableType]:
        result = {}
        for table_name, table in tables.items():
            if self.is_sig_table_name(name=table_name):
                if to == "internal":
                    ui_name, version = Signature.parse_versioned_name(table_name)
                    sig = self.ui_sigs[ui_name, version]
                    new_table_name = sig.versioned_internal_name
                    mapping = sig.ui_to_internal_input_map
                elif to == "ui":
                    internal_name, version = Signature.parse_versioned_name(table_name)
                    sig = self.sigs[internal_name, version]
                    new_table_name = sig.versioned_ui_name
                    mapping = sig.internal_to_ui_input_map
                else:
                    raise ValueError()
                result[new_table_name] = _rename_cols(table=table, mapping=mapping)
            else:
                result[table_name] = table
        return result
