import pyarrow as pa
from duckdb import DuckDBPyConnection as Connection

from ..common_imports import *
from ..core.config import Config, dump_output_name
from ..core.sig import Signature
from .rel_impls.utils import Transactable, transaction
from .rels import RelAdapter, serialize, SigAdapter
from .remote_storage import RemoteStorage


class SigSyncer(Transactable):
    """
    Responsible for syncing the local schema and the server.

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
        sig_adapter: SigAdapter,
        root: Optional[Union[Path, RemoteStorage]] = None,
    ):
        self.sig_adapter = sig_adapter
        self.root = root
        self.rel_storage = self.sig_adapter.rel_storage

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
    ### sync with server
    ############################################################################
    @property
    def has_remote(self) -> bool:
        return self.root is not None

    def pull_signatures(self) -> Dict[Tuple[str, int], Signature]:
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

    def push_signatures(self, sigs: Dict[Tuple[str, int], Signature]):
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
        logger.debug("Syncing signatures from remote...")
        if self.has_remote:
            sigs = self.pull_signatures()
            for (internal_name, version), sig in sigs.items():
                logging.debug(f"Processing signature {sig}")
                if self.sig_adapter.exists_internal(sig=sig, conn=conn):
                    sess.d = locals()
                    # first set the ui name to the current one (if necessary)
                    self.sig_adapter.update_ui_name(sig=sig, conn=conn)
                    # then, update the input names too (if necessary)
                    self.sig_adapter.update_input_ui_names(sig=sig, conn=conn)
                    # now, update the (already name-aligned) signature from the new
                    self.sig_adapter.update_sig(sig=sig, conn=conn)
                else:
                    self.sig_adapter.create_sig(sig=sig, conn=conn)

    ############################################################################
    ### atomic operations by the client
    ############################################################################
    def validate_transaction(
        self, new_sig: Signature, all_sigs: Dict[Tuple[str, int], Signature]
    ) -> bool:
        """
        Check that a new signature is compatible with a current state of the
        signatures WITHOUT actually updating the state. This is used to check
        that a transaction is valid before committing it.
        """
        assert new_sig.has_internal_data
        if self.sig_adapter.exists_internal(sig=new_sig):
            current = all_sigs[new_sig.internal_name, new_sig.version]
            compatible, reason_not = current.is_compatible(new=new_sig)
            if compatible:
                return True
            else:
                raise ValueError(reason_not)
        else:
            return True

    @transaction()
    def sync_create(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        new_sig = sig._generate_internal()
        self.validate_transaction(
            new_sig=new_sig, all_sigs=self.sig_adapter.load_state(conn=conn)
        )
        all_sigs = self.sig_adapter.load_state(conn=conn)
        all_sigs[new_sig.internal_name, new_sig.version] = new_sig
        self.push_signatures(sigs=all_sigs)
        self.sig_adapter.create_sig(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_update(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        current = self.sig_adapter.load_ui_sigs(conn=conn)[sig.ui_name, sig.version]
        new_sig, _ = current.update(new=sig)
        self.validate_transaction(
            new_sig=new_sig, all_sigs=self.sig_adapter.load_state(conn=conn)
        )
        all_sigs = self.sig_adapter.load_state(conn=conn)
        all_sigs[(current.internal_name, current.version)] = new_sig
        self.push_signatures(sigs=all_sigs)
        self.sig_adapter.update_sig(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_new_version(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        if not self.sig_adapter.exists_any_version(sig=sig, conn=conn):
            raise ValueError()
        latest_sig = self.sig_adapter.get_latest_version(sig=sig, conn=conn)
        new_version = latest_sig.version + 1
        if not sig.version == new_version:
            raise ValueError(f"New version must be {new_version}, not {sig.version}")
        new_sig = sig._generate_internal(internal_name=latest_sig.internal_name)
        self.validate_transaction(
            new_sig=new_sig, all_sigs=self.sig_adapter.load_state(conn=conn)
        )
        all_sigs = self.sig_adapter.load_state(conn=conn)
        all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=all_sigs)
        self.sig_adapter.create_new_version(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_rename_sig(
        self, sig: Signature, new_name: str, conn: Optional[Connection] = None
    ) -> Signature:
        self.sync_from_remote(conn=conn)
        #! note: we validate before the renaming. Ideally we should have logic
        # to do this for the new signature directly
        # self.validate_transaction(
        #     new_sig=sig, all_sigs=self.sig_adapter.load_state(conn=conn)
        # )
        new_sig = sig.rename(new_name=new_name)
        all_sigs = self.sig_adapter.update_ui_name(
            sig=new_sig, conn=conn, validate_only=True
        )
        # all_sigs = self.sig_adapter.load_state(conn=conn)
        # all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=all_sigs)
        self.sig_adapter.update_ui_name(sig=new_sig, conn=conn)
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
        #! note: we validate before the renaming. Ideally we should have logic
        # to do this for the new signature directly
        self.validate_transaction(
            new_sig=sig, all_sigs=self.sig_adapter.load_state(conn=conn)
        )
        new_sig = sig.rename_inputs(mapping={input_name: new_input_name})
        all_sigs = self.sig_adapter.load_state(conn=conn)
        all_sigs[(new_sig.internal_name, new_sig.version)] = new_sig
        self.push_signatures(sigs=all_sigs)
        self.sig_adapter.update_input_ui_names(sig=new_sig, conn=conn)
        return new_sig

    @transaction()
    def sync_from_local(
        self, sig: Signature, conn: Optional[Connection] = None
    ) -> Signature:
        """
        Create a new signature, create a new version, or update an existing one,
        and immediately send changes to the server.
        """
        if self.sig_adapter.exists_versioned_ui(sig=sig, conn=conn):
            res = self.sync_update(sig=sig, conn=conn)
        elif self.sig_adapter.exists_any_version(sig=sig, conn=conn):
            res = self.sync_new_version(sig=sig, conn=conn)
        else:
            res = self.sync_create(sig=sig, conn=conn)
        return res
