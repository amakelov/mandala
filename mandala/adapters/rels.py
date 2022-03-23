from abc import abstractmethod, ABC
from collections import defaultdict

from sqlalchemy import Column, String, Table, ForeignKey, Boolean, sql, Float, Index
from sqlalchemy.engine.base import Connection, Engine
from sqlalchemy.sql.expression import select, delete

from .vals import BaseValAdapter
from .tps import BaseTypeAdapter, TypeAdapter
from .ops import BaseOpAdapter, OpAdapter, OpUpdates

from ..util.common_ut import invert_dict
from ..common_imports import *
from ..core.tps import Type
from ..core.bases import Operation, ValueRef, Call, unwrap
from ..core.config import CoreConsts, CoreConfig
from ..core.impl import GetItemList, ConstructList, GetKeyDict, ConstructDict, DeconstructList
from ..core.wrap import wrap_as_atom
from ..core.utils import BackwardCompatible
from ..core.exceptions import SynchronizationError
from ..storages.objects import BaseObjLocation, PartitionedObjLocation
from ..storages.relations import RelStorage, RelSpec
from ..storages.call_graph import BaseCallGraphStorage
from ..storages.rel_impl.utils import transaction, get_in_clause_rhs

from ..session import sess

UID_COL = CoreConsts.UID_COL

class Prov(object):
    ### constants for provenance
    # columns
    call_uid = 'call_uid'
    op_name = 'op_name'
    op_version = 'op_version'
    is_super = 'is_super'
    is_input = 'is_input'
    vref_uid = 'vref_uid'
    vref_name = 'vref_name'
    call_start = 'call_start'
    call_end = 'call_end'

    # directions for traversal
    forward = 'forward'
    backward = 'backward'
    both = 'both'

    # methods for backward traversal
    back_shortest = 'shortest'
    back_longest = 'longest'
    back_all = 'all'
    backward_traversal_methods = (back_shortest, back_longest, back_all)

################################################################################
### interfaces
################################################################################
class BaseRelMeta(ABC):
    
    @property
    @abstractmethod
    def op_adapter(self) -> BaseOpAdapter:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def type_adapter(self) -> BaseTypeAdapter:
        raise NotImplementedError()


class BaseRelAdapter(ABC):
    """
    Glue between the core constructs (vrefs, ops, calls, locations, ...) and the
    relational storage. 
    
    Responsible for:
        - synchronization of operations with the relational storage. 
        - assigning tables to vrefs and ops
        - converting vrefs to relations, and relations back to locations
        - inserting calls
    """
    @abstractmethod
    def init(self, first_time:bool):
        raise NotImplementedError()
    
    ############################################################################ 
    ### composition
    ############################################################################ 
    @property
    @abstractmethod
    def rel_storage(self) -> RelStorage:
        raise NotImplementedError()

    @property
    @abstractmethod
    def call_graph_storage(self) -> BaseCallGraphStorage:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def val_adapter(self) -> BaseValAdapter:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def op_adapter(self) -> BaseOpAdapter:
        raise NotImplementedError()

    @property
    @abstractmethod
    def type_adapter(self) -> BaseTypeAdapter:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def rel_meta(self) -> BaseRelMeta:
        raise NotImplementedError()
    
    ############################################################################ 
    ### op interface
    ############################################################################ 
    @abstractmethod
    def get_op_relname(self, op:Operation) -> str:
        raise NotImplementedError()

    @abstractmethod
    def rename_op(self, op:Operation, new_name:str, conn:Connection=None):
        """
        Update op adapter state by renaming *all versions* of this operation.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def rename_op_args(self, op:Operation, mapping:TDict[str, str], conn:Connection=None):
        """
        Rename the inputs of only this version of the operation.
        """
        raise NotImplementedError()

    @abstractmethod
    def synchronize_many(self, ops:TList[Operation], tracking_state:TAny=None,
                         conn:Connection=None) -> TTuple[TList[Operation], 
                                                         TList[OpUpdates],
                                                         TList[Operation],
                                                         TSet[str]]:
        raise NotImplementedError()

    @abstractmethod
    def get_op_rel(self, op:Operation, rename:bool=False,
                   include_builtin_cols:bool=False, conn:Connection=None) -> pd.DataFrame:
        """
        Get the raw table for an op
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_op_locs(self, op:Operation, rename:bool=False,
                    include_builtin_cols:bool=False,
                    conn:Connection=None) -> pd.DataFrame:
        """
        Get a table of locations
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_op_vrefs(self, op:Operation, rename:bool=False, 
                     include_builtin_cols:bool=False,
                     conn:Connection=None) -> pd.DataFrame:
        """
        Get a table of vrefs
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_op_values(self, op:Operation, rename:bool=False,
                      include_builtin_cols:bool=False,
                      conn:Connection=None) -> pd.DataFrame:
        """
        Get a table of unwrapped values
        """
        raise NotImplementedError()
    
    ############################################################################ 
    ### vref interface
    ############################################################################ 
    @property
    @abstractmethod
    def vref_tables(self) -> TList[str]:
        """
        Names of all tables containing vref data
        """
        raise NotImplementedError()

    @abstractmethod
    def get_type_relname(self, tp:Type) -> str:
        """
        Get the relation where values of this type are stored
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def vref_schema(self) -> TList[str]:
        """
        Return the column names for the relations representing value references
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_vref_df_locations(self, vref_relations:pd.DataFrame) -> TList[BaseObjLocation]:
        """
        Given a table where each row is a vref relation, parse the locations of
        these vrefs.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_vref_locs_from_uids(self, vref_uids:TList[str],
                                conn:Connection=None) -> TList[BaseObjLocation]:
        raise NotImplementedError()

    ############################################################################ 
    ### calls interface
    ############################################################################ 
    @abstractmethod
    def calls_to_tables(self, calls:TList[Call], 
                        ops_only:bool=False) -> TDict[str, pd.DataFrame]:
        """
        Convert calls to operations to a dictionary of tables for insertion in
        the relation storage.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def insert_calls(self, calls:TList[Call], conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def upsert_calls(self, calls:TList[Call], conn:Connection=None):
        raise NotImplementedError()
    
    @abstractmethod
    def get_call_uids(self, op:Operation, version:str,
                      conn:Connection=None) -> TList[str]:
        """
        Return all call uids for a given operation and version
        """
        raise NotImplementedError()

    ############################################################################ 
    ### relations
    ############################################################################ 
    @abstractmethod
    def get_engine(self) -> Engine:
        raise NotImplementedError()
    
    ### 
    @abstractmethod
    def get_tableobj(self, name:str) -> Table:
        raise NotImplementedError()

    @property
    def block_subtransactions(self) -> bool:
        return False

    @abstractmethod
    def setup_storage(self, conn:Connection):
        """
        Create the default initial tables and other metadata for the relation
        storage
        """
        raise NotImplementedError()
    
    @abstractmethod
    def update_meta(self, conn:Connection):
        """
        Write dynamic metadata to db
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def op_tables(self) -> TList[str]:
        """
        Names of op tables
        """
        raise NotImplementedError()

    @abstractmethod
    def get_committed_call_uids(self, conn:Connection=None) -> TList[str]:
        """
        All the committed call UIDs
        """
        raise NotImplementedError()
    
    @abstractmethod
    def create_input(self, relname:str, input_name:str):
        """
        Add input to operation
        """
        raise NotImplementedError()
    
    ############################################################################ 
    ### provenance
    ############################################################################ 
    @property
    @abstractmethod
    def track_provenance(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def read_provenance(self, conn:Connection=None) -> pd.DataFrame:
        """
        Read the full provenance table from the db
        """
        raise NotImplementedError()

    @abstractmethod
    def get_vref_prov_neighbors(self, uids:TSet[str], direction:str=Prov.both, 
                                partial:bool=False, conn:Connection=None) -> pd.DataFrame:
        """
        Return a sub-table of the provenance table where the given vref uids
        appear as outputs (direction=backward), inputs (direction=forward) or
        both. 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_call_prov_neighbors(self, uids:TSet[str], direction:str=Prov.both,
                                conn:Connection=None) -> pd.DataFrame:
        """
        Return a sub-table of the provenance table where the given call uids
        appear as outputs (direction=backward), inputs (direction=forward) or
        both. 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_call_dependents(self, call_uids:TSet[str]=None,
                            conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
        """
        Return vref uids and call uids of the dependents of these calls
        (according to the "correct" definition)
        """
        raise NotImplementedError()
        
    # @abstractmethod
    # def dependency_traversal(self, vref_uids:TSet[str]=None,
    #                          call_uids:TSet[str]=None,
    #                          method:str=None,
    #                          conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
    #     """
    #     Return all (call uids, vref uids) that are dependencies of the given
    #     vref uids and call uids
    #     """
    #     raise NotImplementedError()
    
    # @abstractmethod
    # def dependents_traversal(self, vref_uids:TSet[str]=None, call_uids:TSet[str]=None, conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
    #     """
    #     Return all (call uids, vref uids) that depend on the given vref uids and call uids
    #     """
    #     raise NotImplementedError()
    
    ############################################################################ 
    ### maintenance 
    ############################################################################ 
    # @abstractmethod
    # def mexists_in_rels(self, locs:TList[BaseObjLocation]) -> TList[bool]:
    #     """
    #     Check if given locations exist in relational storage.
    #     """
    #     raise NotImplementedError()
    
    @abstractmethod
    def drop_calls(self, op:Operation, call_uids:TList[str], conn:Connection=None):
        """
        Drop the given calls from a single operation's relation
        """
        raise NotImplementedError()
    
    @abstractmethod
    def mis_orphan(self, locs:TList[BaseObjLocation], conn:Connection=None) -> TList[bool]:
        """
        Check if given locations are orphans in relations. 
        NOTE: there is no requirement that these locations exist in object
        storage. 
        """
        raise NotImplementedError()
        
    @abstractmethod
    def drop_op_relation(self, op:Operation, version:str=None, 
                     must_exist:bool=False, conn:Connection=None):
        """
        Drop the table that holds an operation's calls. 
        
        When attempted on a built-in op, this raises an error.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def drop_all_previous_versions(self, op:Operation, conn:Connection=None):
        """
        Drop the relations for all versions of this operation except the current
        one 
        """
        raise NotImplementedError()

    @abstractmethod
    def drop_rel_orphans(self, conn:Connection=None):
        """
        Drop from relations vrefs that are orphaned
        """
        raise NotImplementedError()

    @abstractmethod
    def drop_vrefs(self, locs:TList[BaseObjLocation], conn:Connection=None):
        """
        Drop the given locations from relations. If they are not all orphans,
        the transaction will abort.
        """
        raise NotImplementedError()

    @abstractmethod
    def drop_uncommitted_vrefs(self, conn:Connection=None):
        """
        Find all the uncommitted vrefs in object storage and delete them
        """
        raise NotImplementedError()
    
    ############################################################################ 
    ### 
    ############################################################################ 
    @abstractmethod
    def describe_vrefs(self) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def describe_rels(self) -> TTuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError()

################################################################################
### implementation
################################################################################
class RelMeta(BaseRelMeta):
    
    def __init__(self, op_adapter:BaseOpAdapter, type_adapter:BaseTypeAdapter):
        self._op_adapter = op_adapter
        self._type_adapter = type_adapter
        
    @property
    def op_adapter(self) -> BaseOpAdapter:
        return self._op_adapter
    
    @property
    def type_adapter(self) -> BaseTypeAdapter:
        return self._type_adapter

    
class RelAdapter(BaseRelAdapter):
    VAR_TABLE = '__var__'
    VAR_INDEX = UID_COL
    VAR_PARTITION = CoreConsts.PARTITION_COL
    CALL_UID_COL = UID_COL
    VERSION_COL = '__version__'
    TAG_COL = '__tag__'
    INDEX_COL = UID_COL
    BUILTIN_COLS = (VERSION_COL, TAG_COL, INDEX_COL)
    PROVENANCE_TABLE = '__provenance__'

    def __init__(self, rel_storage:RelStorage, val_adapter:BaseValAdapter, 
                 call_graph_storage:BaseCallGraphStorage, 
                 track_provenance:bool=None):
        self._rel_storage = rel_storage
        self._val_adapter = val_adapter
        self._call_graph_storage = call_graph_storage
        if track_provenance is None:
            track_provenance = CoreConfig.track_provenance
        self._track_provenance = track_provenance

    def init(self, first_time:bool):
        if first_time:
            self.setup_storage()
        else:
            self.val_adapter.set_type_adapter(value=self.rel_meta.type_adapter)
    
    ############################################################################ 
    ### composition
    ############################################################################ 
    @property
    def rel_storage(self) -> RelStorage:
        return self._rel_storage
    
    @property
    def call_graph_storage(self) -> BaseCallGraphStorage:
        return self._call_graph_storage
    
    @property
    def val_adapter(self) -> BaseValAdapter:
        return self._val_adapter
    
    @property
    def rel_meta(self) -> RelMeta:
        return self.rel_storage.meta
    
    @property
    def op_adapter(self) -> BaseOpAdapter:
        return self.rel_meta.op_adapter
    
    @property
    def type_adapter(self) -> BaseTypeAdapter:
        return self.rel_meta.type_adapter

    ############################################################################ 
    ### op interface
    ############################################################################ 
    def _get_op_relname_from_data(self, name:str, version:str) -> str:
        if name == 'DeconstructList':
            name = 'ConstructList'
        return f'{name.lower()}_{version.lower()}'

    def get_op_relname(self, op:Operation) -> str:
        return self._get_op_relname_from_data(name=op.name, version=op.version)
    
    def get_op_spec(self, op:Operation, add_fks:bool=True) -> RelSpec:
        """
        Get a specification for the table of an op that can be used to create
        the relation
        """
        multicol_indices = op.get_multicol_indices()
        cols = op.get_cols()
        vref_cols = op.get_vref_cols()
        primary_key = op.get_primary_key()
        col_objs = []
        for col in cols:
            is_primary = False if primary_key is None else (primary_key == col)
            if add_fks and col in vref_cols:
                var_qtable = self.rel_storage.get_qtable(name=self.VAR_TABLE)
                var_qindex = f'{var_qtable}.{self.VAR_INDEX}'
                col_obj = Column(col, String(32),
                                 ForeignKey(var_qindex, ondelete='CASCADE'),
                                 primary_key=is_primary)
            else:
                col_obj = Column(col, String(32), primary_key=is_primary)
            col_objs.append(col_obj)
        spec = RelSpec(col_objs=col_objs, indices=multicol_indices)
        return spec
    
    def get_synced(self, op:Operation, op_adapter:BaseOpAdapter) -> Operation:
        """
        Return a synchronized *copy* of the operation, which has the proper
        internal data (signature and name) set. 
        """
        assert (op.ui_name, op.version) in op_adapter.sigmaps.keys()
        op = copy.deepcopy(op)
        internal_name = op_adapter.ui_to_name[op.ui_name]
        op.set_name(name=internal_name)
        sigmap = op_adapter.get_sigmap(op=op)
        op.set_sig(sigmap=sigmap)
        return op

    @transaction()
    def create_input_cols(self, op:Operation, new_input_names:TSet[str],
                          conn:Connection=None):
        """
        Initialize input columns for new arguments of an operation.
        
        This includes:
            - hash and write the default values to the object storage
            - pass the uid as default value for the new columns
        """
        op_relname = self.get_op_relname(op=op)
        if len(new_input_names) > 0:
            assert self.rel_storage.has_table(name=op_relname, conn=conn)
            for new_input_name in new_input_names:
                # manually create the default value in storage
                default = op.sig.defaults[new_input_name]
                assert isinstance(default, BackwardCompatible)
                default_value = default.default
                default_type = op.sig.inputs()[new_input_name]
                vref = wrap_as_atom(obj=default_value, tp=default_type, set_uid=True)
                sess.d = locals()
                # put vref in objects
                self.val_adapter.set(vref=vref)
                # put vref in relations
                vref_rel = self.get_vref_relation(vref=vref)
                self.rel_storage.mfast_upsert(relations_dict={self.VAR_TABLE: vref_rel}, conn=conn)
                default_uid = vref.uid
                self.create_input(relname=op_relname, input_name=new_input_name, 
                                  default=default_uid, conn=conn)
    
    @transaction()
    def create_op_rel(self, op:Operation, conn:Connection=None):
        """
        Create relation for this operation 
        """
        op_relname = self.get_op_relname(op=op)
        if not self.rel_storage.has_table(name=op_relname, conn=conn):
            if (not op.sig.is_fixed) and (not isinstance(op, (GetItemList, ConstructList, GetKeyDict, ConstructDict, DeconstructList))):
                pass
            else:
                rel_spec = self.get_op_spec(op=op)
                self.rel_storage.create_relation(name=op_relname, rel_spec=rel_spec, allow_exist=True, conn=conn)
    
    def apply_renamings(self,
                        op_renamings:TDict[str, TTuple[str, str]],
                        input_renamings:TDict[TTuple[str, str], TDict[str, str]], 
                        op_adapter:BaseOpAdapter,
                        tracking_state:TAny=None) -> BaseOpAdapter:
        """
        Return a new op adapter and modify the tracking state to perform
        renaming of ops and inputs.
        
        Inputs:
            - op_renamings: {internal_name: (current ui name, new ui name)}
            - input_renamings: {(internal_name, version): {current input name: new input name}}
        """
        if tracking_state is not None:
            for internal_name, (cur_ui_name, new_ui_name) in op_renamings.items():
                # rename in tracking
                tracking_state.rename_op(internal_name=internal_name,
                                         new_ui_name=new_ui_name)
        # rename simultaneously in adapter
        op_adapter = op_adapter.rename_ops(internal_to_new_ui_name={internal_name: new_ui_name
                                           for internal_name, (_, new_ui_name) in op_renamings.items()})
        for (internal_name, version), rename_dict in input_renamings.items():
            if tracking_state is not None:
                # rename in tracking
                tracking_state.rename_inputs(internal_name=internal_name, rename_dict=rename_dict)
            # rename in adapter
            op_adapter = op_adapter.rename_inputs(internal_name=internal_name, version=version, 
                                                  rename_dict=rename_dict)
        return op_adapter

    @transaction()
    def rename_op(self, op:Operation, new_name:str, conn:Connection=None):
        ui_name = op.ui_name
        internal_name = self.op_adapter.ui_to_name[ui_name]
        op_renamings = {internal_name: (ui_name, new_name)}
        op_adapter = self.apply_renamings(op_renamings=op_renamings, input_renamings={}, 
                                          tracking_state=None, op_adapter=self.op_adapter)
        self.rel_meta._op_adapter = op_adapter
        self.update_meta(conn=conn)
    
    @transaction()
    def rename_op_args(self, op:Operation, mapping:TDict[str, str], conn:Connection=None):
        ui_name, version = op.ui_name, op.version
        internal_name = self.op_adapter.ui_to_name[ui_name]
        input_renamings = {(internal_name, version): mapping}
        op_adapter = self.apply_renamings(op_renamings={}, input_renamings=input_renamings, 
                                          op_adapter=self.op_adapter, tracking_state=None)
        self.rel_meta._op_adapter = op_adapter
        self.update_meta(conn=conn)
    
    @transaction()
    def synchronize_many(self, ops:TList[Operation], 
                         tracking_state:TAny=None,
                         conn:Connection=None) -> TTuple[TList[Operation], TList[OpUpdates], TList[Operation], TSet[str]]:
        """
        Synchronize multiple operations at once, handling the following
        transitions:
            - creating a version-0 op
            - creating a new version of existing op 
                - and automatically incrementing versions of callers, if any
            - updating an op via interface extension

        Returns:
            - synced_ops: list of all synchronized operations
            - op_updates_list: list of OpUpdate objects for existing ops
            - created_ops_list: list of created operations
            - autoupdated_superops: set of UI names for ops who got their
              version bumped (by 1 or more!)
        """
        op_adapter = self.op_adapter
        ### perform renaming if we have tracking 
        if tracking_state is not None:
            raise NotImplementedError()
            # op_renamings, input_renamings = self.get_renamings(ops=ops, tracking_state=tracking_state)
            # ### RENAME
            # op_adapter = self.apply_renamings(op_renamings=op_renamings, input_renamings=input_renamings, 
            #                                 tracking_state=tracking_state, op_adapter=op_adapter)
        ### now op adapter and tracking state have the renamings implemented, and
        ### we can synchronize operations as usual
        synced_ops = []
        op_updates_list = []
        created_ops_list = []
        autoupdated_superops = []
        for op in ops:
            if op_adapter.has_op(ui_name=op.ui_name, version=op.version):
                internal_name = op_adapter.ui_to_name[op.ui_name]
                latest_version = op_adapter.get_latest_version(internal_name=internal_name)
                if op.version != latest_version:
                    msg = f'Operation: {op.ui_name}, version: {op.version}, latest version: {latest_version}'
                    raise SynchronizationError(f'Synchronizing an old version of an operation is not allowed:\n{msg}')
                ### UPDATE
                # update op adapter
                op_adapter, new_inputs, op_updates = op_adapter.update_op(new_op=op)
                # update tracking
                if tracking_state is not None:
                    raise NotImplementedError()
                    # sigmap = op_adapter.get_sigmap(op=op)
                    # tracking_state.update_op(internal_name=internal_name, input_ui_to_internal=sigmap.fixed_inputs_map())
                synced_op = self.get_synced(op=op, op_adapter=op_adapter)
                self.create_input_cols(op=synced_op, new_input_names=new_inputs, conn=conn)
                synced_ops.append(synced_op)
                op_updates_list.append(op_updates)
            else:
                ### CREATE
                if op_adapter.has_ui_name(op.ui_name):
                    # NEW VERSION OF EXISTING OP
                    # figure out if we need to bump the versions of superops that
                    # call the old version of this op
                    internal_name = op_adapter.ui_to_name[op.ui_name]
                    current_version = op_adapter.get_latest_version(internal_name=internal_name)
                    new_version = op.version
                    if not int(new_version) == int(current_version) + 1:
                        raise SynchronizationError(f'Versions must increase by 1 each time')
                    current_node = Operation.qualified_name_from_data(internal_name=internal_name, version=current_version)
                    caller_nodes = self.call_graph_storage.get_callers(node=current_node)
                    caller_internal_names = [elt.rsplit(sep='_', maxsplit=1)[0] for elt in caller_nodes]
                    caller_ui_names = [op_adapter.internal_to_ui[elt] for elt in caller_internal_names]
                else:
                    # NEW OP
                    caller_ui_names = []
                op_adapter, internal_name, version_changes = op_adapter.create_op(op=op, caller_ui_names=caller_ui_names)
                if caller_ui_names:
                    version_change_lines = '\n'.join([f'{ui_name}: {cur_version} ---> {new_version}'
                                            for ui_name, (cur_version, new_version) in version_changes.items()])
                    logging.info(f'DURING VERSION CHANGE OF OPERATION {op.ui_name}:\n'
                                 'Automatically created new versions of operations using the old version:\n'
                                 f'{version_change_lines}')
                    autoupdated_superops += caller_ui_names
                # update tracking
                if tracking_state is not None:
                    raise NotImplementedError()
                    # sigmap = op_adapter.get_sigmap(op=op)
                    # tracking_state.create_op(internal_name=internal_name, 
                    #                         ui_name=op.ui_name, version=op.version, 
                    #                         input_ui_to_internal=sigmap.fixed_inputs_map())
                synced_op = self.get_synced(op=op, op_adapter=op_adapter)
                self.create_op_rel(op=synced_op, conn=conn)
                synced_ops.append(synced_op)
                created_ops_list.append(op)
        self.rel_meta._op_adapter = op_adapter
        self.update_meta(conn=conn)
        if tracking_state is not None:
            # dump new tracking state at the end of the transaction
            raise NotImplementedError()
            # tracking_state.dump()
        return synced_ops, op_updates_list, created_ops_list, set(autoupdated_superops)
    
    ############################################################################ 
    ### reading op relations in various ways
    ############################################################################ 
    def _rename_df_to_ui(self, op:Operation, df:pd.DataFrame):
        """
        Replace internal column names with ui names in op table
        """
        sigmap = self.op_adapter.get_sigmap(op=op).inverse()
        col_to_ui = {**sigmap.fixed_inputs_map(), **sigmap.fixed_outputs_map()}
        df.rename(columns=col_to_ui, inplace=True)

    @transaction()
    def get_op_rel(self, op:Operation, conn:Connection=None, rename:bool=False,
                   include_builtin_cols:bool=False) -> pd.DataFrame:
        relname = self.get_op_relname(op=op)
        tableobj = self.rel_storage.get_tableobj(name=relname)
        query = select(tableobj)
        df = self.rel_storage.fast_select(query=query, conn=conn)
        if rename:
            self._rename_df_to_ui(op=op, df=df)
        if not include_builtin_cols:
            remaining_cols = [col for col in df.columns 
                              if col not in self.BUILTIN_COLS]
            df = df[remaining_cols]
        return df
    
    @transaction()
    def get_op_locs(self, op:Operation, conn:Connection=None, 
                    rename:bool=False, include_builtin_cols:bool=False) -> pd.DataFrame:
        op_rel = self.get_op_rel(op=op, conn=conn, rename=False,
                                 include_builtin_cols=include_builtin_cols)
        for col in op.get_vref_cols():
            uids = op_rel[col].values.tolist()
            vref_df = self.select_vref_rels_by_uid(uids=uids, conn=conn)
            vref_locs = self.get_vref_df_locations(vref_relations=vref_df)
            op_rel[col] = vref_locs
        if rename:
            self._rename_df_to_ui(op=op, df=op_rel)
        return op_rel
    
    @transaction()
    def get_op_vrefs(self, op:Operation, conn:Connection=None,
                     rename:bool=False, include_builtin_cols:bool=False) -> pd.DataFrame:
        res_df = self.get_op_locs(op=op, conn=conn, rename=False,
                                  include_builtin_cols=include_builtin_cols)
        for col in op.get_vref_cols():
            col_values = self.val_adapter.mget(locs=res_df[col].values.tolist())
            res_df[col] = col_values
        if rename:
            self._rename_df_to_ui(op=op, df=res_df)
        return res_df
    
    @transaction()
    def get_op_values(self, op:Operation, conn:Connection=None, 
                      rename:bool=False, include_builtin_cols:bool=False) -> pd.DataFrame:
        res_df = self.get_op_vrefs(op=op, conn=conn, rename=False,
                                   include_builtin_cols=include_builtin_cols)
        for col in op.get_vref_cols():
            col_values = [unwrap(elt) for elt in res_df[col].values]
            res_df[col] = col_values
        if rename:
            self._rename_df_to_ui(op=op, df=res_df)
        return res_df
        
    ############################################################################ 
    ### val interface
    ############################################################################ 
    ### interface implementation
    def get_vref_relname(self, vref:ValueRef) -> str:
        return self.VAR_TABLE
    
    @property
    def vref_tables(self) -> TList[str]:
        return [self.VAR_TABLE]
    
    def get_type_relname(self, tp:Type) -> str:
        return self.VAR_TABLE
    
    @property
    def vref_schema(self) -> TList[str]:
        return [self.VAR_INDEX, self.VAR_PARTITION]
    
    def get_vref_relation(self, vref:ValueRef) -> pd.DataFrame:
        return pd.DataFrame({self.VAR_INDEX: [vref.uid], 
                             self.VAR_PARTITION: [self.val_adapter.get_vref_dest_partition(vref=vref)]})
    
    def get_vref_df_locations(self, vref_relations:pd.DataFrame) -> TList[BaseObjLocation]:
        return [PartitionedObjLocation(uid=uid, partition=partition)
                for uid, partition in vref_relations[[self.VAR_INDEX, self.VAR_PARTITION]].itertuples(index=False)]
            
    def get_vref_locs_from_uids(self, vref_uids:TList[str], conn:Connection=None) -> TList[BaseObjLocation]:
        tableobj = self.get_var_tableobj()
        query = select(tableobj).where(tableobj.c[self.VAR_INDEX].in_(get_in_clause_rhs(index_like=vref_uids)))
        df = self.rel_storage.fast_select(query=query, conn=conn)
        mapping = dict(zip(df[self.VAR_INDEX], df[self.VAR_PARTITION]))
        locs = [PartitionedObjLocation(uid=uid, partition=mapping[uid]) for uid in vref_uids]
        return locs
    
    def get_var_tableobj(self) -> Table:
        return self.get_tableobj(name=self.VAR_TABLE)
    
    @transaction()
    def select_vref_rels_by_uid(self, uids:TList[str], conn:Connection=None) -> pd.DataFrame:
        """
        Given a list of vref UIDs, return stacked relations for these vrefs 
        **in the same order** as `uids`.
        """
        tableobj = self.get_var_tableobj()
        if not uids:
            query = tableobj.select().where(False)
        else:
            query = tableobj.select().where(tableobj.c[self.VAR_INDEX].in_(get_in_clause_rhs(uids)))
        df = self.rel_storage.fast_select(query=query, conn=conn)
        df = df.set_index(self.VAR_INDEX).reindex(uids).reset_index()
        return df
    
    ############################################################################ 
    ### calls interface
    ############################################################################ 
    def calls_to_tables(self, calls: TList[Call], ops_only:bool=False) -> TDict[str, pd.DataFrame]:
        pre_res = defaultdict(list)
        iterator = calls if len(calls) < 1000 else tqdm.tqdm(calls, desc='Converting calls to tables...')
        for call in iterator:
            # gather call relations
            call_df = call.get_relation()
            call_relname = self.get_op_relname(op=call.op)
            pre_res[call_relname].append(call_df)
            if not ops_only:
                # gather value relations 
                for vref in itertools.chain(call.inputs.values(), call.outputs.values()):
                    vref_relname = self.get_vref_relname(vref=vref)
                    pre_res[vref_relname].append(self.get_vref_relation(vref=vref))
        res:TDict[str, pd.DataFrame] = {k: pd.concat(v) for k, v in pre_res.items()}
        for _, df in res.items():
            df.drop_duplicates(inplace=True)
            df.index = pd.RangeIndex(df.shape[0])
        return res
    
    @transaction()
    def insert_calls(self, calls: TList[Call], conn:Connection=None):
        tables = self.calls_to_tables(calls=calls)
        self.rel_storage.mfast_insert(relations_dict=tables, conn=conn)
        # provenance 
        if self.track_provenance:
            provenance_df = self.mget_provenance_table(calls=calls)
            self.rel_storage.mfast_insert(relations_dict={self.PROVENANCE_TABLE: provenance_df},
                                        conn=conn)
            
    @transaction()
    def upsert_calls(self, calls: TList[Call], conn:Connection=None):
        tables = self.calls_to_tables(calls=calls)
        self.rel_storage.mfast_upsert(relations_dict=tables, conn=conn)
        # provenance 
        if self.track_provenance:
            provenance_df = self.mget_provenance_table(calls=calls)
            self.rel_storage.mfast_upsert(relations_dict={self.PROVENANCE_TABLE: provenance_df},
                                        conn=conn)
    
    @transaction()
    def get_call_uids(self, op:Operation, version:str=None, conn:Connection=None) -> TList[str]:
        if version is None:
            version = op.version
        relname = self.get_op_relname(op=op)
        tableobj = self.rel_storage.get_tableobj(name=relname)
        query = select([tableobj.c[self.CALL_UID_COL]])
        if self.rel_storage.has_column(name=relname, col=self.VERSION_COL):
            query = query.where(tableobj.c[self.VERSION_COL] == version)
        df = self.rel_storage.fast_select(query=query, conn=conn)
        uids = df[self.CALL_UID_COL].values.tolist()
        return uids
    
    def get_provenance_table(self, call:Call) -> pd.DataFrame:
        """
        Return input_provenance, output_provenance tables for a single call
        """
        assert call.exec_interval is not None
        call_start, call_end = call.exec_interval
        call_uid = call.uid
        op_name = call.op.name
        op_version = call.op.version
        input_names = list(call.inputs.keys())
        input_uids = [call.inputs[k].uid for k in input_names]
        in_table = pd.DataFrame({Prov.call_uid: call_uid,
                                 Prov.op_name: op_name,
                                 Prov.op_version: op_version,
                                 Prov.is_super: call.op.is_super,
                                 Prov.vref_name: input_names,
                                 Prov.vref_uid: input_uids,
                                 Prov.is_input: True, 
                                 Prov.call_start: call_start,
                                 Prov.call_end: call_end})
        output_names = list(call.outputs.keys())
        output_uids = [call.outputs[k].uid for k in output_names]
        out_table = pd.DataFrame({Prov.call_uid: call_uid,
                                 Prov.op_name: op_name,
                                 Prov.op_version: op_version,
                                 Prov.is_super: call.op.is_super,
                                 Prov.vref_name: output_names,
                                 Prov.vref_uid: output_uids,
                                 Prov.is_input: False, 
                                 Prov.call_start: call_start,
                                 Prov.call_end: call_end})
        return pd.concat([in_table, out_table], ignore_index=True)
    
    def mget_provenance_table(self, calls:TList[Call]) -> pd.DataFrame:
        tables = []
        for call in calls:
            tables.append(self.get_provenance_table(call=call))
        return pd.concat(tables, ignore_index=True)
            
    ############################################################################ 
    ### relations
    ############################################################################ 
    @transaction()
    def setup_storage(self, conn:Connection=None):
        rel_meta = RelMeta(op_adapter=OpAdapter(), type_adapter=TypeAdapter())
        self.rel_storage.update_meta(value=rel_meta, conn=conn)
        self.val_adapter.set_type_adapter(value=rel_meta.type_adapter)

        var_spec = RelSpec(col_objs=[
            Column(self.VAR_INDEX, String(32), primary_key=True),
            Column(self.VAR_PARTITION, String(32))
        ],
            indices=[], extend_existing=True)
        self.rel_storage.create_relation(name=self.VAR_TABLE,
                                         rel_spec=var_spec,
                                         allow_exist=True, conn=conn)
        
        ### provenance table
        if self.track_provenance:
            #! note that dtypes are not supported by some queries
            var_qtable = self.rel_storage.get_qtable(name=self.VAR_TABLE)
            var_qindex = f'{var_qtable}.{self.VAR_INDEX}'
            provenance_spec = RelSpec(col_objs=[
                Column(Prov.call_uid, String(32),), 
                Column(Prov.op_name, String(40),), 
                Column(Prov.op_version, String(32), ),
                Column(Prov.is_super, Boolean()),
                Column(Prov.vref_name, String(40), ),
                Column(Prov.vref_uid, String(32), ForeignKey(var_qindex, ondelete='CASCADE')), 
                Column(Prov.is_input, Boolean(), ),
                Column(Prov.call_start, Float(), ),
                Column(Prov.call_end, Float(), ),
            ],
                                    indices=[[Prov.call_uid, Prov.vref_name, Prov.is_input]],
                                    extend_existing=True)
            self.rel_storage.create_relation(name=self.PROVENANCE_TABLE, rel_spec=provenance_spec, allow_exist=True, conn=conn)

        if CoreConfig.decompose_struct_as_many:
            builtin_ops = [GetItemList(), ConstructList(), GetKeyDict(), ConstructDict()]
        else:
            builtin_ops = [ConstructList(), GetKeyDict(), ConstructDict(), DeconstructList()]
        for op in builtin_ops:
            self.synchronize_many(ops=[op], conn=conn)
        
    @property
    def op_tables(self) -> TList[str]:
        excluded = (self.VAR_TABLE, self.PROVENANCE_TABLE)
        return [t for t in self.rel_storage.tables() if t not in excluded]
    
    @transaction()
    def get_committed_call_uids(self, conn:Connection=None) -> TList[str]:
        #! TODO: this is inefficient
        res = []
        for relname in self.op_tables:
            res += self.rel_storage.fast_read(name=relname, cols=[self.CALL_UID_COL], 
                                              conn=conn)[self.CALL_UID_COL].values.tolist()
        return res
    
    def get_engine(self) -> Engine:
        return self.rel_storage.get_engine()
    
    def update_meta(self, conn: Connection):
        rel_meta = RelMeta(op_adapter=self.op_adapter, 
                           type_adapter=self.val_adapter.type_adapter)
        self.rel_storage.update_meta(value=rel_meta, conn=conn)
    
    def get_tableobj(self, name: str) -> Table:
        return self.rel_storage.get_tableobj(name=name)
    
    @transaction()
    def create_input(self, relname:str, input_name:str, default:str=None, conn:Connection=None):
        rs = self.rel_storage
        rs.create_column(qtable=rs.get_qtable(name=relname), 
                         name=input_name, dtype='VARCHAR(32)', 
                         with_default=(default is not None),
                         default_value=default,
                         fk_qtable=rs.get_qtable(self.VAR_TABLE),
                         fk_col=self.VAR_INDEX,
                         conn=conn)

    ############################################################################ 
    ### provenance
    ############################################################################ 
    def _fix_provenance_dtypes(self, df:pd.DataFrame) -> pd.DataFrame:
        # for some reason this is necessary
        bool_mapper = {
            'f': False,
            't': True,
            'False': False,
            'True': True
        }
        df[Prov.is_input] = df[Prov.is_input].apply(lambda x: bool_mapper[x])
        df[Prov.is_super] = df[Prov.is_super].apply(lambda x: bool_mapper[x])
        return df
    
    @property
    def track_provenance(self) -> bool:
        return self._track_provenance

    @transaction()
    def read_provenance(self, conn:Connection=None) -> pd.DataFrame:
        df = self.rel_storage.fast_read(self.PROVENANCE_TABLE, conn=conn)
        return self._fix_provenance_dtypes(df=df)

    @transaction()
    def get_vref_prov_neighbors(self, vref_uids:TSet[str], direction:str=Prov.both, 
                     partial:bool=True, conn:Connection=None) -> pd.DataFrame:
        rs = self.rel_storage
        tableobj = rs.get_tableobj(name=self.PROVENANCE_TABLE)
        if direction == Prov.both:
            condition = tableobj.c[Prov.vref_uid].in_(get_in_clause_rhs(vref_uids))
        else:
            if direction == Prov.backward:
                is_input = False # go from output back to call
            elif direction == Prov.forward:
                is_input = True # go from input to call
            else:
                raise NotImplementedError()
            condition = sql.and_(tableobj.c[Prov.vref_uid].in_(get_in_clause_rhs(vref_uids)), 
                                tableobj.c[Prov.is_input] == is_input)
        query = select(tableobj).where(condition)
        partial_result = rs.fast_select(query=query, conn=conn)
        if partial:
            res = partial_result
        else:
            call_uids = list(partial_result[Prov.call_uid])
            query = select(tableobj).where(tableobj.c[Prov.call_uid].in_(get_in_clause_rhs(call_uids)))
            res = rs.fast_select(query=query, conn=conn)
        return self._fix_provenance_dtypes(df=res)

    @transaction()
    def get_call_prov_neighbors(self, call_uids: TSet[str], direction: str = Prov.both, conn: Connection = None) -> pd.DataFrame:
        rs = self.rel_storage
        tableobj = rs.get_tableobj(name=self.PROVENANCE_TABLE)
        if direction == Prov.both:
            condition = tableobj.c[Prov.call_uid].in_(get_in_clause_rhs(call_uids))
        else:
            if direction == Prov.backward:
                is_input = True # go from call back to inputs
            elif direction == Prov.forward:
                is_input = False # go from call to outputs
            else:
                raise NotImplementedError()
            condition = sql.and_(tableobj.c[Prov.call_uid].in_(get_in_clause_rhs(call_uids)), 
                                tableobj.c[Prov.is_input] == is_input)
        query = select(tableobj).where(condition)
        return self._fix_provenance_dtypes(df=rs.fast_select(query=query, conn=conn))
    
    def _filter_vref_backward_neighbors(self, prov_df:pd.DataFrame, method:str) -> pd.DataFrame:
        """
        Given a "backward" provenance table of rows that generate some vrefs,
        filter the operations generating these vrefs by keeping only some of
        them:
            - if method='all', keep all calls
            - if method='highest', keep only highest-level operation calls
            - if method='lowest', keep only lowest-level operation calls
        """
        if method == Prov.back_all:
            return prov_df
        elif method == Prov.back_shortest:
            return prov_df[prov_df.groupby(Prov.vref_uid)[Prov.call_end].transform(max) == prov_df[Prov.call_end]]
        elif method == Prov.back_longest:
            return prov_df[prov_df.groupby(Prov.vref_uid)[Prov.call_end].transform(min) == prov_df[Prov.call_end]]
        else:
            raise NotImplementedError()
        
    @transaction()
    def get_call_dependents(self, call_uids:TSet[str]=None, conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
        U = set()
        H = call_uids.copy()
        call_uid_frontier = H.copy() # the newly discovered calls
        done = False
        while not done:
            # add outputs of *computational* ops to U
            if not call_uid_frontier:
                done = True
                break
            call_df = self.get_call_prov_neighbors(call_uids=call_uid_frontier, direction=Prov.forward, conn=conn)
            call_df = call_df[~call_df[Prov.is_super]]
            new_vref_uids = set(call_df[Prov.vref_uid])
            U = U | new_vref_uids
            # add adjacent calls 
            vref_df = self.get_vref_prov_neighbors(vref_uids=new_vref_uids, 
                                                   direction=Prov.both, conn=conn)
            adjacent_call_uids = set(vref_df[Prov.call_uid])
            call_uid_frontier = adjacent_call_uids - H
            H = H | call_uid_frontier
            if (not new_vref_uids) and (not call_uid_frontier):
                done = True
        return U, H

    @transaction()
    def _provenance_traversal(self, vref_uids:TSet[str]=None, call_uids:TSet[str]=None,
                              direction:str=Prov.backward, 
                              backward_method:str=None, conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
        """
        A general-purpose function to traverse the provenance graph. Starting
        from a collection of values and calls, it can:
            - traverse either backward, forward or in both directions. 
            - when traversing backwards, customize which calls are included when
            there are multiple calls that generate the same value reference:
                - looking at highest-level superops only (backward_method='highest')
                - looking at lowest-level ops only (backward_method='lowest')
                - including all calls (backward_method='all')

        Args:
            vref_uids (TSet[str], optional): [description]. Defaults to None.
            call_uids (TSet[str], optional): [description]. Defaults to None.
            direction (str, optional): [description]. Defaults to Prov.backward.
            minimize_depth (bool, optional): [description]. Defaults to False.
            conn (Connection, optional): [description]. Defaults to None.

        Returns:
            TTuple[TSet[str], TSet[str]]: [description]
        """
        assert direction in ('forward', 'backward')
        if backward_method is not None:
            assert direction == 'backward'
            assert backward_method in Prov.backward_traversal_methods
        all_vref_uids = set() if vref_uids is None else vref_uids
        all_call_uids = set() if call_uids is None else call_uids
        vref_frontier = all_vref_uids.copy()
        call_frontier = all_call_uids.copy()
        done = False
        while not done:
            total_vrefs = len(all_vref_uids)
            total_calls = len(all_call_uids)
            logging.debug(f'Traversing provenance with direction={direction}...\nTotal vrefs: {total_vrefs}, total calls: {total_calls}')
            # current vrefs generate new calls
            vref_df = self.get_vref_prov_neighbors(vref_uids=vref_frontier, 
                                                   direction=direction,
                                                   conn=conn)
            if backward_method is not None:
                vref_df = self._filter_vref_backward_neighbors(prov_df=vref_df, method=backward_method)
            new_call_uids = set(vref_df[Prov.call_uid])
            # current calls generate new vrefs
            call_df = self.get_call_prov_neighbors(call_uids=call_frontier,
                                                   direction=direction, 
                                                   conn=conn)
            new_vref_uids = set(call_df[Prov.vref_uid])
            all_vref_uids |= new_vref_uids
            all_call_uids |= new_call_uids
            vref_frontier = new_vref_uids
            call_frontier = new_call_uids
            if (len(all_vref_uids) == total_vrefs) and (len(all_call_uids) == total_calls):
                done = True
        return all_call_uids, all_vref_uids
    
    @transaction()
    def dependency_traversal(self, vref_uids:TSet[str]=None, 
                             call_uids:TSet[str]=None, method:str=Prov.back_all,
                             conn:Connection=None) -> TTuple[TSet[str], TSet[str]]:
        return self._provenance_traversal(vref_uids=vref_uids, call_uids=call_uids, 
                                          backward_method=method,
                                          direction=Prov.backward, conn=conn)
    
    @transaction()
    def dependents_traversal(self, vref_uids: TSet[str] = None, call_uids: TSet[str] = None, conn: Connection = None) -> TTuple[TSet[str], TSet[str]]:
        return self._provenance_traversal(vref_uids=vref_uids, call_uids=call_uids, 
                                          direction=Prov.forward, conn=conn)
    
    ############################################################################ 
    ### maintenance
    ############################################################################ 
    # @transaction()
    # def mexists_in_rels(self, locs:TList[BaseObjLocation], conn:Connection=None) -> TList[bool]:
    #     unique_uids = set([loc.uid for loc in locs])
    #     var_tableobj = self.rel_storage.get_tableobj(name=self.VAR_TABLE)
    #     query = select([var_tableobj.c[self.VAR_INDEX]]).where(var_tableobj.c[self.VAR_INDEX].in_(get_in_clause_rhs(unique_uids)))
    #     df = self.rel_storage.fast_select(query=query, conn=conn)
    #     existing_uids = set(df[self.VAR_INDEX].values.tolist())
    #     return [loc.uid in existing_uids for loc in locs]
    
    @transaction()
    def drop_calls(self, op:Operation, call_uids:TList[str], conn:Connection=None):
        relname = self.get_op_relname(op=op)
        self.rel_storage.delete_rows(name=relname, index=call_uids,
                                     index_col=self.CALL_UID_COL, conn=conn)
        # delete from provenance
        if self.track_provenance:
            self.rel_storage.delete_rows(name=self.PROVENANCE_TABLE, 
                                        index=call_uids, index_col=Prov.call_uid, 
                                        conn=conn)
    
    @transaction()
    def mis_orphan(self, locs:TList[BaseObjLocation], conn:Connection=None) -> TList[bool]:
        if not locs:
            return []
        rel_st = self.rel_storage
        var_name = self.VAR_TABLE
        nxg = rel_st.get_nx_graph()
        unique_uids = set([loc.uid for loc in locs])
        non_orphan_uids = set()
        for stable, ttable, (scol, tcol) in nxg.edges:
            if ttable == var_name and tcol == self.VAR_INDEX:
                stable_obj = rel_st.get_tableobj(name=stable)
                col_obj = stable_obj.c[scol]
                query = select(col_obj).where(col_obj.in_(get_in_clause_rhs(unique_uids)))
                new_uids = rel_st.fast_select(query=query, conn=conn)[scol].values.tolist()
                non_orphan_uids = non_orphan_uids.union(new_uids)
        return [loc.uid not in non_orphan_uids for loc in locs]

    @transaction()
    def drop_op_relation(self, op:Operation, version:str=None, must_exist:bool=False, conn:Connection=None):
        if op.is_builtin:
            raise ValueError(f'Cannot delete built-in operation')
        if version is None:
            version = op.version
        relname = self._get_op_relname_from_data(name=op.name, version=version)
        if self.rel_storage.has_table(name=relname, conn=conn):
            self.rel_storage.drop_relation(name=relname, conn=conn)
            new_adapter = self.op_adapter.drop_version(ui_name=op.ui_name,
                                                       version=op.version)
            self.rel_meta._op_adapter = new_adapter
            self.update_meta(conn=conn)
        else:
            if must_exist:
                raise ValueError()
    
    @transaction()
    def drop_all_previous_versions(self, op:Operation, conn:Connection=None):
        current = op.version
        previous = [str(i) for i in range(int(current))]
        for version in previous:
            self.drop_op_relation(op=op, version=version, must_exist=False, conn=conn)
        
    @transaction()
    def drop_rel_orphans(self, conn: Connection = None):
        rel_st = self.rel_storage
        var_name = self.VAR_TABLE
        nxg = rel_st.get_nx_graph()
        non_orphan_uids = set()
        for stable, ttable, (scol, tcol) in nxg.edges:
            if ttable == var_name and tcol == self.VAR_INDEX:
                uids = rel_st.fast_read(name=stable, cols=[scol], conn=conn)[scol].values.tolist()
                non_orphan_uids = non_orphan_uids.union(uids)
        logging.info(f'Found {len(non_orphan_uids)} non-orphans')
        var_tableobj = rel_st.get_tableobj(name=var_name)
        query = var_tableobj.delete().where(var_tableobj.c[self.VAR_INDEX].notin_(non_orphan_uids))
        conn.execute(query)
    
    @transaction()
    def drop_vrefs(self, locs:TList[BaseObjLocation], conn:Connection=None):
        uids = [loc.uid for loc in locs]
        table_obj = self.rel_storage.get_tableobj(name=self.VAR_TABLE)
        query = delete(table_obj).where(table_obj.c[self.VAR_INDEX].in_(get_in_clause_rhs(uids)))
        conn.execute(query)
    
    @transaction()
    def drop_uncommitted_vrefs(self, conn:Connection=None):
        obj_st = self.val_adapter.obj_storage
        committed_uids = set(self.rel_storage.fast_read(name=self.VAR_TABLE, 
                                                        cols=[self.VAR_INDEX], conn=conn)[self.VAR_INDEX].values.tolist())
        all_locs = obj_st.locs()
        all_uids = set([loc.uid for loc in all_locs])
        uncommitted_uids = all_uids.difference(committed_uids)
        uncommitted_locs = [loc for loc in all_locs if loc.uid in uncommitted_uids]
        logging.info(f'Dropping {len(uncommitted_uids)} uncommitted vrefs...')
        obj_st.mdelete(locs=uncommitted_locs)
    
    ############################################################################ 
    ### 
    ############################################################################ 
    def describe_vrefs(self) -> pd.DataFrame:
        type_dict = self.type_adapter.type_dict
        type_name_to_partition = {ui_name: self.val_adapter.get_concrete_tp_partition(tp=tp)
                                  for ui_name, tp in type_dict.items()}
        partition_to_type_name = invert_dict(type_name_to_partition)
        rows = []
        for partition in self.val_adapter.obj_storage.partitions():
            if partition in partition_to_type_name:
                type_name = partition_to_type_name[partition]
            else:
                type_name = partition
            data = {
                'type_name': type_name,
                'partition': partition,
                'size': self.val_adapter.obj_storage.get_partition_size(name=partition)
            }
            rows.append(data)
        return pd.DataFrame(rows)
        
    @transaction()
    def describe_rels(self, conn:Connection=None) -> TTuple[pd.DataFrame, pd.DataFrame]:
        sigmaps = self.op_adapter.sigmaps
        op_ui_to_name = self.op_adapter.ui_to_name
        ui_name_and_version_to_relname = {
            (ui_name, version): self._get_op_relname_from_data(name=op_ui_to_name[ui_name], version=version)
            for ui_name, version in sigmaps
        }
        relname_to_ui_name_and_version = invert_dict(ui_name_and_version_to_relname)
        op_descriptions = []
        var_descriptions = []
        for relname in self.rel_storage.tables():
            size = self.rel_storage.get_count(table=relname, conn=conn)
            if relname in relname_to_ui_name_and_version:
                ui_name, version = relname_to_ui_name_and_version[relname]
                op_data = {
                    'ui_name': ui_name,
                    'version': version,
                    'relname': relname,
                    'size': size
                }
                op_descriptions.append(op_data)
            elif relname == self.VAR_TABLE:
                var_data = {
                    'relname': relname,
                    'size': size
                }
                var_descriptions.append(var_data)
            else:
                pass
        op_df = pd.DataFrame(op_descriptions)
        op_df = op_df.sort_values(by='ui_name')
        var_df = pd.DataFrame(var_descriptions)
        return op_df, var_df

    # @transaction()
    # def read_vref_rels(self, partitions:TList[str]=None, conn:Connection=None) -> pd.DataFrame:
    #     """
    #     Read the table corresponding to vrefs in given partitions
    #     """
    #     tableobj = self.get_var_tableobj()
    #     query = tableobj.select()
    #     if partitions is not None:
    #         query = query.where(tableobj.c[self.VAR_PARTITION].in_(get_in_clause_rhs(partitions)))
    #     df = self.rel_storage.fast_select(query=query, conn=conn)
    #     return df
    
    # @transaction()
    # def get_committed_vref_locs(self, partitions:TList[str]=None, conn:Connection=None) -> TList[BaseObjLocation]:
    #     """
    #     Get the locations of vrefs in given partitions committed to relations
    #     """
    #     vref_rels = self.read_vref_rels(partitions=partitions, conn=conn)
    #     raise NotImplementedError()
