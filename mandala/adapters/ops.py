from abc import ABC, abstractmethod

from ..common_imports import *
from ..core.config import CoreConfig
from ..core.tps import Type, is_subtype
from ..core.bases import Operation, BaseSignature
from ..core.sig import BaseSigMap, relabel_sig, Signature 
from ..core.exceptions import SynchronizationError
from ..util.common_ut import invert_dict, rename_dict_keys, get_uid, all_distinct

class BaseOpAdapter(ABC):
    DEFAULT_VERSION = '0'

    @property
    @abstractmethod
    def ui_to_name(self) -> TDict[str, str]:
        """
        ui name -> internal name
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def internal_to_ui(self) -> TDict[str, str]:
        """
        internal name -> ui name
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def internal_sigmaps(self) -> TDict[TTuple[str, str], BaseSigMap]:
        """
        {(internal name, version) : sigmap}
        """
        raise NotImplementedError()

    @abstractmethod
    def get_latest_version(self, internal_name:str) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def sigmaps(self) -> TDict[TTuple[str, str], BaseSigMap]:
        """
        {(ui name, version): sigmap}
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_sigmap(self, op:Operation) -> BaseSigMap:
        raise NotImplementedError()
    
    @abstractmethod
    def has_ui_name(self, ui_name:str) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def has_op(self, ui_name:str, version:str) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def get_ui_to_internal_interface(self, op:Operation, which:str='inputs') -> TDict[str, str]:
        """
        {ui_io_name: internal_io_name} for either inputs or outputs.
        
        - which: one of ('inputs', 'outputs')
        """
        raise NotImplementedError()
    
    ############################################################################ 
    ### PURE refactoring operations
    ############################################################################ 
    @abstractmethod
    def increment_versions(self, ui_names:TList[str]) -> 'BaseOpAdapter':
        raise NotImplementedError()
    
    @abstractmethod
    def update_op(self, new_op:Operation) -> TTuple['BaseOpAdapter', 
                                                    TSet[str], 'OpUpdates']:
        """
        Purely update an existing op along one or more of each of the following:
            - new inputs,
            - extended types,
        
        Returns:
            - new op adapter object
            - new *internal* input names, if any
            - object describing updates 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def create_op(self, op:Operation, 
            caller_ui_names:TList[str]) -> TTuple['BaseOpAdapter', str, 
                                                TDict[str, TTuple[str, str]]]:
        raise NotImplementedError()
    
    # @abstractmethod
    # def drop_all_versions(self, ui_name:str):
    #     raise NotImplementedError()
    
    @abstractmethod
    def drop_version(self, ui_name:str, version:str) -> 'BaseOpAdapter':
        raise NotImplementedError()
    
    @abstractmethod
    def rename_ops(self, 
                   internal_to_new_ui_name:TDict[str, str]) -> 'BaseOpAdapter':
        raise NotImplementedError()
    
    @abstractmethod
    def rename_inputs(self, internal_name:str, version:str,
                      rename_dict:TDict[str, str]) -> 'BaseOpAdapter':
        raise NotImplementedError()
    
    @abstractmethod
    def _dump_state(self) -> TDict[str, TAny]:
        """
        Copy state to an object supporting __eq__
        """
        raise NotImplementedError()
    

class OpAdapter(BaseOpAdapter):
    
    def __init__(self):
        # ui_name -> internal name
        self._name_map:TDict[str, str] = {}
        # (ui_name, version) -> map from UI signature to internal signature
        self._sigmaps:TDict[TTuple[str, str], BaseSigMap] = {} 
    
    ############################################################################ 
    ### interface implementation
    ############################################################################ 
    @property
    def ui_to_name(self) -> TDict[str, str]:
        return self._name_map
    
    @property
    def internal_to_ui(self) -> TDict[str, str]:
        return invert_dict(self.ui_to_name)
    
    @property
    def internal_sigmaps(self) -> TDict[TTuple[str, str], BaseSigMap]:
        res = {}
        for (ui_name, version), sigmap in self._sigmaps.items():
            internal_name = self.ui_to_name[ui_name]
            res[internal_name, version] = sigmap
        return res
    
    def get_latest_version(self, internal_name:str) -> str:
        versions = [version for other_ui_name, version in self.sigmaps.keys()
                    if self.ui_to_name[other_ui_name] == internal_name]
        return max(versions, key=lambda x: int(x))
    
    @property
    def sigmaps(self) -> TDict[TTuple[str, str], BaseSigMap]:
        return self._sigmaps
    
    def get_sigmap(self, op:Operation) -> BaseSigMap:
        return self._sigmaps[op.ui_name, op.version]

    def has_ui_name(self, ui_name: str) -> bool:
        return ui_name in self.ui_to_name
    
    def has_op(self, ui_name:str, version:str) -> bool:
        return (ui_name, version) in self._sigmaps

    def sig_is_compatible(self, op: Operation, new_sig: BaseSignature,
                          base_input_map:TDict[str, str]) -> TTuple[bool, str, 'OpUpdates']:
        """
        Check if a new signature for an operation is compatible with the 
        current state of the adapter. If not, a reason why is returned. 
        
        Returns:
            - res: whether change is compatible
            - reason: if change is not compatible, an explanation as to the
              whyness
            - updates object
        
        TODO: return more complete reasons
        """
        (input_type_diff, output_type_diff, defaults_diff, deleted_internal_inputs,
         created_inputs, order_changes) = self.get_signature_diff(op=op, 
                                new_sig=new_sig, base_input_map=base_input_map)
        updates = OpUpdates(ui_name=op.ui_name, 
                            input_type_diff=input_type_diff, 
                            output_type_diff=output_type_diff, 
                            created_inputs=created_inputs)
        res, reason = None, None
        if order_changes == True:
            cur_ui_sig = self.get_sigmap(op=op).source
            res = False
            reason = (f'Changing order of possibly positional arguments is not'
            f'supported.\nNew order: {new_sig.poskw_names}\nCurrent order: '
            f'{cur_ui_sig.poskw_names}')
        if len(defaults_diff) > 0:
            res, reason = False, f'Changing defaults not supported: {defaults_diff}'
        elif len(input_type_diff) > 0:
            for input_name, (current_type, new_type) in input_type_diff.items():
                if not self.is_valid_type_update(current=current_type, new=new_type):
                    res = False
                    reason = ('Changing input types is only supported via ' 
                              f'extension: {input_name, (current_type, new_type)}')
        elif len(output_type_diff) > 0:
            for output_name, (current_type, new_type) in output_type_diff.items():
                if not self.is_valid_type_update(current=current_type, new=new_type):
                    res = False
                    reason = f'Changing output types is only supported via extension: {output_name, (current_type, new_type)}'
        elif len(deleted_internal_inputs) > 0:
            res, reason = False, f'Deleting inputs not supported: {deleted_internal_inputs}'
        elif len(created_inputs) > 0:
            if not CoreConfig.allow_new_inputs:
                res, reason = False, f'Creating inputs not supported: {created_inputs}'
        if res is None or reason is None:
            res, reason = True, ''
        return res, reason, updates
    
    def get_ui_to_internal_interface(self, op:Operation,
                                     which:str='inputs') -> TDict[str, str]:
        assert which in ('inputs', 'outputs')
        sigmap = self._sigmaps[op.ui_name, op.version]
        if which == 'inputs':
            return sigmap.fixed_inputs_map()
        elif which == 'outputs':
            return sigmap.fixed_outputs_map()
        else:
            raise NotImplementedError()
        
    def increment_versions(self, ui_names:TList[str]) -> TTuple['OpAdapter', 
                                                          TDict[str, TTuple[str, str]]]:
        """
        Return the new op adapter and the mapping from ui names to (old version,
        new version)
        """
        res_sigmaps = copy.deepcopy(self._sigmaps)
        res_name_map = copy.deepcopy(self._name_map)
        version_changes = {}
        for ui_name in ui_names:
            latest_version = self.get_latest_version(internal_name=res_name_map[ui_name])
            latest_sigmap = res_sigmaps[(ui_name, latest_version)]
            new_version = str(int(latest_version) + 1)
            res_sigmaps[(ui_name, new_version)] = copy.deepcopy(latest_sigmap)
            version_changes[ui_name] = (latest_version, new_version)
        return OpAdapter.from_data(name_map=res_name_map, sigmaps=res_sigmaps), version_changes
        
    def update_op(self, new_op:Operation) -> TTuple['OpAdapter', TSet[str], 'OpUpdates']:
        res_sigmaps = copy.deepcopy(self._sigmaps)
        res_name_map = copy.deepcopy(self._name_map)
        ui_name, version = new_op.ui_name, new_op.version
        assert (ui_name, version) in res_sigmaps
        # existing operation 
        sigmap = res_sigmaps[ui_name, version]
        if new_op.sig != sigmap.source:
            base_input_map = sigmap.fixed_inputs_map()
            compatible, reason, updates = self.sig_is_compatible(op=new_op, 
                            new_sig=new_op.sig, base_input_map=base_input_map)
            if not compatible:
                raise SynchronizationError(f'Incompatible new signature for operation {new_op}, reason:\n {reason}')
            ### get new sigmap
            if new_op.sig.has_fixed_outputs:
                base_output_map = sigmap.fixed_outputs_map()
            else:
                base_output_map = None
            new_sigmap = self.get_immutable_relabeling(sig=new_op.sig,
                                                       base_input_map=base_input_map, 
                                                       base_output_map=base_output_map,
                                                       base_target=sigmap.target)
            new_target = new_sigmap.target
            new_input_names = new_target.fixed_input_names.difference(
                              sigmap.target.fixed_input_names)
            ### update sigmap
            res_sigmaps[ui_name, version] = new_sigmap
        else:
            updates = OpUpdates(ui_name=new_op.ui_name, input_type_diff={}, 
                                output_type_diff={}, created_inputs=[])
            new_sigmap = res_sigmaps[ui_name, version] # it's unchanged
            new_input_names = set()
        updates.set_sigmap(sigmap=new_sigmap)
        return (OpAdapter.from_data(name_map=res_name_map, sigmaps=res_sigmaps),
                new_input_names,
                updates)
        
    def create_op(self, 
                  op:Operation, 
                  caller_ui_names:TList[str]) -> TTuple['OpAdapter', str, TDict[str, TTuple[str, str]]]:
        """
        Purely add a new op to this adapter. 
        
        Returns:
            - new op adapter
            - internal name
            - dictionary of {ui_name: (old_version, new_version)} for version
              changes triggered in superops using the op, if this is a version
              change. 
        """
        res_sigmaps = copy.deepcopy(self._sigmaps)
        res_name_map = copy.deepcopy(self._name_map)
        ui_name, version = op.ui_name, op.version
        assert (ui_name, version) not in res_sigmaps
        if op.is_builtin:
            internal_name = ui_name
        else:
            internal_name = self.generate_id()
        res_name_map[ui_name] = internal_name
        # generate new sigmap
        sigmap = self.get_immutable_relabeling(sig=op.sig, base_input_map={}, 
                                                base_output_map={})
        res_sigmaps[ui_name, version] = sigmap
        res = OpAdapter.from_data(name_map=res_name_map, sigmaps=res_sigmaps)
        if len(caller_ui_names) > 0:
            res, version_changes = res.increment_versions(ui_names=caller_ui_names)
        else:
            version_changes = {}
        return res, internal_name, version_changes
    
    # def drop_all_versions(self, ui_name: str):
    #     del self._name_map[ui_name]
    #     sigmap_keys_to_drop = []
    #     for other_ui_name, other_version in self.sigmaps:
    #         if other_ui_name == ui_name:
    #             sigmap_keys_to_drop.append((other_ui_name, other_version))
    #     for k in sigmap_keys_to_drop:
    #         del self._sigmaps[k]
    
    def drop_version(self, ui_name: str, version: str) -> 'OpAdapter':
        res_sigmaps = copy.deepcopy(self._sigmaps)
        res_name_map = copy.deepcopy(self._name_map)
        del res_sigmaps[ui_name, version]
        any_versions_remain = False
        for other_ui_name, _ in res_sigmaps.keys():
            if other_ui_name == ui_name:
                any_versions_remain = True
        if not any_versions_remain:
            del res_name_map[ui_name]
        return OpAdapter.from_data(name_map=res_name_map, sigmaps=res_sigmaps)
    
    def rename_ops(self, internal_to_new_ui_name:TDict[str, str]) -> 'OpAdapter':
        """
        Return a new OpAdater with the given ops renamed.
        """
        internal_to_cur_ui_name = self.internal_to_ui
        new_ui_names = [internal_to_new_ui_name.get(internal, 
                                                    internal_to_cur_ui_name[internal])
                        for internal in internal_to_cur_ui_name.keys()]
        if not all_distinct(elts=new_ui_names):
            raise SynchronizationError('Name collision in renaming')
        sigmaps = {}
        name_map = {}
        for (cur_ui_name, version), sigmap in self.sigmaps.items():
            internal_name = self.ui_to_name[cur_ui_name]
            if internal_name in internal_to_new_ui_name.keys():
                new_ui_name = internal_to_new_ui_name[internal_name]
                sigmaps[new_ui_name, version] = sigmap
            else:
                sigmaps[cur_ui_name, version] = sigmap
        for cur_ui_name, internal_name in self.ui_to_name.items():
            if internal_name in internal_to_new_ui_name.keys():
                new_ui_name = internal_to_new_ui_name[internal_name]
                name_map[new_ui_name] = internal_name
            else:
                name_map[cur_ui_name] = internal_name
        return OpAdapter.from_data(name_map=name_map, sigmaps=sigmaps)
    
    def rename_inputs(self, internal_name:str, version:str, 
                      rename_dict: TDict[str, str]) -> 'OpAdapter':
        """
        Return a new adapter with the given inputs of the given signature
        renamed.
        """
        sigmaps = {}
        name_map = self._name_map
        for (other_ui_name, other_version), sigmap in self.sigmaps.items():
            other_internal_name = self.ui_to_name[other_ui_name]
            if ((internal_name == other_internal_name) and 
                (version == other_version)):
                new_sigmap = sigmap.rename_source_inputs(rename_dict=rename_dict)
            else:
                new_sigmap = sigmap
            sigmaps[other_ui_name, other_version] = new_sigmap
        return OpAdapter.from_data(name_map=name_map, sigmaps=sigmaps)
    
    ############################################################################ 
    ### helpers
    ############################################################################ 
    def get_immutable_relabeling(self, sig:BaseSignature, 
                                 base_input_map:TDict[str, str], 
                                 base_output_map:TDict[str, str]=None,
                                 base_target:BaseSignature=None) -> BaseSigMap:
        """
        Generate an internal version of a signature together with an associated
        mapping into it. Optionally, this can be done by extending an existing
        mapping by keeping the overlap between the current signature's domain
        and the mapping's domain on inputs and/or outputs
        """
        if base_target is None:
            base_target = Signature()
        arg_map = {name: base_input_map[name] if name in base_input_map
                   else self.generate_id() for name in sig.poskw_names}
        kwarg_map = {name: base_input_map[name] if name in base_input_map
                     else self.generate_id() for name in sig.kw}
        if sig.vararg is None:
            new_vararg = None
        else:
            new_vararg = base_target.vararg[0] if base_target.vararg is not None else self.generate_id()
        if sig.varkwarg is None:
            new_varkwarg = None
        else:
            new_varkwarg = base_target.varkwarg[0] if base_target.varkwarg is not None else self.generate_id()
        if sig.has_fixed_outputs:
            assert sig.output_names is not None
            assert base_output_map is not None
            output_map = {name: base_output_map[name] if name in base_output_map
                          else self.generate_id() for name in sig.output_names}
        else:
            output_map = None
        return relabel_sig(sig=sig, arg_map=arg_map, new_vararg=new_vararg, 
                           kwarg_map=kwarg_map, new_varkwarg=new_varkwarg, 
                           output_map=output_map)

    @staticmethod
    def from_data(name_map:TDict[str, str],
                  sigmaps:TDict[TTuple[str, str], BaseSigMap]) -> 'OpAdapter':
        res = OpAdapter()
        res._sigmaps = sigmaps
        res._name_map = name_map
        return res
    
    def generate_id(self) -> str:
        return f'_{get_uid()}'
    
    def is_valid_type_update(self, current:Type, new:Type) -> bool:
        return is_subtype(s=current, t=new)
    
    def get_input_type_changes(self, new_sig:BaseSignature, 
                               cur_internal:BaseSignature,
                               shared_inputs_mapping:TDict[str, str]) -> TDict[str, TTuple[Type, Type]]:
        """
        Return any changes between input types for the inputs present in both
        the new signature and the current internal signature.
        
        Arguments:
            - new_sig: the new *external* signature
            - cur_internal: the current *internal* signature
        
        Returns:
            {internal_input_name: (current_type, new_type)}
        """
        res = {}
        for inp, internal_inp in shared_inputs_mapping.items():
            new_tp = new_sig.inputs()[inp]
            cur_tp = cur_internal.inputs()[internal_inp]
            if new_tp != cur_tp:
                res[internal_inp] = (cur_tp, new_tp)
        return res
                
    def get_output_type_changes(self, new_sig:BaseSignature, 
                               cur_internal:BaseSignature) -> TDict[str, TTuple[Type, Type]]:
        """
        Return any changes between output types between the new signature and
        the current internal signature.
        
        Arguments:
            - new_sig: the new *external* signature
            - cur_internal: the current *internal* signature
        
        Returns:
            {internal_input_name: (current_type, new_type)}
        """
        res = {}
        if (not new_sig.has_fixed_outputs) or (not cur_internal.has_fixed_outputs):
            raise ValueError()
        assert new_sig.output_names is not None
        assert cur_internal.output_names is not None
        if len(new_sig.output_names) != len(cur_internal.output_names):
            raise SynchronizationError('Changing number of outputs not supported')
        for ext_output_name, int_output_name in zip(new_sig.output_names, 
                                                    cur_internal.output_names):
            new_tp = new_sig.outputs[ext_output_name]
            cur_tp = cur_internal.outputs[int_output_name]
            if new_tp != cur_tp:
                res[int_output_name] = (cur_tp, new_tp)
        return res
                
    def get_defaults_changes(self, new_sig:BaseSignature, 
                             cur_internal:BaseSignature,
                             shared_inputs_mapping:TDict[str, str]) -> TDict[str, TTuple[TAny, TAny]]:
        """
        Return any changes between input defaults for the inputs present in both
        the new signature and the current internal signature.
        
        Returns:
            {internal_input_name: (current_default, new_default)}
        """
        res = {}
        for inp, internal_inp in shared_inputs_mapping.items():
            new_default = new_sig.defaults.get(inp, inspect._empty)
            cur_default = cur_internal.defaults.get(internal_inp, inspect._empty)
            if new_default != cur_default:
                res[internal_inp] = (cur_default, new_default)
        return res

    def get_signature_diff(self, op:Operation, new_sig:BaseSignature, 
                           base_input_map:TDict[str, str]) -> TTuple[TDict[str, TTuple[Type, Type]], 
                                                                     TDict[str, TTuple[Type, Type]], 
                                                                     TDict[str, TTuple[TAny, TAny]],
                                                                     TSet[str],
                                                                     TList[TTuple[str, Type, TAny]],
                                                                     TUnion[str, bool]]:
        """
        When an op was refactored within the same version, this function returns
        the changes which are then passed to functions to actually do the
        updates. 
        
        The mental model of this:
            - the new signature is S_new
            - the current signature map has source S_cur and target T_cur
            - `base_input_map` is an old mapping from UI input names to internal
            input names. Its overlap with the current inputs is used to keep the
            signature update consistent with what is already stored.
            
        Returns:
            - type_diff: {internal_name: (cur_type, new_type)}
            - defaults_diff: {internal name: (cur default, new default)}
            - deleted inputs: {internal names of deleted inputs}
            - created_inputs: [(UI name, type, default) list]
            - order_changes: True/False, or 'undefined' for ambiguous cases
        
        """
        if base_input_map is None:
            base_input_map = {}
        cur_sigmap = self.get_sigmap(op=op)
        cur_internal = cur_sigmap.target
        shared_input_names = set(base_input_map.keys()).intersection(new_sig.fixed_input_names)
        shared_input_mapping = {inp: base_input_map[inp] for inp in shared_input_names}
        input_type_diff = self.get_input_type_changes(new_sig=new_sig, 
                                                cur_internal=cur_internal,
                                                shared_inputs_mapping=shared_input_mapping)
        output_type_diff = self.get_output_type_changes(new_sig=new_sig,
                                                        cur_internal=cur_internal)
        defaults_diff = self.get_defaults_changes(new_sig=new_sig, 
                                                cur_internal=cur_internal,
                                                shared_inputs_mapping=shared_input_mapping)
        ### figure out creation/deletion of inputs
        deleted_input_names = set(base_input_map.keys()).difference(new_sig.fixed_input_names)
        deleted_internal_inputs = {base_input_map[x] for x in deleted_input_names}
        created_input_names = new_sig.fixed_input_names.difference(set(base_input_map.keys()))
        created_inputs = [(inp, new_sig.inputs()[inp], new_sig.defaults.get(inp,
                                                        inspect._empty))
                          for inp in created_input_names]
        ### figure out if there is any reordering of input names for potentially positional arguments
        new_poskw_order = [elt[0] for elt in new_sig.ord_poskw]
        current_poskw_order = [elt[0] for elt in cur_sigmap.source.ord_poskw]
        if len(current_poskw_order) > len(new_poskw_order):
            order_changes = 'undefined' # this case does not lead to op update anyway
        elif len(current_poskw_order) == len(new_poskw_order):
            if not set(new_poskw_order) == set(current_poskw_order):
                order_changes = 'undefined'
            else:
                order_changes = new_poskw_order != current_poskw_order
        else:
            order_changes = current_poskw_order != new_poskw_order[:len(current_poskw_order)]
        return (input_type_diff, output_type_diff, defaults_diff, 
                deleted_internal_inputs, created_inputs, order_changes)

    def _dump_state(self) -> TDict[str, TAny]:
        return copy.deepcopy(self.__dict__)


class OpUpdates(object):
    """
    Represent (non-renaming) updates to an existing operation
    """
    def __init__(self, ui_name:str, 
                 input_type_diff:TDict[str, TTuple[Type, Type]], 
                 output_type_diff:TDict[str, TTuple[Type, Type]],
                 created_inputs:TList[TTuple[str, Type, TAny]], 
                 sigmap:BaseSigMap=None):
        """
        Args:
            ui_name (str): UI name of the operation being updated
            input_type_diff: {internal name: (old_type, new_type)}
            output_type_diff: {internal name: (old_type, new_type)}
            created_inputs: [(ui name, type, default)]
        """
        self.ui_name = ui_name
        self.input_type_changes = input_type_diff
        self.output_type_changes = output_type_diff
        self.created_inputs = created_inputs
        self._sigmap = sigmap
    
    @property
    def sigmap(self) -> BaseSigMap:
        return self._sigmap
    
    def set_sigmap(self, sigmap:BaseSigMap):
        self._sigmap = sigmap
    
    def show(self):
        if not self.empty:
            # rename data to UI 
            assert self.sigmap is not None
            internal_to_ui = invert_dict(self.sigmap.fixed_io_map())
            input_type_changes = rename_dict_keys(dct=self.input_type_changes,
                                                  mapping=internal_to_ui)
            output_type_changes = rename_dict_keys(dct=self.output_type_changes,
                                                   mapping=internal_to_ui)
            created_inputs = self.created_inputs # it's renamed already
            # print data
            logging.info(f'UPDATE OPERATION "{self.ui_name}":')
            for input_name, (old_type, new_type) in input_type_changes.items():
                logging.info(f'Extend type of argument "{input_name}": {old_type} -> {new_type}')
            for output_name, (old_type, new_type) in output_type_changes.items():
                logging.info(f'Extend type of output "{output_name}": {old_type} -> {new_type}')
            for new_arg_name, new_arg_type, new_arg_default in created_inputs:
                logging.info(f'Add argument "{new_arg_name}", type={new_arg_type}, default={new_arg_default}')
    
    @property
    def empty(self) -> bool:
        return ((not self.input_type_changes) and
                (not self.output_type_changes) and
                (not self.created_inputs))