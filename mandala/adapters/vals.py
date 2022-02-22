from abc import ABC, abstractmethod

from ..common_imports import *
from ..core.tps import Type, AtomType, ListType, UnionType, AnyType, DictType
from ..core.idx import ImplIndex, ValueIndex
from ..core.bases import ValueRef, VRefCollecUtils, TVRefCollection, unwrap
from ..storages.objects import BaseObjStorage, BaseObjLocation, Forgotten
from ..storages.objects import PartitionedObjLocation
from ..storages.kv import KVStore
from ..util.common_ut import get_collection_atoms, apply_to_collection, concat_lists
from ..util.common_ut import (
    flatten_collection_atoms, unflatten_atoms_like, group_like, ungroup_like
                     )
from .tps import BaseTypeAdapter

################################################################################
### interfaces for combining storage and value primitives in higher-level tools
################################################################################
ObjLocCollection = TUnion[
    TList['ObjLocCollection'],
    TDict[str, 'ObjLocCollection'],
    BaseObjLocation
    ]

class BaseValAdapter(ABC):
    """
    Glue between vrefs and object storage
    """
    @property
    @abstractmethod
    def obj_storage(self) -> BaseObjStorage:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def value_impl_index(self) -> ImplIndex:
        raise NotImplementedError()
    
    @abstractmethod
    def get_vref_location(self, vref:ValueRef) -> BaseObjLocation:
        raise NotImplementedError()
    
    @abstractmethod
    def dump_obj_loc_collection(self, collection:ObjLocCollection) -> TAny:
        raise NotImplementedError()

    @abstractmethod
    def load_obj_loc_collection(self, data:TAny) -> ObjLocCollection:
        raise NotImplementedError()

    @abstractmethod
    def get(self, loc:BaseObjLocation, lazy:bool=False) -> ValueRef:
        """
        Load a value reference from object storage. 

        - lazy (bool): When lazy is set to True, the resulting object is a
          detached value reference (in particular, in_memory is set to False).
          Otherwise, the returned vref is marked as in memory.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def set_shallow(self, vref:ValueRef) -> BaseObjLocation:
        """
        Save only this vref's own data (without constituents)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def set(self, vref:ValueRef, shallow:bool=False) -> BaseObjLocation:
        """
        Save vref to storage
        """
        raise NotImplementedError()
    
    ### mmethods
    @abstractmethod
    def mget(self, locs:TList[BaseObjLocation], lazy:bool=False) -> TList[ValueRef]:
        raise NotImplementedError()
    
    @abstractmethod
    def deeply_persisted(self, loc:BaseObjLocation) -> bool:
        """
        Whether all constituents of this location are present in storage
        """
        raise NotImplementedError()
    
    @abstractmethod
    def mdeeply_persisted(self, locs:TList[BaseObjLocation]) -> TList[bool]:
        raise NotImplementedError()

    @abstractmethod
    def mset(self, vrefs:TList[ValueRef], shallow:bool=False, 
             skip_delayed:bool=False) -> TList[BaseObjLocation]:
        raise NotImplementedError()
    
    @abstractmethod
    def attach(self, vref:ValueRef, shallow:bool=False):
        """
        Load in-place
        """
        raise NotImplementedError()
    
    @abstractmethod
    def mattach(self, vrefs:TList[ValueRef], shallow:bool=False):
        raise NotImplementedError()
    
    @abstractmethod
    def mget_collection(self, collection:ObjLocCollection, 
                        lazy:bool=False, deeplazy:bool=False) -> TVRefCollection:
        raise NotImplementedError()

    ############################################################################ 
    ### partition interface
    ############################################################################ 
    @property
    @abstractmethod
    def type_adapter(self) -> BaseTypeAdapter:
        raise NotImplementedError()
    
    @abstractmethod
    def set_type_adapter(self, value:BaseTypeAdapter):
        raise NotImplementedError()
    
    @abstractmethod
    def synchronize_type(self, ui_name:str, tp:Type, kv:KVStore=None):
        """
        Given a type an its UI-facing name, assign it a new unique immutable
        internal name, or link it to its name if it already exists. 
        
        NOTE: this should set the internal name of the type object.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_vref_dest_partition(self, vref:ValueRef) -> str:
        """
        Return the unique partition where this vref should be stored
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_concrete_tp_partition(self, tp:Type) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_partitions_matching_tp(self, tp:Type) -> TList[str]:
        """
        Return the partitions where values belonging to this type can be found
        """
        raise NotImplementedError()

    ############################################################################ 
    ### query interface
    ############################################################################ 
    @abstractmethod
    def evaluate(self, df:pd.DataFrame,
                 unwrap_vrefs:bool=True, lazy:bool=False) -> pd.DataFrame:
        """
        Evaluate a table of *object locations*
        """
        raise NotImplementedError()
    
    @abstractmethod
    def isin(self, rng:TList[TAny], tp:Type,
             locs:TList[BaseObjLocation]=None) -> TList[BaseObjLocation]:
        """
        Return list of locations matching the given type and whose values belong
        to the given range.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def where(self, pred:TCallable, tp:Type,
              locs:TList[BaseObjLocation]=None) -> TList[BaseObjLocation]:
        """
        Return list of locations matching the given type and whose values 
        satisfy the given predicate
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_tp_locs(self, tp:Type) -> TList[BaseObjLocation]:
        """
        Get locations matching this type
        """
        raise NotImplementedError()
    
    def load_tp_dict(self, tp:Type) -> TDict[BaseObjLocation, ValueRef]:
        """
        Get a dictionary of {location: vref} pairs matching a given type
        """
        locs = self.get_tp_locs(tp=tp)
        vals = self.mget(locs=locs)
        return {loc: val for loc, val in zip(locs, vals)}
    
    def eval_tp(self, tp:Type) -> TDict[BaseObjLocation, TAny]:
        return {k: unwrap(v) for k, v in self.load_tp_dict(tp=tp).items()}

    ############################################################################ 
    ### invariants and correctness
    ############################################################################ 
    def _verify_get(self):
        """
        Check that all objects are loadable
        """
        locs = self.obj_storage.locs()
        for loc in locs:
            value = self.get(loc=loc)

################################################################################
### implementation
################################################################################
class ValAdapter(BaseValAdapter):

    def __init__(self, obj_storage:BaseObjStorage, value_impl_index:ImplIndex=None):
        self._obj_storage = obj_storage
        if value_impl_index is None:
            value_impl_index = ValueIndex
        self._value_impl_index = value_impl_index
        self._type_adapter = None

    @property
    def obj_storage(self) -> BaseObjStorage:
        return self._obj_storage
    
    @property
    def value_impl_index(self) -> ImplIndex:
        return self._value_impl_index
    
    ############################################################################ 
    ### helpers
    ############################################################################ 
    def get_vref_location(self, vref:ValueRef) -> BaseObjLocation:
        partition = self.get_vref_dest_partition(vref=vref)
        if not self.obj_storage.has_partition(name=partition):
            self.obj_storage.create_partition(name=partition)
        return PartitionedObjLocation(uid=vref.uid, partition=partition)
    
    def _constituents_to_locs(self, constituents:TVRefCollection) -> ObjLocCollection:
        loc_getter = lambda vref: self.get_vref_location(vref)
        obj_loc_collection = apply_to_collection(collection=constituents, func=loc_getter)
        obj_loc_data = self.dump_obj_loc_collection(collection=obj_loc_collection)
        return obj_loc_data
    
    def _to_main_and_meta(self, vref:ValueRef, skip_delayed:bool=False) -> TTuple[TAny, ValueRef]:
        """
        Return main record and metadata record to be stored for this vref.
        
        NOTE: this takes care of the `in_memory`, `is_persistable`, and
        `delayed_storage` attributes.
        """
        if not vref.in_memory:
            raise ValueError()
        meta = vref.detached()
        if (not vref.is_persistable) or (skip_delayed and vref.delayed_storage):
            return (Forgotten(), meta)
        primary, constituents = vref.get_residue()
        if vref.is_compound():
            return (self._constituents_to_locs(constituents=constituents), meta)
        else:
            return (primary, meta)

    def dump_obj_loc_collection(self, collection:ObjLocCollection) -> TAny:
        return collection

    def load_obj_loc_collection(self, data:TAny) -> ObjLocCollection:
        return data

    def _process_lazy(self, obj_meta:ValueRef, exists_main:bool) -> ValueRef:
        """
        Set attributes of a lazy result
        """
        obj_meta.set_in_memory(value=False)
        obj_meta.set_is_persistable(value=exists_main)
        return obj_meta
    
    def _process_no_main(self, obj_meta:ValueRef) -> ValueRef:
        """
        Set attributes of a `get()` result without a main record
        """
        obj_meta.set_in_memory(value=False)
        if obj_meta.is_persistable and (not obj_meta.delayed_storage):
            # there must be SOME reason why there's no main
            raise ValueError('Something is very wrong')
        return obj_meta
    
    def get(self, loc:BaseObjLocation, lazy:bool=False) -> ValueRef:
        """
        Satisfies the following invariants:
            - any result satisfies
                `is_persistable = False => in_memory = False` 
            - when a location has/doesn't have a main record, the returned vref
            satisfies 
                `is_persistable = True/False`
            - when lazy=True, the result satisfies
                `in_memory = False`, 
            - when lazy=False, all (recursive) constituents of the result that 
            have main records satisfy
                `in_memory = True`
        """
        if not self.obj_storage.exists_main(loc=loc):
            _, obj_meta = self.obj_storage.get(loc=loc)
            return self._process_no_main(obj_meta=obj_meta)
        else:
            data = self.obj_storage.get(loc=loc)
            obj_data = data[0]
            obj_meta:ValueRef = data[1]
            if lazy:
                return self._process_lazy(obj_meta=obj_meta, exists_main=True)
            vref_impl_id = ValueIndex.get_impl_id(cls=type(obj_meta))
            ImplClass:TType[ValueRef] = self.value_impl_index.get(impl_id=vref_impl_id)
            if not ImplClass.is_compound():
                # bottom of recursion
                obj = ImplClass.obj_from_residue(obj=obj_data, constituents=VRefCollecUtils.get_empty())
            else:
                obj_loc_collection = self.load_obj_loc_collection(data=obj_data)
                loader = lambda obj_loc: self.get(obj_loc)
                constituents = apply_to_collection(collection=obj_loc_collection, func=loader)
                obj = ImplClass.obj_from_residue(obj=None, constituents=constituents)
            res = obj_meta.attached(obj=obj)
            res.set_in_memory(value=True)
            res.set_is_persistable(value=True)
            return res
    
    def set_shallow(self, vref:ValueRef) -> BaseObjLocation:
        return self.set(vref=vref, shallow=True)
    
    def set(self, vref:ValueRef, shallow:bool=False,
            require_in_memory:bool=False) -> BaseObjLocation:
        """
        Satisfies the following invariants:
            - calling `set()` on a vref where `in_memory = False` is forbidden
            (when require_in_memory=True), or otherwise has no effect.  
            - calling `set()` on a vref which satisfies `is_persistable = False` 
            either does not interact at all with the main storage, or otherwise
            deletes the main record for this location. 
            - when `shallow=True`, only the principal records are saved
            - when `shallow=False`, all recursive constituents are saved as if by
            applying set(shallow=True) to each.
        """
        # it is just easier to invoke mset here to handle recursion
        result = self.mset(vrefs=[vref], shallow=shallow, require_in_memory=require_in_memory)
        return result[0]
    
    ############################################################################ 
    ### mmethods
    ############################################################################ 
    ################
    ### mset-related
    ################
    def mdecompose(self, vrefs:TList[ValueRef], shallow:bool=False,
                   result:TDict[BaseObjLocation, ValueRef]=None) -> TDict[BaseObjLocation, ValueRef]:
        """
        Decompose a list of vrefs into {location: vref} pairs, optionally
        traversing any constituents in memory. 
        
        If shallow, does not look inside constituents, otherwise also includes
        constituents of compound vrefs that are in memory. 
        
        This function is very useful throughout this class.
        """
        if result is None:
            result = {}
        for vref in vrefs:
            loc = self.get_vref_location(vref=vref)
            result[loc] = vref
            if (not shallow) and vref.in_memory and vref.is_compound():
                _, constituents = vref.get_residue()
                flat_constituents = get_collection_atoms(collection=constituents)
                self.mdecompose(vrefs=flat_constituents, result=result)
        return result
        
    def mset(self, vrefs:TList[ValueRef], shallow:bool=False, 
             require_in_memory:bool=False, skip_delayed:bool=False) -> TList[BaseObjLocation]:
        all_vrefs = self.mdecompose(vrefs=vrefs, shallow=shallow)
        if require_in_memory and any(not vref.in_memory for vref in all_vrefs.values()):
            raise ValueError()
        mains_and_metas = {loc: self._to_main_and_meta(vref=vref, skip_delayed=skip_delayed)
                           for loc, vref in all_vrefs.items() 
                           if vref.in_memory}
        self.obj_storage.mset(mapping={loc: elt[0] for loc, elt in mains_and_metas.items()},
                              meta_mapping={loc: elt[1] for loc, elt in mains_and_metas.items()})
        return [self.get_vref_location(vref=vref) for vref in vrefs]
    
    ################
    ### mget-related
    ################
    def _mprocess_no_main(self, obj_metas:TList[ValueRef]) -> TList[ValueRef]:
        return [self._process_no_main(obj_meta=obj_meta) for obj_meta in obj_metas]
    
    def _mprocess_lazy(self, obj_metas:TList[ValueRef], mexists_main:TList[bool]) -> TList[ValueRef]:
        for obj_meta, exists_main in zip(obj_metas, mexists_main):
            obj_meta.set_in_memory(value=False)
            obj_meta.set_is_persistable(value=exists_main)
        return obj_metas
    
    def _mget_atoms(self, locs:TList[BaseObjLocation], obj_metas:TList[ValueRef], 
                    impl_classes:TList[TType[ValueRef]], lazy:bool=False) -> TList[ValueRef]:
        """
        mget *persisted, non-lazy* atoms
        """
        if lazy:
            return obj_metas
        obj_datas, _ = self.obj_storage.mget(locs=locs) #! todo optimize 
        objs = [impl_class.obj_from_residue(obj=obj_data, constituents=VRefCollecUtils.get_empty())
                for impl_class, obj_data in zip(impl_classes, obj_datas)]
        res = [obj_meta.attached(obj=obj) for obj_meta, obj in zip(obj_metas, objs)]
        for vref in res:
            vref.set_in_memory(value=True)
            vref.set_is_persistable(value=True)
        return res
    
    def _mget_compound(self, locs:TList[BaseObjLocation], obj_metas:TList[ValueRef], 
                       impl_classes:TList[TType[ValueRef]], depth:int=None,
                       optimized:bool=True) -> TList[ValueRef]:
        """
        mget *persisted, non-lazy* compound values
        """
        obj_datas, _ = self.obj_storage.mget(locs=locs) #! optimzie
        compound_loc_collecs = [self.load_obj_loc_collection(data=obj_data) for obj_data in obj_datas]
        compound_locs_flat, compound_locs_index = flatten_collection_atoms(collection=compound_loc_collecs, start_idx=0)
        if not compound_locs_flat:
            compound_constituents_flat = []
        else:
            ### recursive step
            recursive_depth = None if depth is None else (depth - 1)
            compound_constituents_flat = self.mget(compound_locs_flat, optimized=optimized, 
                                                   depth=recursive_depth)
        # unflatten back to same shape
        compound_constituents = unflatten_atoms_like(atoms=compound_constituents_flat, index_reference=compound_locs_index)
        compound_objs = [impl_class.obj_from_residue(obj=None, constituents=constituents)
                            for impl_class, constituents in zip(impl_classes, compound_constituents)]
        res = [obj_meta.attached(obj=compound_obj) for obj_meta, compound_obj in zip(obj_metas, compound_objs)]
        for vref in res:
            vref.set_in_memory(value=True)
            vref.set_is_persistable(value=True)
        return res

    def mget(self, locs: TList[BaseObjLocation], lazy:bool=False, depth:int=None,
             optimized:bool=True, deeplazy:bool=False) -> TList[ValueRef]:
        """
        "Fast" way to retreive many objects at once.
        """
        if deeplazy:
            if lazy:
                logging.debug('mget() with deeplazy=True and lazy=True: deeplazy overrides')
                lazy = False
        if len(locs) > 1000:
            logging.info(f'running mget on {len(locs)} locations...')
        if lazy:
            depth = 0
        if not optimized:
            lazy = (depth == 0)
            return [self.get(loc, lazy=lazy) for loc in locs]
        else:
            obj_metas:TList[ValueRef] = self.obj_storage.mget_meta(locs=locs)
            mexists_main = self.obj_storage.mexists_main(locs=locs)
            if depth == 0:
                return self._mprocess_lazy(obj_metas=obj_metas, mexists_main=mexists_main)
            else:
                vref_impl_ids = [ValueIndex.get_impl_id(cls=type(obj_meta)) for obj_meta in obj_metas]
                vref_impl_classes:TList[TType[ValueRef]] = [
                    self.value_impl_index.get(impl_id=vref_impl_id) 
                    for vref_impl_id in vref_impl_ids]
                ### figure out cases
                compound_mask = [obj_meta.is_compound() for obj_meta in obj_metas]
                ### split into groups by case 

                def labeler(persisted, compound) -> str:
                    if not persisted:
                        return 'no_main'
                    elif persisted and not compound:
                        return 'atom'
                    else:
                        return 'compound'

                labels = [labeler(prs, cmp) for prs, cmp in zip(mexists_main, compound_mask)]
                loc_groups = group_like(objs=locs, labels=labels)
                meta_groups = group_like(objs=obj_metas, labels=labels)
                impl_class_groups = group_like(objs=vref_impl_classes, labels=labels)
                vref_groups = {
                    'no_main': self._mprocess_no_main(obj_metas=meta_groups['no_main']),
                    'atom': self._mget_atoms(locs=loc_groups['atom'], 
                                             obj_metas=meta_groups['atom'],
                                             impl_classes=impl_class_groups['atom'], lazy=deeplazy),
                    'compound': self._mget_compound(locs=loc_groups['compound'], 
                                                    depth=depth,
                                                    obj_metas=meta_groups['compound'],
                                                    impl_classes=impl_class_groups['compound'],
                                                    optimized=optimized)
                }
                return ungroup_like(groups=vref_groups, labels=labels)
    
    ### 
    def deeply_persisted(self, loc:BaseObjLocation) -> bool:
        main_exists = self.obj_storage.exists_main(loc=loc)
        if not main_exists:
            return False
        meta:ValueRef = self.obj_storage.get_meta(loc=loc)
        if not meta.is_compound():
            return True
        else:
            obj_data, _ = self.obj_storage.get(loc=loc)
            obj_loc_collection = self.load_obj_loc_collection(obj_data)
            return all(self.deeply_persisted(loc=x) for x in get_collection_atoms(obj_loc_collection))
    
    def mdeeply_persisted(self, locs:TList[BaseObjLocation]) -> TList[bool]:
        raise NotImplementedError()

    ############################################################################ 
    ### attaching
    ############################################################################ 
    def attach(self, vref:ValueRef, shallow:bool=False):
        all_vrefs = self.mdecompose(vrefs=[vref], shallow=shallow)
        not_in_mem_vrefs = [vref for vref in all_vrefs.values() if not vref.in_memory]
        self.mattach(vrefs=not_in_mem_vrefs, shallow=shallow)

    def mattach(self, vrefs:TList[ValueRef], shallow:bool=False):
        ### avoid repeat work
        labels = [vref.uid for vref in vrefs]
        vref_groups = group_like(objs=vrefs, labels=labels)
        representatives = [vref_groups[k][0] for k in vref_groups.keys()]
        all_vrefs = self.mdecompose(vrefs=representatives, shallow=shallow)
        not_in_mem_lst = [(loc, vref) for loc, vref in all_vrefs.items() if not vref.in_memory]
        not_in_mem_locs = [elt[0] for elt in not_in_mem_lst]
        not_in_mem_vrefs = [elt[1] for elt in not_in_mem_lst]
        depth = 1 if shallow else None
        loaded_vrefs = self.mget(locs=not_in_mem_locs, lazy=False, depth=depth)
        for vref, loaded_vref in zip(not_in_mem_vrefs, loaded_vrefs):
            if loaded_vref.in_memory:
                for elt in vref_groups[vref.uid]:
                    elt.attach(obj=loaded_vref.obj())

    ############################################################################ 
    ### 
    ############################################################################ 
    @property
    def type_adapter(self) -> BaseTypeAdapter:
        if self._type_adapter is None:
            raise ValueError()
        return self._type_adapter
    
    def set_type_adapter(self, value:BaseTypeAdapter):
        self._type_adapter = value

    def synchronize_type(self, ui_name:str, tp:Type, kv:KVStore=None):
        type_id = self.type_adapter.synchronize(ui_name=ui_name, tp=tp)
        if kv is not None:
            self.obj_storage.create_partition(name=type_id, main=kv)
        tp.set_name(name=type_id)
    
    def get_vref_dest_partition(self, vref:ValueRef) -> str:
        """
        Return the unique partition where this vref should be stored
        """
        tp = vref.get_type()
        return self.get_concrete_tp_partition(tp=tp)
    
    def get_concrete_tp_partition(self, tp:Type) -> str:
        if tp.is_named:
            assert tp.name is not None
            partition = tp.name
        elif isinstance(tp, AnyType):
            partition = '__any__'
        elif isinstance(tp, AtomType):
            if isinstance(tp.annotation, type):
                partition = tp.annotation.__name__
            else:
                partition = '__any__'
        elif isinstance(tp, ListType):
            partition = '__list__'
        elif isinstance(tp, DictType):
            partition = '__dict__'
        else:
            raise NotImplementedError()
        return partition
    
    def get_partitions_matching_tp(self, tp:Type) -> TList[str]:
        """
        Return the partitions where values belonging to this type can be found
        """
        if isinstance(tp, ListType):
            partitions = [tp.name if tp.is_named else '__list__']
        elif isinstance(tp, DictType):
            partitions = [tp.name if tp.is_named else '__dict__']
        elif isinstance(tp, AnyType):
            partitions = self.obj_storage.partitions()
        elif isinstance(tp, AtomType):
            partitions = [self.get_concrete_tp_partition(tp=tp)]
            # partitions = [tp.name if tp.is_named else '__any__']
        elif isinstance(tp, UnionType):
            partitions = concat_lists([self.get_partitions_matching_tp(tp=op_tp) for op_tp in tp.operands])
        else:
            raise NotImplementedError()
        return partitions
        
    def get_tp_locs(self, tp:Type) -> TList[BaseObjLocation]:
        partitions = self.get_partitions_matching_tp(tp=tp)
        locs = self.obj_storage.locs(partitions=partitions)
        return locs

    ############################################################################ 
    ### 
    ############################################################################ 
    def isin(self, rng:TList[TAny], tp:Type, 
             locs:TList[PartitionedObjLocation]=None) -> TList[PartitionedObjLocation]:
        matching_partitions = self.get_partitions_matching_tp(tp=tp)
        if locs is not None:
            uids_by_partition = group_like(objs=[loc.uid for loc in locs], labels=[loc.partition for loc in locs])
        else:
            uids_by_partition = {partition: None for partition in matching_partitions}
        res = []
        for partition in matching_partitions:
            uids = self.obj_storage._main_kvs[partition].isin(rng=rng, keys=uids_by_partition[partition])
            res += [PartitionedObjLocation(uid=uid, partition=partition) for uid in uids]
        return res
    
    def where(self, pred:TCallable, tp:Type, 
              locs:TList[PartitionedObjLocation]=None) -> TList[PartitionedObjLocation]:
        matching_partitions = self.get_partitions_matching_tp(tp=tp)
        if locs is not None:
            uids_by_partition = group_like(objs=[loc.uid for loc in locs], labels=[loc.partition for loc in locs])
        else:
            uids_by_partition = {partition: None for partition in matching_partitions}
        res = []
        for partition in matching_partitions:
            uids = self.obj_storage._main_kvs[partition].where(pred=pred, keys=uids_by_partition[partition])
            res += [PartitionedObjLocation(uid=uid, partition=partition) for uid in uids]
        return res

    def evaluate(self, df:pd.DataFrame, unwrap_vrefs:bool=True, 
                 lazy:bool=False) -> pd.DataFrame:
        cols = df.columns
        assert len(cols) == len(set(cols))
        result_cols:TList[TTuple[str, TList[ValueRef]]] = []
        for col in cols:
            col_vrefs = self.mget(locs=df[col].values.tolist(), lazy=lazy)
            if unwrap_vrefs:
                col_data = [elt.unwrap() for elt in col_vrefs]
            else:
                col_data = col_vrefs
            result_cols.append((col, col_data))
        return pd.DataFrame({col: col_data for col, col_data in result_cols}, 
                            columns=[col for col, _ in result_cols])
    
    def mget_collection(self, collection:ObjLocCollection, lazy:bool=False,
                        deeplazy:bool=False) -> TVRefCollection:
        flat, index_reference = flatten_collection_atoms(collection=collection,
                                                         start_idx=0)
        flat_results = self.mget(locs=flat, lazy=lazy, deeplazy=deeplazy)
        return unflatten_atoms_like(atoms=flat_results, 
                                    index_reference=index_reference)