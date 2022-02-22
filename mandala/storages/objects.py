from abc import ABC, abstractmethod

from .kv import KVStore, KVGroup
from .kv_impl.joblib_impl import JoblibStorage
from .kv_impl.dict_impl import DictStorage

from ..common_imports import *
from ..util.shell_ut import ask
from ..util.common_ut import group_like, ungroup_like


class BaseObjLocation(ABC):

    @property
    @abstractmethod
    def uid(self) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def partition(self) -> str:
        raise NotImplementedError()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseObjLocation):
            return False
        return self.uid == other.uid and self.partition == other.partition
    
    def __hash__(self) -> int:
        return hash((self.uid, self.partition))
    
    def __repr__(self) -> str:
        return f'ObjLocation(uid={self.uid}, partition={self.partition})'


class BaseObjStorage(ABC):
    """
    The addition of this interface compared to the lower-level interfaces:
        - the introduction of metadata that is stored and loaded alongside objects.
        - a uniform interface to (possibly) different KVStore implementations
        underneath, dispatched by location.
        - enforces invariants, such as never overwriting a location
    """
    @abstractmethod
    def drop(self, must_exist:bool=True, answer:bool=None):
        raise NotImplementedError()

    @abstractmethod
    def get_main_kv(self, partition:str) -> KVStore:
        raise NotImplementedError()
    
    @abstractmethod
    def get_meta_kv(self, partition:str) -> KVStore:
        raise NotImplementedError()

    @abstractmethod
    def partitions(self) -> TList[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def has_partition(self, name:str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def create_partition(self, name:str, main:KVStore=None):
        raise NotImplementedError()
    
    @abstractmethod
    def get_partition_size(self, name:str) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def exists_main(self, loc:BaseObjLocation) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def exists_meta(self, loc:BaseObjLocation) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get(self, loc:BaseObjLocation) -> TTuple[TAny, TAny]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_meta(self, loc:BaseObjLocation) -> TAny:
        raise NotImplementedError()
    
    @abstractmethod
    def set(self, loc:BaseObjLocation, obj:TAny, meta:TAny, 
            skip_obj:bool=False):
        raise NotImplementedError()
    
    @abstractmethod
    def delete(self, loc:BaseObjLocation):
        raise NotImplementedError()

    @abstractmethod
    def delete_all(self, answer:bool=None):
        """
        Delete all locations in this storage
        """
        raise NotImplementedError()
    
    @abstractmethod
    def locs(self, partitions:TList[str]=None) -> TList[BaseObjLocation]:
        raise NotImplementedError()

    ############################################################################ 
    ### mmethods
    ############################################################################ 
    @abstractmethod
    def mexists_main(self, locs:TList[BaseObjLocation]) -> TList[bool]:
        raise NotImplementedError()

    @abstractmethod
    def mget(self, 
             locs:TList[BaseObjLocation]) -> TTuple[TList[TAny], TList[TAny]]:
        raise NotImplementedError()
    
    @abstractmethod
    def mset(self, mapping:TDict[BaseObjLocation, TAny], 
             meta_mapping:TDict[BaseObjLocation, TAny], 
             skip_obj_mapping:TDict[BaseObjLocation, bool]=None):
        raise NotImplementedError()
    
    @abstractmethod
    def mdelete(self, locs:TList[BaseObjLocation]):
        raise NotImplementedError()
    
    @abstractmethod
    def mget_meta(self, locs:TList[BaseObjLocation]) -> TList[TAny]:
        """
        When you don't want to load object data
        """
        raise NotImplementedError()
    

################################################################################
### default implementation
################################################################################
class PartitionedObjLocation(BaseObjLocation):
    
    def __init__(self, uid:str, partition:str):
        self._uid = uid
        self._partition = partition
        
    @property
    def uid(self) -> str:
        return self._uid
    
    @property
    def partition(self) -> str:
        return self._partition


class Forgotten(object):
    pass


class ObjStorage(BaseObjStorage):
    
    def __init__(self, root:Path, transient:bool=False, 
                 DefaultKVClass:TType[KVStore]=None):
        """
        NOTE: 
            in transient mode, we make the metadata storage copy its contents; 
            this prevents vrefs from being identical objects in situations when
            this would be confusing and counter to the metaphor, e.g. saying
                ```
                x = inc(23)
                y = inc(23)
                ```
            should not lead to x, y being identical objects.
        """
        self._root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._transient = transient
        if self._transient:
            self._DefaultKVClass = DictStorage 
            self._main_kvs = KVGroup(root=self.root / 'main', default_kv=DictStorage())
            self._meta_kvs = KVGroup(root=self.root / 'meta', default_kv=DictStorage(copy_outputs=True))
        else:    
            if DefaultKVClass is None:
                DefaultKVClass = JoblibStorage
            self._DefaultKVClass = DefaultKVClass 
            self._main_kvs = KVGroup(root=self.root / 'main', default_kv=self._DefaultKVClass())
            self._meta_kvs = KVGroup(root=self.root / 'meta', default_kv=self._DefaultKVClass())
    
    def get_new_meta_kv(self) -> KVStore:
        if self._transient:
            return DictStorage(copy_outputs=True)
        else:
            return self._DefaultKVClass()
    
    def get_new_main_kv(self) -> KVStore:
        if self._transient:
            return DictStorage()
        else:
            return self._DefaultKVClass()
    
    @property
    def root(self) -> Path:
        return self._root
    
    def partitions(self) -> TList[str]:
        return self._main_kvs.keys()
    
    def has_partition(self, name:str) -> bool:
        return name in self.partitions()

    def create_partition(self, name:str, main:KVStore=None):
        if main is None or self._transient:
            main = self.get_new_main_kv()
        self._meta_kvs.set(k=name, kv=self.get_new_meta_kv())
        self._main_kvs.set(k=name, kv=main)

    def get_partition_size(self, name: str) -> int:
        return self._meta_kvs[name].size
    
    def get_main_kv(self, partition:str) -> KVStore:
        return self._main_kvs[partition]
    
    def get_meta_kv(self, partition:str) -> KVStore:
        return self._meta_kvs[partition]

    ############################################################################ 
    ###
    ############################################################################ 
    def exists_main(self, loc: BaseObjLocation) -> bool:
        partition, uid = loc.partition, loc.uid
        return self._main_kvs[partition].exists(uid)
    
    def exists_meta(self, loc: BaseObjLocation) -> bool:
        return self._meta_kvs[loc.partition].exists(loc.uid)

    def get(self, loc:PartitionedObjLocation) -> TTuple[TAny, TAny]:
        partition, uid = loc.partition, loc.uid
        main_kv = self._main_kvs[partition]
        meta_kv = self._meta_kvs[partition]
        return (main_kv.get_default(k=uid, default=Forgotten()), 
                meta_kv.get(k=uid))
    
    def get_meta(self, loc: BaseObjLocation) -> TAny:
        partition, uid = loc.partition, loc.uid
        meta_kv = self._meta_kvs[partition]
        return meta_kv.get(k=uid)
    
    def set(self, loc:PartitionedObjLocation, obj:TAny, 
            meta:TAny, skip_obj:bool=False):
        partition, uid = loc.partition, loc.uid
        if not self.has_partition(name=partition):
            self.create_partition(name=partition, main=None)
        main_kv = self._main_kvs[partition]
        meta_kv = self._meta_kvs[partition]
        #! the use of `mset_if_not_exists()` guarantees the invariant that no
        #! location is ever overwritten
        if not skip_obj and (not isinstance(obj, Forgotten)):
            main_kv.set_if_not_exists(k=uid, v=obj)
        meta_kv.set_if_not_exists(k=uid, v=meta)
    
    def delete(self, loc: BaseObjLocation):
        partition, uid = loc.partition, loc.uid
        self._main_kvs[partition].delete(k=uid, must_exist=False)
        self._meta_kvs[partition].delete(k=uid, must_exist=True)
    
    ############################################################################ 
    ###
    ############################################################################ 
    def _group_like_locs(self, locs:TList[BaseObjLocation],
                         objs:TList[TAny]) -> TDict[str, TList[TAny]]:
        """
        Given a list of locations and a matching list of objects, group the 
        objects in a {partition: [objects]} dict where order within each list
        matches the order in the list of objects itself
        """
        return group_like(objs=objs, labels=[loc.partition for loc in locs])
    
    def _order_like_locs(self, locs:TList[BaseObjLocation], 
                        obj_groups:TDict[str, TList[TAny]]) -> TList[TAny]:
        """
        Given a list of locations and a grouping of objects by partition that
        matches this list of locations, return the matching flattened list of
        objects.
        """
        return ungroup_like(groups=obj_groups, labels=[loc.partition for loc in locs])
    
    def mexists_main(self, locs: TList[BaseObjLocation]) -> TList[bool]:
        uid_groups = self._group_like_locs(locs=locs, objs=[loc.uid for loc in locs])
        grouped_results = {partition: self._main_kvs[partition].mexists(uids)
                           for partition, uids in uid_groups.items()}
        return self._order_like_locs(locs=locs, obj_groups=grouped_results)
    
    def mget(self, locs:TList[BaseObjLocation]) -> TTuple[TList[TAny], TList[TAny]]:
        uid_groups = self._group_like_locs(locs=locs, objs=[loc.uid for loc in locs])
        grouped_mains = {partition:
                         self._main_kvs[partition].mget_default(
                             uids, default=Forgotten())
                         for partition, uids in uid_groups.items()}
        grouped_metas = {partition: self._meta_kvs[partition].mget(uids)
                         for partition, uids in uid_groups.items()}
        ord_mains = self._order_like_locs(locs=locs, obj_groups=grouped_mains)
        ord_metas = self._order_like_locs(locs=locs, obj_groups=grouped_metas)
        return ord_mains, ord_metas
    
    def mget_meta(self, locs:TList[BaseObjLocation]) -> TList[TAny]:
        uid_groups = self._group_like_locs(
            locs=locs, objs=[loc.uid for loc in locs])
        grouped_metas = {partition: self._meta_kvs[partition].mget(uids)
                         for partition, uids in uid_groups.items()}
        ord_metas = self._order_like_locs(locs=locs, obj_groups=grouped_metas)
        return ord_metas
    
    def mset(self, mapping:TDict[BaseObjLocation, TAny], 
             meta_mapping:TDict[BaseObjLocation, TAny], 
             skip_obj_mapping:TDict[BaseObjLocation, bool]=None):
        locs = list(mapping.keys())
        mains = [mapping[k] for k in locs]
        metas = [meta_mapping[k] for k in locs]
        if skip_obj_mapping is None:
            skip_obj_mapping = {loc: False for loc in locs}
        skip_objs = [skip_obj_mapping[k] for k in locs]
        main_groups = self._group_like_locs(locs=locs, objs=mains)
        meta_groups = self._group_like_locs(locs=locs, objs=metas)
        skip_obj_groups = self._group_like_locs(locs=locs, objs=skip_objs)
        uid_groups = self._group_like_locs(locs=locs, objs=[loc.uid for loc in locs])
        for partition in main_groups.keys():
            part_uids = uid_groups[partition]
            part_mains = main_groups[partition]
            part_skip_objs = skip_obj_groups[partition]
            part_main_mapping = {uid: obj for uid, obj, skip_obj in zip(
                part_uids, part_mains, part_skip_objs) if not skip_obj}
            part_main_mapping = {uid: obj for uid, obj in part_main_mapping.items()
                                 if not isinstance(obj, Forgotten)}
            #! the use of `mset_if_not_exists()` guarantees the invariant that no
            #! location is ever overwritten
            self._main_kvs[partition].mset_if_not_exists(part_main_mapping)
            part_metas = meta_groups[partition]
            self._meta_kvs[partition].mset_if_not_exists(
                {uid: obj for uid, obj in zip(part_uids, part_metas)})
    
    def mdelete(self, locs: TList[BaseObjLocation]):
        uid_groups = self._group_like_locs(
            locs=locs, objs=[loc.uid for loc in locs])
        for partition, uids in uid_groups.items():
            self._meta_kvs[partition].mdelete(ks=uids, must_exist=True)
            self._main_kvs[partition].mdelete(ks=uids, must_exist=False)
    
    def locs(self, partitions:TList[str]=None) -> TList[BaseObjLocation]:
        res = []
        partitions = self.partitions() if partitions is None else partitions
        for partition in partitions:
            #! using metadata, which is guaranteed to always be there
            keys = self._meta_kvs[partition].keys()
            res += [PartitionedObjLocation(uid=key, partition=partition)
                    for key in keys]
        return res
    
    ############################################################################ 
    def describe(self) -> TDict[str, TAny]:
        res = {
            'partition_sizes': {k: self._meta_kvs[k].size 
                                for k in self.partitions()}
        }
        return res

    @ask(question='Are you sure you want to drop this object storage?', 
         desc_getter=lambda x: x.describe())
    def drop(self, must_exist: bool = True, answer:bool=None):
        if must_exist:
            assert self.root.is_dir()
        shutil.rmtree(path=self.root)

    @ask(question='Are you sure you want to delete all objects from this storage?',
         desc_getter=lambda x: x.describe())
    def delete_all(self, answer:bool=None):
        logging.info('Deleting all data from object storage...')
        locs = self.locs()
        self.mdelete(locs=locs)
    
    def space_usage(self) -> TDict[str, str]:
        return self._main_kvs.space_usage()