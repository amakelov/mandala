from abc import ABC, abstractmethod

from .kv import KVGroup, KVStore
from .kv_impl.joblib_impl import JoblibStorage
from .kv_impl.dict_impl import DictStorage

from ..common_imports import *
from ..core.config import CALLS
from ..util.shell_ut import ask
from ..util.common_ut import group_like, ungroup_like


class CallLocation(ABC):
    @property
    @abstractmethod
    def uid(self) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def partition(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def moved(self, new_partition:str) -> 'CallLocation':
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        return f'CallLocation(uid={self.uid}, partition={self.partition})'


class BaseCallStorage(ABC):
    @abstractmethod
    def exists(self, loc:CallLocation, allow_fallback:bool=False) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def create(self, loc:CallLocation, call_data:TAny):
        raise NotImplementedError()
    
    @abstractmethod
    def set_if_not_exists(self, loc:CallLocation, call_data:TAny):
        raise NotImplementedError()
    
    @abstractmethod
    def get(self, loc:CallLocation, allow_fallback:bool=False) -> TAny:
        raise NotImplementedError()
    
    @abstractmethod
    def delete(self, loc:CallLocation):
        raise NotImplementedError()
    
    ############################################################################ 
    ### mmethods
    ############################################################################ 
    @abstractmethod
    def mexists(self, locs:TList[CallLocation],
                allow_fallback:bool=False) -> TList[bool]:
        raise NotImplementedError()
    
    @abstractmethod
    def mset_if_not_exists(self, locs:TList[CallLocation],
                           call_datas:TList[TAny]):
        raise NotImplementedError()
    
    def mcreate(self, locs:TList[CallLocation],
                call_datas:TList[TAny]):
        raise NotImplementedError()
        
    def mget(self, locs:TList[CallLocation],
             allow_fallback:bool=True) -> TList[TAny]:
        raise NotImplementedError()
    
    @abstractmethod
    def mdelete(self, locs:TList[CallLocation]):
        raise NotImplementedError()
    
    @abstractmethod
    def delete_all(self, answer:bool=None):
        """
        Delete all locations from this storage
        """
        raise NotImplementedError()
    
    @abstractmethod
    def lookup_partitions(self, uids:TList[str]) -> TList[str]:
        """
        Given a list of uids, return a list of partitions they belong to, with
        None when no partition is found.
        """
        raise NotImplementedError()
            
    ############################################################################ 
    @abstractmethod
    def drop(self, must_exist:bool=True, answer:bool=None):
        raise NotImplementedError()
    
    @abstractmethod
    def locs(self, partitions:TList[str]=None) -> TList[CallLocation]:
        raise NotImplementedError()
    
    @abstractmethod
    def partitions(self) -> TList[str]:
        raise NotImplementedError()

    @abstractmethod
    def create_partition(self, name:str, kv:KVStore):
        raise NotImplementedError()
    
    @abstractmethod
    def is_empty(self, partition:str) -> bool:
        raise NotImplementedError()
    
    @abstractmethod    
    def get_kv(self, partition:str) -> KVStore:
        raise NotImplementedError()

################################################################################
### implementation
################################################################################
class PartitionedCallLocation(CallLocation):

    def __init__(self, uid:str, partition:str=CALLS.main_partition):
        self._uid = uid
        self._partition = partition
    
    @property
    def uid(self) -> str:
        return self._uid
    
    @property
    def partition(self) -> str:
        return self._partition
    
    def moved(self, new_partition:str) -> 'PartitionedCallLocation':
        return PartitionedCallLocation(uid=self.uid, partition=new_partition)


class PartitionedCallStorage(BaseCallStorage):

    def __init__(self, root:Path, transient:bool=False, 
                 default_kv_class:TType[KVStore]=None):
        self._root = root
        self._transient = transient
        if default_kv_class is None:
            default_kv_class = JoblibStorage
        self._DefaultKVClass = (default_kv_class if not self._transient
                                else DictStorage)
        self._kvs = KVGroup(root=self._root, default_kv=self._DefaultKVClass())
        self._kvs[CALLS.main_partition] = self._DefaultKVClass()
    
    @property
    def root(self) -> Path:
        return self._root
    
    def get(self, loc:PartitionedCallLocation,
            allow_fallback:bool=False) -> TAny:
        uid = loc.uid
        partition = loc.partition
        if not allow_fallback:
            return self._kvs[partition].get(uid)
        else:
            part_kv = self._kvs[partition]
            if part_kv.exists(k=uid):
                return part_kv.get(k=uid)
            else:
                main_kv = self._kvs[CALLS.main_partition]
                return main_kv.get(k=uid)
    
    def set_if_not_exists(self, loc: CallLocation, call_data: TAny):
        uid = loc.uid
        partition = loc.partition
        assert uid is not None
        self._kvs[partition].set_if_not_exists(k=uid, v=call_data)
    
    def create(self, loc:PartitionedCallLocation, call_data:TAny):
        uid = loc.uid
        partition = loc.partition
        assert uid is not None
        self._kvs[partition].create(k=uid, v=call_data)
    
    def delete(self, loc: CallLocation):
        uid, partition = loc.uid, loc.partition
        self._kvs[partition].delete(k=uid)

    def exists(self, loc:PartitionedCallLocation, allow_fallback:bool=False) -> bool:
        if not allow_fallback:
            return self._kvs[loc.partition].exists(k=loc.uid)
        else:
            uid, partition = loc.uid, loc.partition
            part_kv = self._kvs[partition]
            if part_kv.exists(k=uid):
                return True
            else:
                main_kv = self._kvs[CALLS.main_partition]
                return main_kv.exists(k=uid)
    
    ############################################################################ 
    ### mmethods
    ############################################################################ 
    def _group_like_locs(self, locs:TList[CallLocation], 
                         objs:TList[TAny]) -> TDict[str, TList[TAny]]:
        """
        Given a list of locations and a matching list of objects, group the 
        objects in a {partition: [objects]} dict where order within each list
        matches the order in the list of objects itself
        """
        return group_like(objs=objs, labels=[loc.partition for loc in locs])
    
    def _order_like_locs(self, locs:TList[CallLocation], 
                        obj_groups:TDict[str, TList[TAny]]) -> TList[TAny]:
        """
        Given a list of locations and a grouping of objects by partition that
        matches this list of locations, return the matching flattened list of
        objects.
        """
        return ungroup_like(groups=obj_groups,
                            labels=[loc.partition for loc in locs])
    
    def mexists(self, locs:TList[CallLocation], 
                allow_fallback:bool=False) -> TList[bool]:
        """
        Return presence mask for the given locations in this storage with 
        optional falling back to the main partition.
        
        Notes:
            - this could be optimized further for fallbacks in the case when all
            calls were already found in the partition(s) associated with the
            locations.
        """
        uids_by_partition = self._group_like_locs(locs=locs,
                                                  objs=[loc.uid for loc in locs])
        # will hold {partition: mexists mask for this partition's locations}
        partition_masks = {} 
        if not allow_fallback:
            for partition, uids in uids_by_partition.items():
                partition_masks[partition] = self._kvs[partition].mexists(ks=uids)
        else:
            for partition, uids in uids_by_partition.items():
                mask = self._kvs[partition].mexists(ks=uids)
                fallback_mask = self._kvs[CALLS.main_partition].mexists(ks=uids)
                partition_masks[partition] = [mask_elt or fallback_elt 
                                              for mask_elt, fallback_elt in 
                                              zip(mask, fallback_mask)]
        return self._order_like_locs(locs=locs, obj_groups=partition_masks)
    
    def _mget_with_fallback(self, partition_kv:KVStore, main_kv:KVStore,
                            uids:TList[str]) -> TList[TAny]:
        min_partition = partition_kv.mexists(ks=uids)
        if all(min_partition):
            return partition_kv.mget(ks=uids)
        else:
            min_main = main_kv.mexists(ks=uids)
            assert all(in_p or in_m 
                       for in_p, in_m in zip(min_partition, min_main))
            labels = ['partition' if in_p else 'main' for in_p in min_partition]
            groups = group_like(objs=uids, labels=labels)
            result_groups = {
                'partition': partition_kv.mget(ks=groups['partition']),
                'main': main_kv.mget(ks=groups['main'])
            }
            return ungroup_like(groups=result_groups, labels=labels)
    
    def mget(self, locs:TList[CallLocation],
             allow_fallback:bool=True) -> TList[TAny]:
        uid_groups = self._group_like_locs(locs=locs, 
                                           objs=[loc.uid for loc in locs])
        grouped_res = {}
        for partition, uids in uid_groups.items():
            if not allow_fallback:
                grouped_res[partition] = self._kvs[partition].mget(uids)
            else:
                grouped_res[partition] = self._mget_with_fallback(
                    partition_kv=self._kvs[partition],
                    main_kv=self._kvs[CALLS.main_partition],
                    uids=uids
                )
        ord_res = self._order_like_locs(locs=locs, obj_groups=grouped_res)
        return ord_res
    
    def mset_if_not_exists(self, locs:TList[CallLocation], 
                           call_datas:TList[TAny]):
        data_groups = self._group_like_locs(locs=locs, objs=call_datas)
        uid_groups = self._group_like_locs(locs=locs,
                                           objs=[loc.uid for loc in locs])
        for partition in data_groups:
            uids = uid_groups[partition]
            datas = data_groups[partition]
            self._kvs[partition].mset_if_not_exists(
                {uid: obj for uid, obj in zip(uids, datas)}
            )
    
    def mcreate(self, locs: TList[CallLocation], call_datas: TList[TAny]):
        data_groups = self._group_like_locs(locs=locs, objs=call_datas)
        uid_groups = self._group_like_locs(locs=locs,
                                           objs=[loc.uid for loc in locs])
        for partition in data_groups:
            uids = uid_groups[partition]
            datas = data_groups[partition]
            self._kvs[partition].mcreate(
                mapping={uid: obj for uid, obj in zip(uids, datas)}
            )
    
    def mdelete(self, locs: TList[CallLocation]):
        uid_groups = self._group_like_locs(locs=locs, 
                                           objs=[loc.uid for loc in locs])
        for partition, uids in uid_groups.items():
            self._kvs[partition].mdelete(ks=uids)
        
    def lookup_partitions(self, uids:TList[str]) -> TList[str]:
        result = np.full(len(uids), fill_value=None)
        for partition in self.partitions():
            mask = self._kvs[partition].mexists(ks=uids)
            result[mask] = partition
        if None in result:
            raise ValueError()
        return result.tolist()
        
    ############################################################################ 
    def locs(self, partitions:TList[str]=None) -> TList[CallLocation]:
        res = []
        if partitions is None:
            partitions = list(self._kvs.keys())
        for partition in partitions:
            res += [PartitionedCallLocation(
                partition=partition, uid=uid
            ) for uid in self._kvs[partition].keys()]
        return res
    
    def partitions(self) -> TList[str]:
        return self._kvs.keys()
    
    def create_partition(self, name: str, kv: KVStore):
        self._kvs.set(k=name, kv=kv)
    
    def is_empty(self, partition: str) -> bool:
        keys = self._kvs.keys()
        if partition not in keys:
            raise ValueError(f'Partition {partition} does not exist')
        kv = self._kvs[partition]
        return kv.empty
    
    def get_kv(self, partition:str) -> KVStore:
        return self._kvs[partition]
    ### 
    def describe(self) -> TDict[str, TAny]:
        partition_sizes = {k: self._kvs[k].size for k in self.partitions()}
        res = {
            'path': self.root,
            'partitions': self._kvs.keys(),
            'partition_sizes': partition_sizes
        }
        return res

    @ask(question='Are you sure you want to delete this call storage?', 
         desc_getter=lambda x: x.describe())
    def drop(self, must_exist: bool = True, answer:bool=None):
        if must_exist:
            assert self.root.is_dir()
        shutil.rmtree(path=self.root)
    
    @ask(question='Are you sure you want to delete all calls from this storage?',
         desc_getter=lambda x: x.describe())
    def delete_all(self, answer:bool=None):
        logging.info('Deleting all data from call storage...')
        self.mdelete(locs=self.locs())