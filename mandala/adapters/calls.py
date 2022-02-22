from abc import ABC, abstractmethod

from .vals import BaseObjLocation, BaseValAdapter

from ..common_imports import *
from ..core.bases import Call
from ..storages.calls import (
    BaseCallStorage, CallLocation, PartitionedCallLocation, PartitionedCallStorage
)

class BaseCallAdapter(ABC):

    @property
    @abstractmethod
    def val_adapter(self) -> BaseValAdapter:
        raise NotImplementedError()

    @property
    @abstractmethod
    def call_storage(self) -> BaseCallStorage:
        raise NotImplementedError()
    
    @abstractmethod
    def get_location(self, uid:str, metadata:TDict[str, TAny]) -> CallLocation:
        raise NotImplementedError()
    
    def get_input_locs(self, call:Call) -> TDict[str, BaseObjLocation]:
        return {k: self.val_adapter.get_vref_location(v) 
                for k, v in call.inputs.items()}
        
    def get_output_locs(self, call:Call) -> TDict[str, BaseObjLocation]:
        return {k: self.val_adapter.get_vref_location(v)
                for k, v in call.outputs.items()}
    
    def verify_signatures(self):
        """
        Check that all stored calls satisfy the signatures of their operations.
        """
        locs = self.call_storage.locs()
        calls = self.call_storage.mget(locs=locs)
        for call in calls:
            input_types = {k: v.get_type() for k, v in call.inputs.items()}
            output_types = {k: v.get_type() for k, v in call.outputs.items()}
            call.op.sig.check_instance(input_types=input_types,
                                       output_types=output_types)


class CallAdapter(BaseCallAdapter):
    
    def __init__(self, 
                 call_storage:PartitionedCallStorage, 
                 val_adapter:BaseValAdapter):
        self._call_storage = call_storage
        self._val_adapter = val_adapter

    @property
    def val_adapter(self) -> BaseValAdapter:
        return self._val_adapter

    @property
    def call_storage(self) -> BaseCallStorage:
        return self._call_storage
    
    def get_location(self, uid:str, metadata:TDict[str, TAny]) -> CallLocation:
        assert 'partition' in metadata
        return PartitionedCallLocation(uid=uid, partition=metadata['partition'])