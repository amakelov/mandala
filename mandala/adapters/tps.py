from abc import ABC, abstractmethod

from ..common_imports import *
from ..core.tps import Type
from ..util.common_ut import get_uid


class BaseTypeAdapter(ABC):
    """
    A connector between named types and their internal data
    """
    @abstractmethod
    def has_ui_name(self, ui_name:str) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def has_type(self, tp:Type) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self, ui_name:str, tp:Type) -> str:
        """
        Return the internal name of the type
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def type_dict(self) -> TDict[str, TAny]:
        """
        ui name -> internal type object
        """
        raise NotImplementedError()
    

class TypeAdapter(BaseTypeAdapter):

    def __init__(self) -> None:
        # UI name -> (anonymized) Type
        self._type_dict:TDict[str, Type] = {}
        # Internal type name -> UI name
        self._inv_dict = {}
    
    @property
    def type_dict(self) -> TDict[str, TAny]:
        return self._type_dict
    
    def generate_new_uuid(self) -> str:
        return f'{get_uid()}'
    
    def synchronize(self, ui_name:str, tp:Type) -> str:
        if ui_name in self._type_dict:
            stored = self._type_dict[ui_name]
            tp.set_name(name=stored.name)
            if tp != stored:
                tp._reset_name()
                raise ValueError(f'Type synchronization failed for type {ui_name}: stored {stored}, got {tp}')
            assert stored.is_named
            res = stored.name
        else:
            type_id = self.generate_new_uuid()
            tp.set_name(name=type_id)
            self._type_dict[ui_name] = tp
            self._inv_dict[tp.name] = ui_name
            res = type_id
        return res
    
    def has_ui_name(self, ui_name: str) -> bool:
        return ui_name in self._type_dict

    def has_type(self, tp: Type) -> bool:
        return tp.name in self._inv_dict