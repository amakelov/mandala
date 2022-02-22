from abc import ABC, abstractmethod

from ..common_imports import *
from ..util.common_ut import concat_lists


class OpWeave(ABC):
    
    @property
    @abstractmethod
    def inputs(self) -> TDict[str, 'ValWeave']:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def outputs(self) -> TDict[str, 'ValWeave']:
        raise NotImplementedError()
    
    @abstractmethod
    def weave_input(self, name:str, inp:'ValWeave'):
        """
        NOTE: the same value weave may appear multiple times in the inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def weave_output(self, name:str, outp:'ValWeave'):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, **inputs:TDict[str, TAny]) -> TDict[str, 'ValWeave']:
        raise NotImplementedError()

    @property
    def neighbors(self) -> TList['ValWeave']:
        return [x for x in itertools.chain(self.inputs.values(),
                                           self.outputs.values())]
    
    

class ValWeave(ABC):
    
    @property
    @abstractmethod
    def creator(self) -> TOption[OpWeave]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def created_as(self) -> TOption[str]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def consumers(self) -> TList[OpWeave]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def consumed_as(self) -> TList[str]:
        # a list is used because elements need not be unique (so, you can't
        # put this data in a dict)
        raise NotImplementedError()
    
    @abstractmethod
    def set_creator(self, creator:OpWeave, created_as:str):
        raise NotImplementedError()
    
    @abstractmethod
    def add_consumer(self, consumer:OpWeave, consumed_as:str):
        raise NotImplementedError()
    
    @property
    def neighbors(self) -> TList[OpWeave]:
        res = []
        if self.creator is not None:
            res.append(self.creator)
        for cons in self.consumers:
            res.append(cons)
        return res
    

ValClass = typing.TypeVar('ValClass', bound=ValWeave)
OpClass = typing.TypeVar('OpClass', bound=OpWeave)
def traverse_all(val_weaves:TTuple[ValClass,...],
                 op_weaves:TTuple[OpClass,...],
                 ) -> TTuple[TTuple[ValClass,...], TTuple[OpClass,...]]:
    """
    Extend the given weave objects to all objects connected to them.
    """
    val_weaves_ = [_ for _ in val_weaves]
    op_weaves_ = [_ for _ in op_weaves]
    found_new = True
    while found_new:
        found_new = False
        val_neighbors = concat_lists([v.neighbors for v in val_weaves_])
        op_neighbors = concat_lists([o.neighbors for o in op_weaves_])
        if any(k not in op_weaves_ for k in val_neighbors):
            found_new = True
            for neigh in val_neighbors:
                if neigh not in op_weaves_:
                    op_weaves_.append(neigh)
        if any(k not in val_weaves_ for k in op_neighbors):
            found_new = True
            for neigh in op_neighbors:
                if neigh not in val_weaves_:
                    val_weaves_.append(neigh)
    return tuple(val_weaves_), tuple(op_weaves_)

def traverse_dependencies(val_weaves:TList[ValClass], 
                          op_weaves:TList[OpClass],
                          ) -> TTuple[TSet[ValClass], TSet[OpClass]]:
    val_result = set(val_weaves)
    op_result = set(op_weaves)
    found_new = True
    while found_new:
        found_new = False
        val_creators = [v.creator for v in val_result if v.creator is not None]
        op_inputs = concat_lists([list(o.inputs.values()) for o in op_result])
        if any(k not in op_result for k in val_creators):
            found_new = True
            for neigh in val_creators:
                op_result.add(neigh)
        if any(k not in val_result for k in op_inputs):
            found_new = True
            for neigh in op_inputs:
                val_result.add(neigh)
    return val_result, op_result