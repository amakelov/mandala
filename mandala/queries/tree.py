from abc import ABC, abstractmethod
from .weaver_bases import traverse_dependencies

from ..common_imports import *
from ..core.config import EnvConfig
if EnvConfig.has_rich:
    import rich
    from rich.tree import Tree as RichTree
    from rich.pretty import Pretty
from ..queries.rel_weaver import ValQuery, OpQuery

QueryObj = TUnion[ValQuery, OpQuery]

################################################################################
### branching
################################################################################
class BaseQTree(ABC):

    @property
    @abstractmethod
    def parent(self) -> TOption['BaseQTree']:
        raise NotImplementedError()
    
    @parent.setter
    @abstractmethod
    def parent(self, other:'BaseQTree'):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def children(self) -> TList['BaseQTree']:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def qobjs(self) -> TList[QueryObj]:
        """
        The query objects associated with this tree's root node
        """
        raise NotImplementedError()
    
    @abstractmethod
    def add_child(self, tree:'BaseQTree'):
        """
        Append a child to the list of child trees and link its parent to `self`
        """
        raise NotImplementedError()
    
    @abstractmethod
    def add_qobj(self, qobj:QueryObj):
        """
        Append a query object to the contents of this tree's root node
        """
        raise NotImplementedError()

    ############################################################################ 
    ### 
    ############################################################################ 
    @property
    def depth(self) -> int:
        """
        The distance of this tree's root node from the absolute root node
        """
        if not self.children:
            return 0
        else:
            return max([1 + child.depth for child in self.children])

    def get_root(self) -> 'BaseQTree':
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()
    
    def get_constituents_index(self) -> TDict[QueryObj, 'BaseQTree']:
        """
        Get a dictionary of {query obj: tree} pairs for all queries contained in
        this tree AND in the trees below it
        """
        result = {}
        for child in self.children:
            child_result = child.get_constituents_index()
            result.update(child_result)
        result.update({qobj: self for qobj in self.qobjs})
        return result
    
    def get_path_to_root(self, include_self:bool=True) -> TList['BaseQTree']:
        res = [self] if include_self else []
        if self.parent is not None:
            res = res + self.parent.get_path_to_root(include_self=True)
        return res
    
    @property
    def size(self) -> int:
        return len(self.qobjs) + sum([child.size for child in self.children])
    


def induced_queries(root:BaseQTree, 
                    val_queries:TList[ValQuery]) -> TList[QueryObj]:
    """
    Given some value queries spread around a set S of nodes of the tree,
    return all the queries contained in the nodes S.
    """
    assert root.parent is None
    constituents_index = root.get_constituents_index()
    # find history closure of the queries
    val_deps, op_deps = traverse_dependencies(val_weaves=val_queries,
                                              op_weaves=[])
    res = {}
    for vq in val_deps:
        containing_tree = constituents_index[vq]
        res.update({x: None for x in containing_tree.qobjs})
    return list(res.keys())


class QTree(BaseQTree):
    
    def __init__(self, parent:'QTree'=None) -> None:
        self._children = []
        self._parent = parent
        self._qobjs = []

    @property
    def parent(self) -> TOption['QTree']:
        return self._parent
    
    @parent.setter
    def parent(self, other:'QTree'):
        self._parent = other
    
    @property
    def children(self) -> TList['QTree']:
        return self._children
    
    @property
    def qobjs(self) -> TList[QueryObj]:
        return self._qobjs

    def add_child(self, tree:'QTree'):
        self._children.append(tree)
        tree.parent = self
    
    def add_qobj(self, qobj: QueryObj):
        self._qobjs.append(qobj)
    
    def _str_default(self, level=0) -> str:
        ret = "\t"*level+repr(self)+"\n"
        for child in self.children:
            ret += child._str_default(level+1)
        return ret
    
    def __str__(self) -> str:
        return f'QueryTree on {self.size} query objects'
        
    def show(self):
        if EnvConfig.has_rich:
            rich.print(get_rich_tree(self))
        else:
            print(self._str_default(level=0))

    def __repr__(self) -> str:
        return f'QueryTree on {self.size} query objects'
    

if EnvConfig.has_rich:
    def get_rich_tree(qtree:BaseQTree) -> RichTree:
        res = RichTree(label=Pretty(qtree.qobjs))
        for child in qtree.children:
            res.add(get_rich_tree(qtree=child))
        return res