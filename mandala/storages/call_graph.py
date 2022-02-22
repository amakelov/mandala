from abc import ABC, abstractmethod
from networkx.algorithms.dag import ancestors

from ..common_imports import *

class BaseCallGraphStorage(ABC):

    @abstractmethod
    def add_edge(self, source:str, target:str):
        raise NotImplementedError()
    
    @abstractmethod
    def delete_node(self, name:str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_nodes(self) -> TList[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def delete_edge(self, source:str, target:str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_neighbors(self, node:str) -> TList[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_callers(self, node:str) -> TList[str]:
        raise NotImplementedError()
    

class CallGraphStorage(BaseCallGraphStorage):
    def __init__(self, root:Path):
        self._root = root
        self.root.mkdir(exist_ok=True)
        
    ### interface implementation
    def delete_node(self, name:str):
        nxg = self._get_graph()
        for source, target, _ in nxg.edges:
            if name in (source, target):
                self.delete_edge(source=source, target=target)
        node_path = self._get_node_path(name=name)
        if node_path.is_dir():
            shutil.rmtree(node_path)
    
    def add_edge(self, source: str, target: str):
        edge_path = self._get_edge_path(source=source, target=target)
        if not edge_path.is_file():
            edge_path.parent.mkdir(exist_ok=True)
            with open(edge_path, 'w') as out_file:
                json.dump(obj={}, fp=out_file)
    
    def delete_edge(self, source: str, target: str):
        edge_path = self._get_edge_path(source=source, target=target)
        if edge_path.is_file():
            os.remove(edge_path)
    
    def get_neighbors(self, node: str) -> TList[str]:
        node_path = self._get_node_path(name=node)
        edge_paths = [node_path / elt for elt in os.listdir(node_path)]
        return [self._get_target_from_edge_path(edge_path=edge_path) 
                for edge_path in edge_paths]
    
    def get_callers(self, node: str) -> TList[str]:
        nxg = self._get_graph()
        if node not in nxg.nodes:
            return []
        else:
            return list(ancestors(G=nxg, source=node))
    
    def get_nodes(self) -> TList[str]:
        nxg = self._get_graph()
        return list(nxg.nodes)

    ### internals
    @property
    def root(self) -> Path:
        return self._root
    
    def _get_node_path(self, name:str) -> Path:
        return self.root / name
    
    def _get_edge_path(self, source:str, target:str) -> Path:
        return self._get_node_path(name=source) / f'{target}.json'

    def _get_target_from_edge_path(self, edge_path:Path) -> str:
        return edge_path.stem
    
    def _get_superop_nodes(self) -> TList[str]:
        return os.listdir(self.root)

    def _get_graph(self) -> nx.MultiDiGraph:
        nxg = nx.MultiDiGraph()
        nodes = self._get_superop_nodes()
        for node in nodes:
            neighbors = self.get_neighbors(node=node)
            for neighbor in neighbors:
                nxg.add_edge(node, neighbor)
        return nxg