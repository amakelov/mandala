from ..common_imports import *
from .kv import InMemoryStorage
from .sigs import SigStorage

class Storage:
    """
    Groups together all the components of the storage system. 
    
    Responsible for things that require multiple components to work together,
    e.g. moving calls from the "temporary" partition to the "main" partition.
    """
    def __init__(self, root:Path=None):
        self.root = root
        # where calls committed in the DB are
        self.calls_main = InMemoryStorage()
        # where calls are first added
        self.calls_temp = InMemoryStorage()
        # all objects (inputs and outputs to operations) are saved here
        self.objs = InMemoryStorage()
        # # stores the memoization tables
        # self.rel_storage = RelStorage()
        # # manipulates the memoization tables
        # self.rel_adapter = RelAdapter(rel_storage=self.rel_storage) 
        # # stores the signatures of the operations connected to this storage
        self.sigs = SigStorage()