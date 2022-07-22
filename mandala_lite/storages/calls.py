from ..common_imports import *
from ..core.model import Call
from .kv import InMemoryStorage


class CallStorage:
    """
    Stores (detached) call objects. 
    
    As soon as a call is executed for the first time, it is placed in the
    `self.temp` call partition. The `Storage.commit()` method is used to move
    all calls in the temp partition to the `self.main` partition, and indexes
    them in the relational storage along the way. 
    
    See also: 
        - `Call.detached()`
        - `Storage.commit()`
    """
    def __init__(self):
        # calls are initially stored here
        self.temp = InMemoryStorage()
        # this is where we move them eventually, and index them into the
        # relational storage along the way
        self.main = InMemoryStorage()
    
    def exists(self, uid:str) -> bool:
        """
        Whether a call has been saved in either the main or temporary partition.
        """
        return self.temp.exists(k=uid) or self.main.exists(k=uid)
    
    def get(self, uid:str) -> Call:
        """
        Get a call if it exists, looking in the temp storage first. 
        """
        if self.temp.exists(k=uid):
            return self.temp.get(k=uid)
        return self.main.get(k=uid)