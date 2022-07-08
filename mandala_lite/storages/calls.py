from ..common_imports import *
from ..core.model import Call
from .kv import InMemoryStorage


class CallStorage:
    def __init__(self):
        # calls are initially stored here
        self.temp = InMemoryStorage()
        # this is where we move them eventually, and index them into the
        # relational storage along the way
        self.main = InMemoryStorage()
    
    def exists(self, k:str) -> bool:
        return self.temp.exists(k=k) or self.main.exists(k=k)
    
    def get(self, k:str) -> Call:
        if self.temp.exists(k=k):
            return self.temp.get(k=k)
        return self.main.get(k=k)