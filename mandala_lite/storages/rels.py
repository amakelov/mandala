from ..common_imports import *

class RelStorage:
    """
    Responsible for the low-level (i.e., unaware of mandala-specific concepts)
    interactions with the RDBMS part of the storage, such as creating and
    extending tables, running queries, etc. This is intended to be a pretty
    generic database interface supporting just the things we need.
    """
    def create_relation(self, name:str, columns:List[str]):
        raise NotImplementedError()
    
    def delete_relation(self, name:str):
        raise NotImplementedError()

    def create_column(self, relation:str, name:str, default_value:str):
        raise NotImplementedError()

    def select(self, query):
        raise NotImplementedError()
    
    def insert(self, name:str, df:pd.DataFrame):
        raise NotImplementedError()

    def delete(self, name:str, index:List[str]):
        raise NotImplementedError()


class RelAdapter:
    """
    Responsible for high-level RDBMS interactions, such as taking a bunch of
    calls and putting their data inside the database; uses `RelStorage` to do
    the actual work. 
    """
    def __init__(self, rel_storage:RelStorage):
        self.rel_storage = rel_storage 

    @staticmethod
    def tabulate_calls(calls:List[Call]) -> pd.DataFrame:
        raise NotImplementedError()

