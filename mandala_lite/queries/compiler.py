from ..common_imports import *
from .weaver import ValQuery, OpQuery

class Compiler:
    """
    Compile a computational graph of connected `OpQuery` and `ValQuery` objects
    into an SQL query.
    """
    def __init__(self, 
                 val_queries:List[ValQuery], 
                 op_queries:List[OpQuery]):
        self.val_queries = val_queries
        self.op_queries = op_queries
        # OpQuery -> sqlalchemy alias object
        self.op_aliases = {}
        # ValQuery -> sqlalchemy alias object
        self.val_aliases = {}
    
    def compile(self, select_queries:List[ValQuery]) -> Select:
        """
        Given value queries whose corresponding variables want to select as
        columns, this compiles the graph of `ValQuery` and `OpQuery` objects
        into a SQLAlchemy `Select` object that can be executed against the DB
        backend.
        """
        raise NotImplementedError()
