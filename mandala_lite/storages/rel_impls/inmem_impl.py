from ...common_imports import *
from ...core.config import Config
from ..rels import RelStorage, upsert_df



class InMemRelStorage(RelStorage):
    """
    Responsible for the low-level (i.e., unaware of mandala-specific concepts)
    interactions with the relational part of the storage, such as creating and
    extending tables, running queries, etc. This is intended to be a pretty
    generic, minimal database interface, supporting just the things we need.

    It's deliberately referred to as "relational storage" as opposed to a
    "relational database" because simpler implementations exist.
    """

    def __init__(self):
        # internal name -> dataframe
        self.relations: Dict[str, pd.DataFrame] = {}

    ############################################################################
    ### schema management
    ############################################################################
    def create_relation(self, name: str, columns: List[str]):
        """
        Create a (memoization) table with given columns
        """
        assert Config.uid_col in columns
        self.relations[name] = pd.DataFrame(columns=columns).set_index(Config.uid_col)

    def delete_relation(self, name: str):
        """
        Delete a (memoization) table
        """
        del self.relations[name]

    def create_column(self, relation: str, name: str, default_value: str):
        """
        Add a new column to a table.
        """
        assert name not in self.relations[relation].columns
        self.relations[relation][name] = default_value

    ############################################################################
    ### instance management
    ############################################################################
    def insert(self, name: str, df: pd.DataFrame):
        """
        Append rows to a table
        """
        self.relations[name] = self.relations[name].append(
            df.set_index(Config.uid_col), verify_integrity=True
        )

    def upsert(self, name: str, df: pd.DataFrame):
        """
        Upsert rows in a table based on index
        """
        current = self.relations[name]
        df = df.set_index(Config.uid_col)
        self.relations[name] = upsert_df(current=current, new=df)

    def delete(self, name: str, index: List[str]):
        """
        Delete rows from a table based on index
        """
        self.relations[name] = self.relations[name].drop(labels=index)

    ############################################################################
    ### queries
    ############################################################################
    def select(self, query: Any) -> pd.DataFrame:
        raise NotImplementedError()

