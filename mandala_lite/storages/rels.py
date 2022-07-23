from abc import ABC, abstractmethod
from collections import defaultdict

from pypika import Query, Table

from ..common_imports import *
from ..core.config import Config, Prov
from ..core.model import Call


class RelStorage(ABC):
    """
    Responsible for the low-level (i.e., unaware of mandala-specific concepts)
    interactions with the relational part of the storage, such as creating and
    extending tables, running queries, etc. This is intended to be a pretty
    generic, minimal database interface, supporting just the things we need.

    It's deliberately referred to as "relational storage" as opposed to a
    "relational database" because simpler implementations exist.
    """

    @abstractmethod
    def create_relation(self, name: str, columns: List[str]):
        """
        Create a relation with the given name and columns.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_relation(self, name: str):
        raise NotImplementedError()

    @abstractmethod
    def create_column(self, relation: str, name: str, default_value: str):
        raise NotImplementedError()

    @abstractmethod
    def insert(self, name: str, df: pd.DataFrame):
        """
        Append rows to a table
        """
        raise NotImplementedError()

    @abstractmethod
    def upsert(self, name: str, df: pd.DataFrame):
        """
        Upsert rows in a table based on index
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, name: str, index: List[str]):
        """
        Delete rows from a table based on index
        """
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, table: str) -> pd.DataFrame:
        """
        Fetch data from a table.
        """
        raise NotImplementedError()


RemoteEventLogEntry = dict[str, bytes]


class RelAdapter:
    """
    Responsible for high-level RDBMS interactions, such as
        - taking a bunch of calls and putting their data inside the database;
        - keeping track of data provenance (i.e., putting calls in a table that
          makes it easier to derive the history of a value reference)

    Uses `RelStorage` to do the actual work.
    """

    EVENT_LOG_TABLE = "__event_log__"

    def __init__(self, rel_storage: RelStorage):
        self.rel_storage = rel_storage
        # initialize provenance table
        self.prov_df = pd.DataFrame(
            columns=[
                Prov.call_uid,
                Prov.op_name,
                Prov.op_version,
                Prov.is_super,
                Prov.vref_name,
                Prov.vref_uid,
                Prov.is_input,
            ]
        )
        self.prov_df.set_index([Prov.call_uid, Prov.vref_name, Prov.is_input])
        # Initialize the event log.
        # The event log is just a list of UIDs that changed, for now.
        self.rel_storage.create_relation(self.EVENT_LOG_TABLE, ["table"])

    def upsert_calls(self, calls: List[Call]):
        """
        Upserts calls in the relational storage so that they will show up in
        declarative queries.
        """
        if len(calls) > 0:
            # Write changes to the event log table.

            for name, df in self.tabulate_calls(calls).items():
                self.rel_storage.upsert(name, df)
                self.rel_storage.upsert(
                    self.EVENT_LOG_TABLE,
                    pd.DataFrame.from_dict(
                        {Config.uid_col: df[Config.uid_col], "table": [name] * len(df)}
                    ),
                )
            # update provenance table
            new_prov_df = self.get_provenance_table(calls=calls)
            self.prov_df = upsert_df(current=self.prov_df, new=new_prov_df)

    def bundle_to_remote(self) -> RemoteEventLogEntry:
        # TODO do this transactionally

        # Bundle event log and referenced calls into tables.
        event_log_df = self.rel_storage.get_data(self.EVENT_LOG_TABLE)
        tables_with_changes = {}
        table_names_with_changes = event_log_df["table"].unique()

        event_log_table = Table(self.EVENT_LOG_TABLE)
        for table_name in table_names_with_changes:
            table = Table(table_name)
            tables_with_changes[table_name] = self.rel_storage.execute(
                Query.from_(table)
                .join(event_log_table)
                .on(table[Config.uid_col] == event_log_table[Config.uid_col])
            )

        output = {}
        for table_name in tables_with_changes:
            buffer = io.BytesIO()
            tables_with_changes[table_name].to_parquet(buffer)
            output[table_name] = buffer.getbuffer()

        self.rel_storage.execute(Query.from_(event_log_table).delete())
        return output

    def apply_from_remote(self, changes: list[RemoteEventLogEntry]):
        # TODO do this in a transaction
        for raw_changeset in changes:
            for table_name in raw_changeset:
                buffer = io.BytesIO(raw_changeset[table_name])
                table = pd.read_parquet(buffer)
                self.rel_storage.upsert(table_name, table)

    @staticmethod
    def tabulate_calls(calls: List[Call]) -> Dict[str, pd.DataFrame]:
        """
        Converts call objects to a dictionary of {relation name: table
        to upsert} pairs.
        """
        # split by operation internal name
        calls_by_op = defaultdict(list)
        for call in calls:
            calls_by_op[call.op.sig.internal_name].append(call)
        res = {}
        vref_uids = []
        for k, v in calls_by_op.items():
            res[k] = pd.DataFrame(
                [
                    {
                        Config.uid_col: call.uid,
                        **{k: v.uid for k, v in call.inputs.items()},
                        **{f"output_{i}": v.uid for i, v in enumerate(call.outputs)},
                    }
                    for call in v
                ]
            )
            for call in v:
                vref_uids += [v.uid for v in call.inputs.values()]
                vref_uids += [v.uid for v in call.outputs]
        res[Config.vref_table] = pd.DataFrame({Config.uid_col: list(set(vref_uids))})
        return res

    ############################################################################
    ### provenance
    ############################################################################
    @staticmethod
    def get_provenance_table(calls: List[Call]) -> pd.DataFrame:
        """
        Converts call objects to a dataframe of provenance information. Calls to
        all operations are in the same table. Traversing "backward" in this
        table allows you to reconstruct the history of any value reference.
        """
        dfs = []
        for call in calls:
            call_uid = call.uid
            op_name = call.op.sig.internal_name
            op_version = call.op.sig.version
            input_names = list(call.inputs.keys())
            input_uids = [call.inputs[k].uid for k in input_names]
            in_table = pd.DataFrame(
                {
                    Prov.call_uid: call_uid,
                    Prov.op_name: op_name,
                    Prov.op_version: op_version,
                    Prov.is_super: call.op.sig.is_super,
                    Prov.vref_name: input_names,
                    Prov.vref_uid: input_uids,
                    Prov.is_input: True,
                }
            )
            output_names = list([f"output_{i}" for i in range(len(call.outputs))])
            output_uids = [call.outputs[i].uid for i in range(len(call.outputs))]
            out_table = pd.DataFrame(
                {
                    Prov.call_uid: call_uid,
                    Prov.op_name: op_name,
                    Prov.op_version: op_version,
                    Prov.is_super: call.op.sig.is_super,
                    Prov.vref_name: output_names,
                    Prov.vref_uid: output_uids,
                    Prov.is_input: False,
                }
            )
            df = pd.concat([in_table, out_table], ignore_index=True)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


def upsert_df(current: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Upsert for dataframes with the same columns
    """
    return pd.concat([current, new[~new.index.isin(current.index)]])
