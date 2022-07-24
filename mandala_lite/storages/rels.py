from abc import ABC, abstractmethod
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from pypika import Query, Table, Parameter
from pypika.terms import LiteralValue

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
    def create_relation(self, name: str, columns: List[tuple[str, str]]):
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
    def upsert(self, name: str, df: pa.Table):
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
        self.rel_storage.create_relation(self.EVENT_LOG_TABLE, [("table", "varchar")])

    def log_change(self, table: str, key: str):
        self.rel_storage.upsert(
            self.EVENT_LOG_TABLE,
            pa.Table.from_pylist([{Config.uid_col: key, "table": table}])
        )

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
                    pa.Table.from_pydict(
                        {Config.uid_col: df[Config.uid_col], "table": [name] * len(df)}
                    ),
                )
            # update provenance table
            new_prov_df = self.get_provenance_table(calls=calls)
            self.prov_df = upsert_df(current=self.prov_df, new=new_prov_df)

    def _query_call(self, call_uid: str) -> pd.DataFrame:
        all_tables = [
            Query.from_(table_name)
            .where(Table(table_name)[Config.uid_col] == Parameter("$1"))
            .select(Table(table_name)[Config.uid_col], LiteralValue(f"'{table_name}'"))
            for table_name in self.rel_storage.get_tables()
        ]
        query = sum(all_tables[1:], start=all_tables[0])
        return self.rel_storage.execute(query, [call_uid])

    def call_exists(self, call_uid: str) -> bool:
        return len(self._query_call(call_uid))

    def call_get(self, call_uid: str) -> Call:
        row = self._query_call(call_uid).iloc[0]
        table_name = row[1]
        table = Table(table_name)
        query = Query.from_(table).where(table[Config.uid_col] == Parameter("$1"))
        results = self.rel_storage.execute(query, [call_uid])
        return Call.from_row(results.iloc[0])

    def call_set(self, call_uid: str, call: Call) -> None:
        self.obj_sets({vref.uid: vref.obj for vref in list(call.inputs.values()) + call.outputs})
        self.upsert_calls([call])

    def obj_exists(self, keys: Union[str, list[str]]) -> Union[bool, list[bool]]:
        if isinstance(keys, str):
            all_keys = [keys]
        else:
            all_keys = keys
        df = self.obj_gets(all_keys)
        results = [len(df[df[Config.uid_col] == key]) > 0 for key in all_keys]
        if isinstance(keys, str):
            return results[0]
        else:
            return results

    def obj_gets(self, keys: list[str]) -> pd.DataFrame:
        if len(keys) == 0:
            return pd.DataFrame()

        table = Table(Config.vref_table)
        query = (
            Query.from_(table)
            .where(table[Config.uid_col].isin(keys))
            .select(table[Config.uid_col], table.value)
        )
        output = self.rel_storage.execute(query)
        output["value"] = output["value"].map(lambda x: deserialize(bytes(x)))
        return output

    def obj_get(self, key: str) -> Any:
        df = self.obj_gets([key])
        return df.loc[0, "value"]

    def obj_set(self, key: str, value: Any) -> None:
        if self.obj_exists(key):
            return

        buffer = io.BytesIO()
        joblib.dump(value, buffer)
        query = Query.into(Config.vref_table).insert(Parameter("$1"), Parameter("$2"))
        self.rel_storage.execute(query, parameters=[key, buffer.getvalue()])

    def obj_sets(self, kvs: dict[str, Any]) -> None:
        keys = list(kvs.keys())
        indicators = self.obj_exists(keys)
        new_keys = [key for key, indicator in zip(keys, indicators) if not indicator]
        df = pa.Table.from_pylist([{Config.uid_col: key, 'value': serialize(kvs[key])} for key in new_keys])
        self.rel_storage.upsert(Config.vref_table, df)

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
                table = pq.read_table(buffer)
                self.rel_storage.upsert(table_name, table)

    @staticmethod
    def tabulate_calls(calls: List[Call]) -> Dict[str, pa.Table]:
        """
        Converts call objects to a dictionary of {relation name: table
        to upsert} pairs.
        """
        # split by operation internal name
        calls_by_op = defaultdict(list)
        for call in calls:
            calls_by_op[call.op.sig.internal_name].append(call)
        res = {}
        for k, v in calls_by_op.items():
            res[k] = pa.Table.from_pylist(
                [
                    {
                        Config.uid_col: call.uid,
                        **{k: v.uid for k, v in call.inputs.items()},
                        **{f"output_{i}": v.uid for i, v in enumerate(call.outputs)},
                    }
                    for call in v
                ]
            )

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


def serialize(obj: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def deserialize(value: bytes) -> Any:
    buffer = io.BytesIO(value)
    return joblib.load(buffer)
