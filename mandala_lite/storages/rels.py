from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from pypika import Query, Table, Parameter
from pypika.terms import LiteralValue
from duckdb import DuckDBPyConnection as Connection

from ..common_imports import *
from ..core.config import Config, Prov, dump_output_name
from ..core.model import Call
from ..core.sig import Signature
from .rel_impls.duckdb_impl import DuckDBRelStorage
from .rel_impls.utils import Transactable, transaction


RemoteEventLogEntry = dict[str, bytes]


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


class RelAdapter(Transactable):
    """
    Responsible for high-level RDBMS interactions, such as
        - taking a bunch of calls and putting their data inside the database;
        - keeping track of data provenance (i.e., putting calls in a table that
          makes it easier to derive the history of a value reference)

    Uses `RelStorage` to do the actual work.
    """

    EVENT_LOG_TABLE = Config.event_log_table
    VREF_TABLE = Config.vref_table
    SIGNATURES_TABLE = Config.schema_table
    # tables to be excluded from certain operations
    SPECIAL_TABLES = (
        EVENT_LOG_TABLE,
        DuckDBRelStorage.TEMP_ARROW_TABLE,
        SIGNATURES_TABLE,
    )

    def __init__(self, rel_storage: DuckDBRelStorage):
        self.rel_storage = rel_storage
        self.init()

    @transaction()
    def init(self, conn: Optional[Connection] = None):
        self.rel_storage.create_relation(
            name=self.VREF_TABLE,
            columns=[(Config.uid_col, None), ("value", "blob")],
            primary_key=Config.uid_col,
            conn=conn,
        )
        # Initialize the event log.
        # The event log is just a list of UIDs that changed, for now.
        # the UID column stores the vref/call uid, the `table` column stores the
        # table in which this UID is to be found.
        self.rel_storage.create_relation(
            name=self.EVENT_LOG_TABLE,
            columns=[(Config.uid_col, None), ("table", "varchar")],
            primary_key=Config.uid_col,
            conn=conn,
        )
        # The signatures table is a binary dump of the signatures
        self.rel_storage.create_relation(
            name=self.SIGNATURES_TABLE,
            columns=[
                ("index", "int"),
                ("signatures", "blob"),
            ],
            primary_key="index",
            conn=conn,
        )

    @transaction()
    def get_call_tables(self, conn: Optional[Connection] = None) -> List[str]:
        tables = self.rel_storage.get_tables(conn=conn)
        return [
            t for t in tables if t not in self.SPECIAL_TABLES and t != self.VREF_TABLE
        ]

    ############################################################################
    ### event log stuff
    ############################################################################
    # def log_change(self, table: str, key: str):
    #     self.rel_storage.upsert(
    #         self.EVENT_LOG_TABLE,
    #         pa.Table.from_pylist([{Config.uid_col: key, "table": table}]),
    #     )

    @transaction()
    def get_event_log(self, conn: Optional[Connection] = None) -> pd.DataFrame:
        return self.rel_storage.get_data(table=self.EVENT_LOG_TABLE, conn=conn)

    @transaction()
    def event_log_is_clean(self, conn: Optional[Connection] = None) -> bool:
        return self.rel_storage.get_count(table=self.EVENT_LOG_TABLE, conn=conn) == 0

    @transaction()
    def clear_event_log(self, conn: Optional[Connection] = None):
        event_log_table = Table(self.EVENT_LOG_TABLE)
        self.rel_storage.execute_no_results(
            query=Query.from_(event_log_table).delete(), conn=conn
        )

    ############################################################################
    ### `Transactable` interface
    ############################################################################
    def _get_connection(self) -> Connection:
        return self.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_storage._end_transaction(conn=conn)

    ############################################################################
    ### call methods
    ############################################################################
    @staticmethod
    def tabulate_calls(calls: List[Call]) -> Dict[str, pa.Table]:
        """
        Converts call objects to a dictionary of {relation name: table
        to upsert} pairs.
        """
        # split by operation internal name
        calls_by_op = defaultdict(list)
        for call in calls:
            calls_by_op[call.op.sig.versioned_ui_name].append(call)
        res = {}
        for k, v in calls_by_op.items():
            res[k] = pa.Table.from_pylist(
                [
                    {
                        Config.uid_col: call.uid,
                        **{k: v.uid for k, v in call.inputs.items()},
                        **{
                            dump_output_name(index=i): v.uid
                            for i, v in enumerate(call.outputs)
                        },
                    }
                    for call in v
                ]
            )

        return res

    @transaction()
    def upsert_calls(self, calls: List[Call], conn: Optional[Connection] = None):
        """
        Upserts calls in the relational storage so that they will show up in
        declarative queries.
        """
        if len(calls) > 0:  # avoid dealing with empty dataframes
            for table_name, ta in self.tabulate_calls(calls).items():
                self.rel_storage.upsert(relation=table_name, ta=ta, conn=conn)
                # Write changes to the event log table
                self.rel_storage.upsert(
                    relation=self.EVENT_LOG_TABLE,
                    ta=pa.Table.from_pydict(
                        {
                            Config.uid_col: ta[Config.uid_col],
                            "table": [table_name] * len(ta),
                        }
                    ),
                    conn=conn,
                )

    @transaction()
    def _query_call(self, call_uid: str, conn: Optional[Connection] = None) -> pa.Table:
        all_tables = [
            Query.from_(table_name)
            .where(Table(table_name)[Config.uid_col] == Parameter("$1"))
            .select(Table(table_name)[Config.uid_col], LiteralValue(f"'{table_name}'"))
            for table_name in self.get_call_tables()
        ]
        query = sum(all_tables[1:], start=all_tables[0])
        return self.rel_storage.execute_arrow(query, [call_uid], conn=conn)

    @transaction()
    def call_exists(self, call_uid: str, conn: Optional[Connection] = None) -> bool:
        return len(self._query_call(call_uid, conn=conn)) > 0

    @transaction()
    def call_get(self, call_uid: str, conn: Optional[Connection] = None) -> Call:
        row = self._query_call(call_uid, conn=conn).take([0])
        table_name = row.column(1)[0]
        table = Table(table_name)
        query = (
            Query.from_(table)
            .where(table[Config.uid_col] == Parameter("$1"))
            .select(table.star)
        )
        results = self.rel_storage.execute_arrow(query, [call_uid], conn=conn)
        return Call.from_row(results)

    @transaction()
    def call_set(
        self, call_uid: str, call: Call, conn: Optional[Connection] = None
    ) -> None:
        self.obj_sets(
            {vref.uid: vref.obj for vref in list(call.inputs.values()) + call.outputs}
        )
        self.upsert_calls([call], conn=conn)

    ############################################################################
    ### object methods
    ############################################################################
    @transaction()
    def obj_gets(
        self, uids: list[str], conn: Optional[Connection] = None
    ) -> pd.DataFrame:
        if len(uids) == 0:
            return pd.DataFrame()

        table = Table(Config.vref_table)
        query = (
            Query.from_(table)
            .where(table[Config.uid_col].isin(uids))
            .select(table[Config.uid_col], table.value)
        )
        output = self.rel_storage.execute_arrow(query, conn=conn).to_pandas()
        output["value"] = output["value"].map(lambda x: deserialize(bytes(x)))
        return output

    @transaction()
    def obj_exists(
        self, uids: Union[str, list[str]], conn: Optional[Connection] = None
    ) -> Union[bool, list[bool]]:
        if isinstance(uids, str):
            all_uids = [uids]
        else:
            all_uids = uids
        df = self.obj_gets(all_uids, conn=conn)
        results = [len(df[df[Config.uid_col] == uid]) > 0 for uid in all_uids]
        if isinstance(uids, str):
            return results[0]
        else:
            return results

    @transaction()
    def obj_get(self, uid: str, conn: Optional[Connection] = None) -> Any:
        df = self.obj_gets(uids=[uid], conn=conn)
        return df.loc[0, "value"]

    @transaction()
    def obj_set(self, uid: str, value: Any, conn: Optional[Connection] = None) -> None:
        if self.obj_exists(uids=[uid], conn=conn):
            return  # guarantees no overwriting of values
        serialized_value = serialize(value)
        query = Query.into(Config.vref_table).insert(Parameter("$1"), Parameter("$2"))
        self.rel_storage.execute_arrow(
            query=query, parameters=[uid, serialized_value], conn=conn
        )
        log_query = Query.into(self.EVENT_LOG_TABLE).insert(
            Parameter("$1"), Parameter("$2")
        )
        self.rel_storage.execute_arrow(
            log_query, parameters=[uid, Config.vref_table], conn=conn
        )

    @transaction()
    def obj_sets(self, kvs: dict[str, Any], conn: Optional[Connection] = None) -> None:
        uids = list(kvs.keys())
        indicators = self.obj_exists(uids, conn=conn)
        new_uids = [uid for uid, indicator in zip(uids, indicators) if not indicator]
        ta = pa.Table.from_pylist(
            [
                {Config.uid_col: new_uid, "value": serialize(kvs[new_uid])}
                for new_uid in new_uids
            ]
        )
        self.rel_storage.upsert(relation=Config.vref_table, ta=ta, conn=conn)
        log_ta = pa.Table.from_pylist(
            [
                {Config.uid_col: new_uid, "table": Config.vref_table}
                for new_uid in new_uids
            ]
        )
        self.rel_storage.upsert(relation=self.EVENT_LOG_TABLE, ta=log_ta, conn=conn)
