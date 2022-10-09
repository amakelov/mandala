import mongomock

from ..common_imports import *
from mandala_lite.all import *
from mandala_lite.ui.main import MODES
from mandala_lite.core.config import Config
from mandala_lite.storages.remote_impls.mongo_impl import MongoRemoteStorage
from mandala_lite.storages.remote_impls.mongo_mock import MongoMockRemoteStorage


def signatures_are_equal(storage_1: Storage, storage_2: Storage) -> bool:
    sigs_1 = storage_1.sig_adapter.load_state()
    sigs_2 = storage_2.sig_adapter.load_state()
    if sigs_1.keys() != sigs_2.keys():
        return False
    for (internal_name, version), sig_1 in sigs_1.items():
        sig_2 = sigs_2[internal_name, version]
        if sig_1 != sig_2:
            return False
    return True


def _sanitize_value(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        if isinstance(value, bytearray):
            return value.hex()
        else:
            raise NotImplementedError(f"Got value of type {type(value)}")


def compare_dfs_as_relations(
    df_1: pd.DataFrame, df_2: pd.DataFrame
) -> Tuple[bool, str]:
    if df_1.shape != df_2.shape:
        return False, f"Shapes differ: {df_1.shape} vs {df_2.shape}"
    if set(df_1.columns) != set(df_2.columns):
        return False, f"Columns differ: {df_1.columns} vs {df_2.columns}"
    # reorder columns of df_2 to match df_1
    df_2 = df_2[df_1.columns]
    # sanitize values to make them hashable
    df_1 = df_1.applymap(_sanitize_value)
    df_2 = df_2.applymap(_sanitize_value)
    # compare as sets of tuples
    result = set(map(tuple, df_1.itertuples(index=False))) == set(
        map(tuple, df_2.itertuples(index=False))
    )
    if result:
        reason = ""
    else:
        reason = f"Dataframe rows differ: {df_1} vs {df_2}"
    return result, reason


def data_is_equal(
    storage_1: Storage, storage_2: Storage, return_reason: bool = False
) -> Union[bool, Tuple[bool, str]]:
    data_1 = storage_1.rel_storage.get_all_data()
    data_2 = storage_2.rel_storage.get_all_data()
    # compare the keys
    if data_1.keys() != data_2.keys():
        result, reason = False, f"Tables differ: {data_1.keys()} vs {data_2.keys()}"
    # compare the signatures
    sigs_1 = storage_1.sig_adapter.load_state()
    sigs_2 = storage_2.sig_adapter.load_state()
    if sigs_1.keys() != sigs_2.keys():
        result, reason = (
            False,
            f"Signature keys differ: {sigs_1.keys()} vs {sigs_2.keys()}",
        )
    if sigs_1 != sigs_2:
        result, reason = False, f"Signatures differ: {sigs_1} vs {sigs_2}"
    # compare the data
    elementwise_comparisons = {
        k: compare_dfs_as_relations(data_1[k], data_2[k])
        for k in data_1.keys()
        if k != Config.schema_table
    }
    if all(result for result, _ in elementwise_comparisons.values()):
        result, reason = True, ""
    else:
        result, reason = (
            False,
            f"Found differences between tables: {elementwise_comparisons}",
        )
    if return_reason:
        return result, reason
    else:
        return result


def check_invariants(storage: Storage):
    # check that signatures match tables
    ui_sigs = storage.sig_adapter.load_ui_sigs()
    call_tables = storage.rel_adapter.get_call_tables()
    columns_by_table = {}
    # collect table columns
    for call_table in call_tables:
        columns = storage.rel_storage._get_cols(relation=call_table)
        columns_by_table[call_table] = columns
    # check that all signatures are accounted for in the tables
    for sig in ui_sigs.values():
        table_name = sig.versioned_ui_name
        assert table_name in columns_by_table
        columns = columns_by_table[table_name]
        assert sig.input_names.issubset(set(columns))
    # check that all tables are accounted for in the signatures
    for call_table, columns in columns_by_table.items():
        input_cols = [
            col
            for col in columns
            if not col.startswith(Config.output_name_prefix) and col != Config.uid_col
        ]
        ui_name, version = Signature.parse_versioned_name(versioned_name=call_table)
        assert set(input_cols).issubset(ui_sigs[ui_name, version].input_names)


def call_matches_signature(call: Call, sig: Signature) -> bool:
    return (
        call.op.sig.ui_name == sig.ui_name
        and set(call.inputs.keys()).issubset(sig.input_names)
        and len(call.outputs) == sig.n_outputs
    )


def _check_invariants(storage: Storage):
    return
    obj_uids = storage.objs.keys()
    call_uids = storage.all_calls()
    calls = [storage.calls.get(uid=uid) for uid in call_uids]
    committed_calls = [storage.calls.get(uid=uid) for uid in storage.calls.main.keys()]
    input_uids = [vref.uid for call in calls for vref in call.inputs.values()]
    output_uids = [vref.uid for call in calls for vref in call.outputs]
    io_uids = input_uids + output_uids

    # check that all calls reference valid objects
    # default values may exist in obj_uids, but not in calls, hence the subset
    # instead of equality
    assert set(io_uids).issubset(obj_uids)

    # check that the signatures stored are consistent with the calls
    for call in calls:
        sig = call.op.sig
        ui_name, version = sig.external_name, sig.version
        stored_sig = storage.sigs[ui_name, version]
        assert call_matches_signature(call, stored_sig)

    # check that the committed calls are accounted for in the relational storage
    committed_tables = storage.rel_adapter.tabulate_calls(calls=committed_calls)
    # TODO: this is complicated by the fact that there may be calls for an old
    # signature of a function (missing some new inputs)
