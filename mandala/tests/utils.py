import mongomock
import uuid
import pytest

from ..common_imports import *
from mandala.all import *
from mandala.ui.storage import Storage
from mandala.ui.storage import MODES
from mandala.core.config import Config, is_output_name
from mandala.core.model import Ref
from mandala.core.wrapping import compare_dfs_as_relations
from mandala.storages.remote_impls.mongo_impl import MongoRemoteStorage
from mandala.storages.remote_impls.mongo_mock import MongoMockRemoteStorage
from mandala.storages.rels import RelAdapter


def generate_db_path() -> Path:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/output")
    fname = str(uuid.uuid4()) + ".db"
    return output_dir / fname


def generate_spillover_dir() -> Path:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/output")
    fname = str(uuid.uuid4())
    return output_dir / fname


def generate_path(ext: str) -> Path:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/output")
    fname = str(uuid.uuid4()) + ext
    return output_dir / fname


def generate_storages() -> List[Storage]:
    results = []
    for db_backend in ("sqlite",):
        for persistent in (True, False):
            for spillover in (True, False):
                results.append(
                    Storage(
                        db_backend=db_backend,
                        db_path=generate_db_path() if persistent else None,
                        spillover_dir=generate_spillover_dir() if spillover else None,
                        spillover_threshold_mb=0,
                    )
                )
    return results


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


def data_is_equal(
    storage_1: Storage, storage_2: Storage, return_reason: bool = False
) -> Union[bool, Tuple[bool, str]]:
    data_1 = storage_1.rel_storage.get_all_data()
    data_2 = storage_2.rel_storage.get_all_data()
    #! remove some internal tables from the comparison
    for _internal_table in [RelAdapter.DEPS_TABLE, RelAdapter.PROVENANCE_TABLE]:
        if _internal_table in data_1:
            data_1.pop(_internal_table)
            data_2.pop(_internal_table)
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
        k: compare_dfs_as_relations(data_1[k], data_2[k], return_reason=True)
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
            if not is_output_name(col) and col not in Config.special_call_cols
        ]
        ui_name, version = Signature.parse_versioned_name(versioned_name=call_table)
        assert set(input_cols).issubset(ui_sigs[ui_name, version].input_names)
