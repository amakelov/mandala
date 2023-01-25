from ..common_imports import *
from typing import Literal


def get_mandala_path() -> Path:
    import mandala

    return Path(os.path.dirname(mandala.__file__))


class Config:
    ### options modifying library behavior
    # whether ops automatically wrap their inputs as value references, or
    # require to be explicitly passed value references
    ### settings
    # whether to automatically wrap inputs when a call is made to an op
    autowrap_inputs = True
    # whether to automatically unwrap inputs when an op is actually executed
    autounwrap_inputs = True
    # how to assign UIDs to outputs
    output_wrap_method = "content"
    # whether to empty the call and vref caches upon committing to the RDBMS
    evict_on_commit = True
    # whether to commit on context exit
    autocommit = True
    # whether signatures are verified against the database each time a function
    # is called
    check_signature_on_each_call = False
    # always create storage with a persistent database
    _persistent_storage_testing = False
    enable_ref_magics = False
    warnings = True
    spillover_threshold_mb = 50

    ### constants
    # used for columns containing UIDs of value references or calls
    uid_col = "__uid__"
    # name for the table that holds the value reference UIDs
    vref_table = "__vrefs__"
    vref_value_col = "value"
    # name for the event log table
    event_log_table = "__event_log__"
    # todo: currently unused?
    # schema_event_log_table = "__schema_event_log__"
    schema_table = "__schema__"
    # table for keeping track of function dependencies
    deps_table = "__deps__"
    provenance_table = "__provenance__"
    # all output names begin with this string
    # todo: prevent creating inputs with this name
    output_name_prefix = "output_"

    ### checking for optional dependencies
    try:
        import dask

        has_dask = True
    except ImportError:
        has_dask = False

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    try:
        import cityhash

        has_cityhash = True
    except ImportError:
        has_cityhash = False

    try:
        import PIL

        has_pil = True
    except ImportError:
        has_pil = False

    try:
        import rich

        has_rich = True
    except ImportError:
        has_rich = False

    if has_rich:
        from rich import pretty

        pretty.install()

    ### some module names needed by internals
    mandala_path = get_mandala_path()
    module_name = "mandala"
    tests_module_name = "mandala.tests"

    # hashing method
    content_hasher: Literal["cityhash", "blake2b"] = (
        "cityhash" if has_cityhash else "blake2b"
    )


def dump_output_name(index: int) -> str:
    return f"{Config.output_name_prefix}{index}"


def parse_output_idx(output_name: str) -> int:
    return int(output_name[len(Config.output_name_prefix) :])


class Prov:
    relname = "__provenance__"
    call_uid = "call_uid"
    op_name = "op_name"
    op_version = "op_version"
    vref_name = "vref_name"
    vref_uid = "vref_uid"
    is_input = "is_input"


if Config.has_torch:
    import torch
