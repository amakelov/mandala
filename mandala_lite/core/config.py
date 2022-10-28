from ..common_imports import *


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
    # all output names begin with this string
    # todo: prevent creating inputs with this name
    output_name_prefix = "output_"


def dump_output_name(index: int) -> str:
    return f"{Config.output_name_prefix}{index}"


class Prov:
    relname = "__provenance__"
    call_uid = "call_uid"
    op_name = "op_name"
    op_version = "op_version"
    vref_name = "vref_name"
    vref_uid = "vref_uid"
    is_input = "is_input"
