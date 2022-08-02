from ..common_imports import *


class Config:
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

    ### constants
    uid_col = "__uid__"
    vref_table = "__vrefs__"
    event_log_table = "__event_log__"


class Prov:
    relname = "__provenance__"
    call_uid = "call_uid"
    op_name = "op_name"
    op_version = "op_version"
    vref_name = "vref_name"
    vref_uid = "vref_uid"
    is_input = "is_input"
