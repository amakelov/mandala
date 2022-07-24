from ..common_imports import *
from mandala_lite.all import *


def call_matches_signature(call: Call, sig: Signature) -> bool:
    return (
        call.op.sig.internal_name == sig.internal_name
        and set(call.inputs.keys()).issubset(sig.internal_input_names)
        and len(call.outputs) == sig.n_outputs
    )


def check_invariants(storage: Storage):
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
        ext_name, version = sig.external_name, sig.version
        stored_sig = storage.sigs[ext_name, version]
        assert call_matches_signature(call, stored_sig)

    # check that the committed calls are accounted for in the relational storage
    committed_tables = storage.rel_adapter.tabulate_calls(calls=committed_calls)
    # TODO: this is complicated by the fact that there may be calls for an old
    # signature of a function (missing some new inputs)
