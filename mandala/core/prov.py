from ..common_imports import *
from .builtins_ import ListRef, DictRef, SetRef

BUILTIN_IDS = [ListRef.builtin_id, DictRef.builtin_id, SetRef.builtin_id]
BUILTIN_OP_IDS = [f"{x}_0" for x in BUILTIN_IDS]


def propagate_struct_provenance(prov_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directions of structural calls in a new column `direction_new` by
    inferring from the data. Currently for backward compatibility.

    The algorithm is as follows:
    - find all the refs that are the direct result if a non-struct op call
    - find all the struct calls where these refs appear as the struct
    - find all the struct calls expressing the items of these structs
    - mark these items as outputs
    - repeat the process for these items, until no new structs are found among them

    Note that this assigns `direction_new` for all calls involved in the structs
    found by this process.

    For every struct call that hasn't been assigned `direction_new` yet, we mark
    the struct as output, and the items (and indices) as inputs.
    """
    prov_df = prov_df.copy()
    prov_df["direction_new"] = [None for _ in range(len(prov_df))]
    nonstruct_outputs_causal_uids = prov_df.query(
        'direction == "output" and op_id not in @BUILTIN_OP_IDS'
    ).causal.values
    structs_df = get_structs_df(prov_df, nonstruct_outputs_causal_uids)
    items_df = get_items_df(prov_df, structs_df.call_causal.values)
    while len(structs_df) > 0:
        # mark only the items (not structs or indices) as outputs
        prov_df["direction_new"][
            prov_df.call_causal.isin(items_df.call_causal)
            & ~(prov_df.name.isin(["lst", "dct", "st", "idx", "key"]))
        ] = "output"
        items_causal_uids = items_df.causal.values
        structs_df = get_structs_df(prov_df, items_causal_uids)
        items_df = get_items_df(prov_df, structs_df.call_causal.values)
    remaining_struct_mask = (
        (prov_df["direction_new"] != prov_df["direction_new"])
        & (prov_df["op_id"].isin(BUILTIN_OP_IDS))
        & (prov_df["name"].isin(["lst", "dct", "st"]))
    )
    prov_df.loc[remaining_struct_mask, "direction_new"] = "output"

    remaining_things = prov_df.query("direction_new != direction_new").index
    prov_df.loc[remaining_things, "direction_new"] = prov_df.query(
        "direction_new != direction_new"
    ).direction
    return prov_df


def get_structs_df(prov_df: pd.DataFrame, causal_uids: Iterable[str]) -> pd.DataFrame:
    """
    Given some causal UIDs and a provenance dataframe, return the sub-dataframe
    where these causal UIDs appear in the role of the struct in a structural
    call
    """
    return prov_df.query(
        'causal in @causal_uids and op_id in @BUILTIN_OP_IDS and name in ["lst", "dct", "st"]'
    )


def get_items_df(
    prov_df: pd.DataFrame, struct_call_uids: Iterable[str]
) -> pd.DataFrame:
    """
    Given some structural causal call UIDs and a provenance dataframe, return
    the sub-dataframe where these structural calls are associated with items
    (elements/values) of the structs
    """
    # get the sub-dataframe for these structural calls containing the items (elts/values) of the structs
    return prov_df.query('call_causal in @struct_call_uids and name in ["elt", "val"]')


def get_idx_df(prov_df: pd.DataFrame, struct_call_uids: Iterable[str]) -> pd.DataFrame:
    return prov_df.query('call_causal in @struct_call_uids and name in ["idx", "key"]')
