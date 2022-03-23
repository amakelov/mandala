from ..common_imports import *

################################################################################    
### helpers for queries with collection columns
################################################################################    
def explode_lists_as_rows(df:pd.DataFrame, list_cols:TList[str], 
                          fill_value:TAny='') -> pd.DataFrame:
    """
    Based on https://stackoverflow.com/a/45846861/6538618
    """
    # make sure `lst_cols` is a list
    if list_cols and not isinstance(list_cols, list):
        list_cols = [list_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(list_cols)

    # calculate lengths of lists
    lens = df[list_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[list_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in list_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[list_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in list_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

def explode_lists_as_rows_old(df:pd.DataFrame,
                              list_cols:TList[str]) -> pd.DataFrame:
    """
    Given a dataframe where some columns hold list values, "explode" these lists
    along rows. 
    
    Formally, if C is the set of all columns and C_L are the ones holding lists
    that we are exploding, then for each row of the form
        (C_i = v_i, i in C)
    we replace it by rows 
        (C_i = v_i, i in C-C_L, C_i = v_ij, i in C_L)
    where j is the *common length* of the lists v_i for i in C_L.
    
    NOTE: 
    ! this has a bug when the non-list columns are not scalars/hashable
    """
    if not list_cols:
        return df
    all_cols = list(df.columns)
    ### verify equal lengths
    lengths_df = pd.DataFrame({col: df[col].apply(lambda x: len(x)) 
                               for col in list_cols})
    for lengths in lengths_df.itertuples(index=False):
        assert all(lengths[0] == elt for elt in lengths[1:])
    ### 
    nonexplode_cols = [col for col in all_cols if col not in list_cols]
    if nonexplode_cols:
        df = df.set_index(nonexplode_cols)
    ### apply explode to each column
    return df.apply(pd.Series.explode, axis='index').reset_index()

def explode_dicts_as_cols(df:pd.DataFrame, dict_cols:TList[str], 
                          require_same_keys:bool=True) -> pd.DataFrame:
    """
    Given a dataframe and designated columns holding dictionary values, where
    each such column holds dictionaries with the same set of keys, explode each
    such column into multiple columns corresponding to each key.
    """
    ### verify equal keys 
    if require_same_keys:
        for col in dict_cols:
            keys_series = df[col].apply(lambda x: tuple(x.keys()))
            assert len(set(keys_series.values)) in [0, 1]
    result_cols = {}
    for col in df.columns:
        if col in dict_cols:
            col_series = df[col]
            # df.drop(columns=col, inplace=True)
            normalized = pd.json_normalize(col_series, max_level=1)
            for norm_col in normalized.columns:
                result_cols[f'{col}.{norm_col}'] = normalized[norm_col]
        else:
            result_cols[col] = df[col]
    return pd.DataFrame(result_cols)