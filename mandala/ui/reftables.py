from ..common_imports import *
from ..core.config import *
from .storage import Storage

################################################################################
### tools to summarize tables of refs
################################################################################
def estimate_column_sizes(
    uid_df: pd.DataFrame,
    storage: Storage,
    sample_size: int = 5,
    units: Literal["bytes", "KB", "MB", "GB"] = "bytes",
) -> Dict[str, int]:
    """
    Given a table of UIDs, estimate the average size (in bytes) of the
    serialized values in each column of the table.

    Returns a dictionary of {column_name: average_size_in_bytes} estimated from
    independent samples of `sample_size` elements from each column.

    ! Note that this may give misleading results for structs, which are stored
    only as references to their elements.
    """
    res = {}
    for col in uid_df.columns:
        # sample 5 random elements from the column
        sample_uids = uid_df[col].sample(sample_size).values
        query = (
            "SELECT length(value) AS size_in_bytes FROM __vrefs__ WHERE __uid__ IN "
            + str(tuple(sample_uids))
        )
        sample_counts = storage.rel_storage.execute_df(query)
        avg_column_size = int(sample_counts["size_in_bytes"].astype(int).mean())
        # convert to requested units
        if units == "KB":
            avg_column_size = avg_column_size / 1024
        elif units == "MB":
            avg_column_size = avg_column_size / (1024**2)
        elif units == "GB":
            avg_column_size = avg_column_size / (1024**3)
        res[col] = avg_column_size
    return res
