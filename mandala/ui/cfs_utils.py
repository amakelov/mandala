from ..common_imports import *
from ..core.config import *
from .storage import Storage

################################################################################
### tools to summarize tables of refs
################################################################################
def convert_bytes_to(
    num_bytes: float, units: Literal["bytes", "KB", "MB", "GB"]
) -> float:
    if units == "KB":
        return num_bytes / 1024
    elif units == "MB":
        return num_bytes / (1024**2)
    elif units == "GB":
        return num_bytes / (1024**3)
    elif units == "bytes":
        return num_bytes
    else:
        raise ValueError(f"Unknown units: {units}")


def estimate_uid_storage(
    uids: List[str],
    storage: Storage,
    units: Literal["bytes", "KB", "MB", "GB"] = "bytes",
    sample_size: int = 20,
) -> Tuple[float, float]:
    # sample 5 random elements from the column
    sample_uids = pd.Series(uids).sample(sample_size, replace=True).values
    query = (
        "SELECT length(value) AS size_in_bytes FROM __vrefs__ WHERE __uid__ IN "
        + str(tuple(sample_uids))
    )
    sample_counts = storage.rel_storage.execute_df(query)
    mean, std = (
        sample_counts["size_in_bytes"].astype(int).mean(),
        sample_counts["size_in_bytes"].astype(int).std(),
    )
    # check if std is nan
    if np.isnan(std):
        std = 0
    # convert to requested units
    if units == "KB":
        avg_column_size = mean / 1024
        std = std / 1024
    elif units == "MB":
        avg_column_size = mean / (1024**2)
        std = std / (1024**2)
    elif units == "GB":
        avg_column_size = mean / (1024**3)
        std = std / (1024**3)
    elif units == "bytes":
        avg_column_size = mean
    else:
        raise ValueError(f"Unknown units: {units}")
    # round to 2 decimal places
    avg_column_size = round(avg_column_size, 2)
    std = round(std, 2)
    return avg_column_size, std
