from .common_imports import *


def serialize(obj: Any) -> bytes:
    """
    ! this may lead to different serializations for objects x, y such that x
    ! == y in Python. This is because of things like set ordering, which is not
    ! determined by the contents of the set. For example, {1, 2} and {2, 1} would
    ! `serialize()` to different things, but they would be equal in Python.
    """
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def deserialize(value: bytes) -> Any:
    buffer = io.BytesIO(value)
    return joblib.load(buffer)


def _rename_cols_pandas(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping, inplace=False)


def _rename_cols_arrow(table: pa.Table, mapping: Dict[str, str]) -> pa.Table:
    columns = table.column_names
    new_columns = [mapping.get(col, col) for col in columns]
    table = table.rename_columns(new_columns)
    return table


def _rename_cols(table: TableType, mapping: Dict[str, str]) -> TableType:
    if isinstance(table, pd.DataFrame):
        return _rename_cols_pandas(df=table, mapping=mapping)
    elif isinstance(table, pa.Table):
        return _rename_cols_arrow(table=table, mapping=mapping)
    else:
        raise NotImplementedError(f"rename_cols not implemented for {type(table)}")


def ask_user(question: str, valid_options: List[str]) -> str:
    """
    Ask the user a question and return their response.
    """
    prompt = f"{question} "
    while True:
        print(prompt)
        response = input().strip().lower()
        if response in valid_options:
            return response
        else:
            print(f"Invalid response: {response}")
