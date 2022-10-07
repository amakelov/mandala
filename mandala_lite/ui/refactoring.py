from ..common_imports import *
from .main import Storage
from .execution import FuncInterface, synchronize


def _check_rename_precondition(storage: Storage, func: FuncInterface):
    """
    In order to rename function data, the function must be synced with the
    storage, and the storage must be clean
    """
    if not func.is_synchronized:
        raise RuntimeError("Cannot rename while function is not synchronized.")
    if not storage.is_clean:
        raise RuntimeError("Cannot rename while there is uncommited work.")


def rename_func(storage: Storage, func: FuncInterface, new_name: str):
    """
    Rename a memoized function.

    What happens here:
        - check renaming preconditions
        - check there is no name clash with the new name
        - rename the memoization table
        - update signature object
        - invalidate the function (making it impossible to compute with it)
    """
    _check_rename_precondition(storage=storage, func=func)
    storage.sig_syncer.sync_rename_sig(sig=func.op.sig, new_name=new_name)
    func.invalidate()


def rename_arg(storage: Storage, func: FuncInterface, name: str, new_name: str):
    """
    Rename memoized function argument.

    What happens here:
        - check renaming preconditions
        - update signature object
        - rename table
        - invalidate the function (making it impossible to compute with it)
    """
    _check_rename_precondition(storage=storage, func=func)
    storage.sig_syncer.sync_rename_input(
        sig=func.op.sig, input_name=name, new_input_name=new_name
    )
    func.invalidate()
