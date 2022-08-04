from ..common_imports import *
from ..storages.main import Storage
from .execution import FuncInterface


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
    # TODO: remote sync logic
    _check_rename_precondition(storage=storage, func=func)
    sig = func.op.sig
    new_sig = sig.rename(new_name=new_name)
    storage.synchronize_many([new_sig])
    func.is_synchronized = False
    func.is_invalidated = True


def rename_arg(storage: Storage, func: FuncInterface, name: str, new_name: str):
    """
    Rename memoized function argument.

    What happens here:
        - check renaming preconditions
        - update signature object
        - rename table
        - invalidate the function (making it impossible to compute with it)
    """
    # TODO: remote sync logic
    _check_rename_precondition(storage=storage, func=func)
    sig = func.op.sig
    new_sig = sig.rename_inputs(mapping={name: new_name})
    storage.synchronize_many([new_sig])
    # invalidate func
    func.is_synchronized = False
    func.is_invalidated = True
