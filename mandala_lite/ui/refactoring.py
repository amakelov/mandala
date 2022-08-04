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
    sigs = storage.rel_adapter.signature_gets()
    if new_name in [elt[0] for elt in sigs.keys()]:
        raise RuntimeError(
            f"Cannot rename function {func.op.sig.ui_name} to {new_name}, which already exists."
        )
    conn = storage.rel_adapter._get_connection()
    # rename in signature object
    sig = func.op.sig
    new_sig = sig.rename(new_name=new_name)
    storage.rel_adapter.signature_set(sig=new_sig, conn=conn)
    # rename table
    storage.rel_storage.rename_relation(
        name=func.op.sig.versioned_ui_name,
        new_name=new_sig.versioned_ui_name,
        conn=conn,
    )
    # invalidate func
    func.is_synchronized = False
    func.is_invalidated = True
    storage.rel_adapter._end_transaction(conn=conn)


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
    conn = storage.rel_adapter._get_connection()
    # rename in signature object
    sig = func.op.sig
    new_sig = sig.rename_input(name=name, new_name=new_name)
    storage.rel_adapter.signature_set(sig=new_sig, conn=conn)
    # rename in table
    storage.rel_storage.rename_column(
        relation=func.op.sig.versioned_ui_name, name=name, new_name=new_name, conn=conn
    )
    # invalidate func
    func.is_synchronized = False
    func.is_invalidated = True
    storage.rel_adapter._end_transaction(conn=conn)
