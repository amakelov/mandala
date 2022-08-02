from ..common_imports import *
from ..storages.main import Storage
from .execution import FuncInterface

def _check_rename_precondition(storage:Storage, func:FuncInterface):
    if not func.is_synchronized:
        raise RuntimeError("Cannot rename while function is not synchronized.")
    if not storage.is_clean:
        raise RuntimeError("Cannot rename while there is uncommited work.")

def rename_func(storage:Storage, func:FuncInterface, new_name:str):
    """
    Rename a memoized function.
    
    What happens here:
        - check that the storage is in a clean state
        - check that this function already exists in the storage
        - check there is no name clash with the new name
        - rename the memoization table for 
    """
    _check_rename_precondition(storage=storage, func=func)
    sigs = storage.rel_adapter.load_signatures()
    if new_name in [elt[0] for elt in sigs.keys()]:
        raise RuntimeError(f"Cannot rename function {func.op.sig.name} to {new_name}, which already exists.")
    conn = storage.rel_adapter._get_connection()
    # rename in signature object
    sig = func.op.sig
    new_sig = sig.rename(new_name=new_name)
    print('Doing the thing')
    storage.rel_adapter.write_signature(sig=new_sig, conn=conn)
    # rename table
    storage.rel_storage.rename_relation(name=func.op.sig.versioned_name, 
                                        new_name=new_name, conn=conn)
    # invalidate func
    func.is_synchronized = False
    storage.rel_adapter._end_transaction(conn=conn)

def rename_arg(storage:Storage, func:FuncInterface, name:str, new_name:str):
    _check_rename_precondition(storage=storage, func=func)
    if new_name in func.op.sig.input_names:
        raise RuntimeError(f"Cannot rename argument {name} to {new_name}, which already exists.")
    conn = storage.rel_adapter._get_connection()
    # rename in signature object
    sig = func.op.sig
    new_sig = sig.rename_input(name=name, new_name=new_name)
    storage.rel_adapter.write_signature(sig=new_sig, conn=conn)
    # rename in table
    storage.rel_storage.rename_column(relation=func.op.sig.versioned_name,
                                    name=name, new_name=new_name, conn=conn)
    # invalidate func
    func.is_synchronized = False
    storage.rel_adapter._end_transaction(conn=conn)