from ..common_imports import *
from ..core.config import Config
from ..core.model import FuncOp, ValueRef, Call, wrap
from ..core.utils import Hashing
from .main import Storage
from ..queries.weaver import ValQuery, FuncQuery
from .main import GlobalContext, MODES
from .utils import format_as_outputs, bind_inputs, wrap_inputs, wrap_outputs


class FuncInterface:
    """
    Wrapper around a memoized function.

    This is the object the `@op` decorator converts functions into.
    """

    def __init__(self, op: FuncOp):
        self.op = op
        self.__name__ = self.op.func.__name__
        self.is_synchronized = False
        self.is_invalidated = False

    def invalidate(self):
        self.is_invalidated = True
        self.is_synchronized = False

    def __call__(self, *args, **kwargs) -> Union[None, Any, Tuple[Any]]:
        context = GlobalContext.current
        if context is None:
            return self.op.func(*args, **kwargs)
        if self.is_invalidated:
            raise RuntimeError(
                "This function has been invalidated due to a change in the signature, and cannot be called"
            )
        storage = context.storage
        if not self.op.sig.has_internal_data:
            # synchronize if necessary
            synchronize(func=self, storage=context.storage)
        if Config.check_signature_on_each_call:
            # to prevent stale signatures from being able to make calls.
            # not necessary to ensure correctness at this stage
            is_synced, reason = storage.sig_adapter.is_synced(sig=self.op.sig)
            if not is_synced:
                raise SyncException(reason)
        inputs = bind_inputs(args, kwargs, mode=context.mode, op=self.op)
        if context is None:
            raise RuntimeError("No context to call from")
        mode = context.mode
        if mode == MODES.run:
            outputs, call = storage.call_run(op=self.op, inputs=inputs)
            return format_as_outputs(outputs=outputs)
        elif mode == MODES.query:
            return format_as_outputs(
                outputs=storage.call_query(op=self.op, inputs=inputs)
            )
        elif mode == MODES.batch:
            wrapped_inputs = wrap_inputs(inputs)
            outputs = [
                ValueRef(uid=None, obj=None, in_memory=False)
                for _ in range(self.op.sig.n_outputs)
            ]
            context._call_structs.append((self.op, wrapped_inputs, outputs))
            return format_as_outputs(outputs=outputs)
        else:
            raise ValueError()

    def get_table(self) -> pd.DataFrame:
        storage = GlobalContext.current.storage
        assert storage is not None
        return storage.rel_storage.get_data(table=self.op.sig.versioned_ui_name)


def Q() -> ValQuery:
    """
    Create a `ValQuery` instance.

    Later on, we can add parameters to this to enforce query constraints in a
    more natural way.
    """
    return ValQuery(creator=None, created_as=None)


class FuncDecorator:
    # parametrized version of `@op` decorator
    def __init__(self, version: int = 0, ui_name: Optional[str] = None):
        self.version = version
        self.ui_name = ui_name

    def __call__(self, func: Callable) -> "func":
        op = FuncOp(func=func, version=self.version, ui_name=self.ui_name)
        return FuncInterface(op=op)


def op(*args, **kwargs) -> FuncInterface:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # @op case
        op = FuncOp(func=args[0])
        return FuncInterface(op=op)
    else:
        # @op(...) case
        version = kwargs.get("version", 0)
        ui_name = kwargs.get("ui_name", None)
        return FuncDecorator(version=version, ui_name=ui_name)


def synchronize(func: FuncInterface, storage: Storage):
    """
    Synchronize a function in-place.
    """
    # first, pull the current data from the remote!
    storage.sig_syncer.sync_from_remote()
    new_sig = storage.sig_syncer.sync_from_local(sig=func.op.sig)
    func.op.sig = new_sig
    func.is_synchronized = True


def synchronize_op(op: FuncOp, storage: Storage):
    """
    Synchronize a function in-place.
    """
    # first, pull the current data from the remote!
    storage.sig_syncer.sync_from_remote()
    new_sig = storage.sig_syncer.sync_from_local(sig=op.sig)
    op.sig = new_sig
    op.is_synchronized = True
