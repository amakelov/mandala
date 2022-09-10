from ..common_imports import *
from ..core.config import Config
from ..core.model import FuncOp, ValueRef, Call, wrap
from ..core.utils import Hashing
from ..storages.main import Storage
from ..queries.weaver import ValQuery, FuncQuery
from .context import GlobalContext, MODES


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

    def wrap_inputs(self, inputs: Dict[str, Any]) -> Dict[str, ValueRef]:
        # check if we allow implicit wrapping
        if Config.autowrap_inputs:
            return {k: wrap(v) for k, v in inputs.items()}
        else:
            assert all(isinstance(v, ValueRef) for v in inputs.values())
            return inputs

    def wrap_outputs(self, outputs: List[Any], call_uid: str) -> List[ValueRef]:
        """
        Wrap the outputs of a call as value references.

        If the function happens to return value references, they are returned as
        they are. Otherwise, a UID is assigned depending on the configuration
        settings of the library.
        """
        wrapped_outputs = []
        if Config.output_wrap_method == "content":
            uid_generator = lambda i, x: Hashing.get_content_hash(x)
        elif Config.output_wrap_method == "causal":
            uid_generator = lambda i, x: Hashing.get_content_hash(obj=(call_uid, i))
        else:
            raise ValueError()
        wrapped_outputs = [
            wrap(obj=x, uid=uid_generator(i, x)) if not isinstance(x, ValueRef) else x
            for i, x in enumerate(outputs)
        ]
        return wrapped_outputs

    def bind_inputs(self, args, kwargs, mode:str) -> Dict[str, Any]:
        """
        Given args and kwargs passed by the user from python, this adds defaults
        and returns a dict where they are indexed via internal names.
        """
        bound_args = self.op.py_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs_dict = dict(bound_args.arguments)
        if mode == MODES.query:
            # TODO: add a point constraint for defaults
            for k in inputs_dict.keys():
                if not isinstance(inputs_dict[k], ValQuery):
                    inputs_dict[k] = Q()
        return inputs_dict

    def format_as_outputs(
        self, outputs: List[ValueRef]
    ) -> Union[None, Any, Tuple[Any]]:
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def call_run(
        self, inputs: Dict[str, Union[Any, ValueRef]], storage: Storage
    ) -> Tuple[List[ValueRef], Call]:
        """
        Run the function and return the outputs and the call object.
        """
        if self.is_invalidated:
            raise RuntimeError(
                "This function has been invalidated due to a change in the signature, and cannot be called"
            )
        # wrap inputs
        wrapped_inputs = self.wrap_inputs(inputs)
        # get call UID using *internal names* to guarantee the same UID will be
        # assigned regardless of renamings
        hashable_input_uids = {}
        for k, v in wrapped_inputs.items():
            internal_k = self.op.sig.ui_to_internal_input_map[k]
            if internal_k in self.op.sig._new_input_defaults_uids:
                if self.op.sig._new_input_defaults_uids[internal_k] == v.uid:
                    continue
            hashable_input_uids[internal_k] = v.uid
        call_uid = Hashing.get_content_hash(
            obj=[
                hashable_input_uids,
                self.op.sig.internal_name,
            ]
        )
        # check if call UID exists in call storage
        if storage.call_exists(call_uid):
            # get call from call storage
            call = storage.call_get(call_uid)
            # get outputs from obj storage
            storage.preload_objs([v.uid for v in call.outputs])
            wrapped_outputs = [storage.obj_get(v.uid) for v in call.outputs]
            # return outputs and call
            return wrapped_outputs, call
        else:
            # compute op
            if Config.autounwrap_inputs:
                raw_inputs = {k: v.obj for k, v in wrapped_inputs.items()}
            else:
                raw_inputs = wrapped_inputs
            outputs = self.op.compute(raw_inputs)
            # wrap outputs
            wrapped_outputs = self.wrap_outputs(outputs, call_uid=call_uid)
            # create call
            call = Call(
                uid=call_uid, inputs=wrapped_inputs, outputs=wrapped_outputs, op=self.op
            )
            # save *detached* call in call storage
            storage.call_set(call_uid, call)
            # set inputs and outputs in obj storage
            for v in itertools.chain(wrapped_outputs, wrapped_inputs.values()):
                storage.obj_set(v.uid, v)
            # return outputs and call
            return wrapped_outputs, call

    def call_query(self, inputs: Dict[str, ValQuery]) -> List[ValQuery]:
        if not all(isinstance(inp, ValQuery) for inp in inputs.values()):
            raise NotImplementedError()
        func_query = FuncQuery(op=self.op, inputs=inputs)
        for k, v in inputs.items():
            v.add_consumer(consumer=func_query, consumed_as=k)
        outputs = [
            ValQuery(creator=func_query, created_as=i)
            for i in range(self.op.sig.n_outputs)
        ]
        func_query.set_outputs(outputs=outputs)
        return outputs

    def __call__(self, *args, **kwargs) -> List[ValueRef]:
        context = GlobalContext.current
        if context is None:
            raise RuntimeError()
        if not self.op.sig.has_internal_data:
            new_sig = context.storage.synchronize_many(sigs=[self.op.sig])[0]
            self.op.sig = new_sig
            self.is_synchronized = True
        inputs = self.bind_inputs(args, kwargs, mode=context.mode)
        if context is None:
            raise RuntimeError("No context to call from")
        mode = context.mode
        if mode == MODES.run:
            outputs, call = self.call_run(inputs, context.storage)
            return self.format_as_outputs(outputs=outputs)
        elif mode == MODES.query:
            return self.format_as_outputs(outputs=self.call_query(inputs))
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
    def __init__(self, version: int = 0):
        self.version = version

    def __call__(self, func: Callable) -> "func":
        op = FuncOp(func=func, version=self.version)
        return FuncInterface(op=op)


def op(*args, **kwargs) -> FuncInterface:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # @op case
        op = FuncOp(func=args[0])
        return FuncInterface(op=op)
    else:
        # @op(...) case
        version = kwargs.get("version", 0)
        return FuncDecorator(version=version)


def synchronize(func: FuncInterface, storage: Storage) -> FuncInterface:
    """
    Manually synchronize a function
    """
    new_sig = storage.synchronize_many(sigs=[func.op.sig])[0]
    func.op.sig = new_sig
    func.is_synchronized = True
    return func
