from ..common_imports import *
from ..core.config import Config
from ..core.model import FuncOp, ValueRef, Call, wrap
from ..core.utils import Hashing
from ..storages.main import Storage
from ..queries.weaver import ValQuery, FuncQuery


class FuncInterface:
    """
    Wrapper around a memoized function.

    This is the object the `@op` decorator converts functions into.
    """

    def __init__(self, op: FuncOp, storage: Storage):
        self.op = op
        self.storage = storage

    def wrap_inputs(self, inputs: Dict[str, Any]) -> Dict[str, ValueRef]:
        # "explicit" superops must always receive wrapped inputs
        if self.op.sig.is_super:
            assert all(isinstance(v, ValueRef) for v in inputs.values())
            return inputs
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

    def bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        """
        Given args and kwargs passed by the user from python, this adds defaults
        and returns a dict where they are indexed via internal names.
        """
        bound_args = self.op.py_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        inputs_dict = dict(bound_args.arguments)
        # rename to internal names
        inputs_dict = {k: v for k, v in inputs_dict.items()}
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
        self, inputs: Dict[str, Union[Any, ValueRef]]
    ) -> Tuple[List[ValueRef], Call]:
        """
        Run the function and return the outputs and the call object.
        """
        # wrap inputs
        wrapped_inputs = self.wrap_inputs(inputs)
        # get call UID
        call_uid = Hashing.get_content_hash(
            obj=[
                {k: v.uid for k, v in wrapped_inputs.items()},
                self.op.sig.name,
            ]
        )
        # check if call UID exists in call storage
        if self.storage.call_exists(call_uid):
            # get call from call storage
            call = self.storage.call_get(call_uid)
            # get outputs from obj storage
            self.storage.preload_objs([v.uid for v in call.outputs])
            wrapped_outputs = [self.storage.obj_get(v.uid) for v in call.outputs]
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
            self.storage.call_set(call_uid, call)
            # set inputs and outputs in obj storage
            for v in itertools.chain(wrapped_outputs, wrapped_inputs.values()):
                self.storage.obj_set(v.uid, v)
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

    def __call__(self, *args, **kwargs) -> Union[List[ValueRef], List[ValQuery]]:
        inputs = self.bind_inputs(args, kwargs)
        for x in inputs.values():
            if isinstance(x, ValQuery):
                # Operate in query mode.
                return self.format_as_outputs(outputs=self.call_query(inputs))

        # Operate in run mode.
        outputs, call = self.call_run(inputs)
        return self.format_as_outputs(outputs=outputs)


class FuncDecorator:
    """
    This is the `@op` decorator internally
    """

    def __init__(self, storage: Storage, is_super: bool = False):
        self.storage = storage
        self.super = is_super

    def __call__(self, func) -> "FuncInterface":
        op = FuncOp(func=func, is_super=self.super)
        op.sig = self.storage.synchronize(sig=op.sig)
        return FuncInterface(op=op, storage=self.storage)


def Q() -> ValQuery:
    """
    Create a `ValQuery` instance.

    Later on, we can add parameters to this to enforce query constraints in a
    more natural way.
    """
    return ValQuery(creator=None, created_as=None)


op = FuncDecorator
