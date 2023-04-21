from ..common_imports import *
from ..core.model import Call, Ref
from ..core.config import Config
from ..core.builtins_ import StructOrientations
from .storage import FuncOp, Connection
from .utils import bind_inputs, format_as_outputs, wrap_atom
from .contexts import Context

from ..queries.weaver import call_query

from .utils import MODES, debug_call, get_terminal_data, check_determinism

from ..common_imports import *
from ..core.config import Config
from ..core.model import Ref, Call, FuncOp
from ..core.builtins_ import Builtins
from ..core.wrapping import (
    wrap_inputs,
    wrap_outputs,
    causify_outputs,
    decausify,
    unwrap,
    contains_transient,
    contains_not_in_memory,
)

from ..storages.rel_impls.utils import Transactable, transaction, Connection

from ..deps.tracers import TracerABC
from ..deps.versioner import Versioner

from ..queries.weaver import StructOrientations


class Runner(Transactable):  # this is terrible
    def __init__(self, context: Optional[Context], func_op: FuncOp):
        self.context: Context = context
        self.storage = context.storage if context is not None else None
        self.func_op = func_op
        self.mode = context.mode if context is not None else MODES.noop
        self.code_state = context._code_state if context is not None else None
        #! todo - these should be set by the context
        self.recurse = False
        self.collect_calls = False

        ### set by preprocess
        self.linking_on: bool = None
        self.must_execute: bool = None
        self.suspended_trace_obj: Optional[Any] = None
        self.tracer_option: Optional[TracerABC] = None
        self.versioner: Optional[Versioner] = None
        self.wrapped_inputs: Dict[str, Ref] = None
        self.input_calls: List[Call] = None
        self.pre_call_uid: str = None
        self.call_option: Optional[Call] = None

        ### set by prep_execute
        self.must_save: bool = None
        self.is_recompute: bool = None
        self.func_inputs: Dict[str, Any] = None

    def process_other_modes(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Union[None, Any, Tuple[Any]]:
        if self.mode == MODES.query:
            inputs = bind_inputs(args, kwargs, mode=self.mode, func_op=self.func_op)
            return self.process_query(inputs=inputs)
        elif self.mode == MODES.batch:
            inputs = bind_inputs(args, kwargs, mode=self.mode, func_op=self.func_op)
            return self.process_batch(inputs=inputs)
        elif self.mode == MODES.noop:
            return self.func_op.func(*args, **kwargs)
        else:
            raise NotImplementedError

    def process_query(self, inputs: Dict[str, Any]) -> Union[None, Any, Tuple[Any]]:
        return format_as_outputs(
            outputs=call_query(func_op=self.func_op, inputs=inputs)
        )

    def process_batch(self, inputs: Dict[str, Any]) -> Union[None, Any, Tuple[Any]]:
        wrapped_inputs = {k: wrap_atom(v) for k, v in inputs.items()}
        outputs, call_struct = self.storage.call_batch(
            func_op=self.func_op, inputs=wrapped_inputs
        )
        self.context._call_structs.append(call_struct)
        return format_as_outputs(outputs=outputs)

    def preprocess(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        conn: Optional[Connection] = None,
    ):
        self.context._call_depth += 1
        self.linking_on = self.context._call_depth == 1
        func_op = self.func_op
        inputs = bind_inputs(args, kwargs, mode=self.mode, func_op=func_op)

        if self.storage.versioned:
            self.versioner = self.context._cached_versioner
            self.suspended_trace_obj = self.versioner.TracerCls.get_active_trace_obj()
            self.versioner.TracerCls.set_active_trace_obj(trace_obj=None)
        else:
            self.versioner = None
            self.suspended_trace_obj = None

        wrapped_inputs, input_calls = wrap_inputs(
            objs=inputs,
            annotations=func_op.input_annotations,
        )
        if self.linking_on:
            for input_call in input_calls:
                input_call.link(
                    orientation=StructOrientations.construct,
                )
        pre_call_uid = func_op.get_pre_call_uid(
            input_uids={k: v.uid for k, v in wrapped_inputs.items()}
        )
        call_option = self.storage.lookup_call(
            func_op=func_op,
            pre_call_uid=pre_call_uid,
            input_uids={k: v.uid for k, v in wrapped_inputs.items()},
            input_causal_uids={k: v.causal_uid for k, v in wrapped_inputs.items()},
            conn=conn,
            code_state=self.code_state,
            versioner=self.versioner,
        )
        self.tracer_option = (
            self.versioner.make_tracer() if self.storage.versioned else None
        )

        # condition determining whether we will actually call the underlying function
        self.must_execute = (
            call_option is None
            or (self.recurse and func_op.is_super)
            or (
                call_option is not None
                and call_option.transient
                and self.context.recompute_transient
            )
        )
        self.wrapped_inputs = wrapped_inputs
        self.input_calls = input_calls
        self.pre_call_uid = pre_call_uid
        self.call_option = call_option

    @transaction()
    def pre_execute(
        self,
        conn: Optional[Connection],
    ):
        call_option = self.call_option
        wrapped_inputs = self.wrapped_inputs
        func_op = self.func_op
        self.is_recompute = (
            call_option is not None
            and call_option.transient
            and self.context.recompute_transient
        )
        needs_input_values = (
            call_option is None or call_option is not None and call_option.transient
        )
        pass_inputs_unwrapped = Config.autounwrap_inputs and not func_op.is_super
        self.must_save = call_option is None
        if not (self.recurse and func_op.is_super) and not self.context.allow_calls:
            raise ValueError(
                f"Call to {func_op.sig.ui_name} not found in call storage."
            )
        if needs_input_values:
            self.storage.cache.mattach(vrefs=list(wrapped_inputs.values()))
            if any(contains_not_in_memory(ref=ref) for ref in wrapped_inputs.values()):
                msg = (
                    "Cannot execute function whose inputs are transient values "
                    "that are not in memory. "
                    "Use `recompute_transient=True` to force recomputation of these inputs."
                )
                raise ValueError(msg)
        if pass_inputs_unwrapped:
            self.func_inputs = unwrap(obj=wrapped_inputs, through_collections=True)
        else:
            self.func_inputs = wrapped_inputs

    def post_execute(self, outputs: List[Any]):
        call_option = self.call_option
        wrapped_inputs = self.wrapped_inputs
        pre_call_uid = self.pre_call_uid
        input_calls = self.input_calls
        func_op = self.func_op

        if self.tracer_option is not None:
            # check the trace against the code state hypothesis
            self.versioner.apply_state_hypothesis(
                hypothesis=self.code_state, trace_result=self.tracer_option.graph.nodes
            )
            # update the global topology and code state
            self.versioner.update_global_topology(graph=self.tracer_option.graph)
            self.code_state.add_globals_from(graph=self.tracer_option.graph)

        content_version, semantic_version = (
            self.versioner.get_version_ids(
                pre_call_uid=pre_call_uid,
                tracer_option=self.tracer_option,
                is_recompute=self.is_recompute,
            )
            if self.storage.versioned
            else (None, None)
        )
        call_uid = func_op.get_call_uid(
            pre_call_uid=pre_call_uid, semantic_version=semantic_version
        )
        wrapped_outputs, output_calls = wrap_outputs(
            objs=outputs,
            annotations=func_op.output_annotations,
        )
        transient = any(contains_transient(ref) for ref in wrapped_outputs)
        if self.is_recompute:
            check_determinism(
                observed_semver=semantic_version,
                stored_semver=call_option.semantic_version,
                observed_output_uids=[v.uid for v in wrapped_outputs],
                stored_output_uids=[w.uid for w in call_option.outputs],
                func_op=func_op,
            )
        if self.linking_on:
            wrapped_outputs = [x.unlinked(keep_causal=True) for x in wrapped_outputs]
        call = Call(
            uid=call_uid,
            func_op=func_op,
            inputs=wrapped_inputs,
            outputs=wrapped_outputs,
            content_version=content_version,
            semantic_version=semantic_version,
            transient=transient,
        )
        for outp in wrapped_outputs:
            decausify(ref=outp)  # for now, a "clean slate" approach
        causify_outputs(refs=wrapped_outputs, call_causal_uid=call.causal_uid)
        if self.linking_on:
            call.link(orientation=None)
            output_calls = [Builtins.collect_all_calls(x) for x in wrapped_outputs]
            output_calls = [x for y in output_calls for x in y]
            for output_call in output_calls:
                output_call.link(
                    orientation=StructOrientations.destruct,
                )
        if self.must_save:
            for constituent_call in itertools.chain([call], input_calls, output_calls):
                self.storage.cache.cache_call_and_objs(call=constituent_call)
        return call

    @transaction()
    def load_call(self, conn: Optional[Connection] = None):
        call_option = self.call_option
        wrapped_inputs = self.wrapped_inputs
        input_calls = self.input_calls
        func_op = self.func_op

        assert call_option is not None

        if not self.context.lazy:
            self.storage.cache.preload_objs(
                [v.uid for v in call_option.outputs], conn=conn
            )
            wrapped_outputs = [
                self.storage.cache.obj_get(v.uid) for v in call_option.outputs
            ]
        else:
            wrapped_outputs = [v for v in call_option.outputs]
        # recreate call
        call = Call(
            uid=call_option.uid,
            func_op=func_op,
            inputs=wrapped_inputs,
            outputs=wrapped_outputs,
            content_version=call_option.content_version,
            semantic_version=call_option.semantic_version,
            transient=call_option.transient,
        )
        if self.linking_on:
            call.outputs = [x.unlinked(keep_causal=False) for x in call.outputs]
            causify_outputs(refs=call.outputs, call_causal_uid=call.causal_uid)
            if not self.context.lazy:
                output_calls = [Builtins.collect_all_calls(x) for x in call.outputs]
                output_calls = [x for y in output_calls for x in y]

            else:
                output_calls = []
            call.link(orientation=None)
            for output_call in output_calls:
                output_call.link(
                    orientation=StructOrientations.destruct,
                )
        else:
            causify_outputs(refs=call.outputs, call_causal_uid=call.causal_uid)
        if call.causal_uid != call_option.causal_uid:
            # this is a new call causally; must save it and its constituents
            output_calls = [Builtins.collect_all_calls(x) for x in call.outputs]
            output_calls = [x for y in output_calls for x in y]
            for constituent_call in itertools.chain(input_calls, [call], output_calls):
                self.storage.cache.cache_call_and_objs(call=constituent_call)
        return call

    def postprocess(
        self, call: Call, output_format: str = "returns"
    ) -> Union[None, Any, Tuple[Any]]:
        func_op = self.func_op
        if self.storage.versioned and self.suspended_trace_obj is not None:
            self.versioner.TracerCls.set_active_trace_obj(
                trace_obj=self.suspended_trace_obj
            )
            terminal_data = get_terminal_data(func_op=func_op, call=call)
            # Tracer.leaf_signal(data=terminal_data)
            self.versioner.TracerCls.register_leaf_event(
                trace_obj=self.suspended_trace_obj, data=terminal_data
            )
            # self.tracer_impl.suspended_tracer.register_leaf(data=terminal_data)
        if self.context.debug_calls:
            debug_call(
                func_name=func_op.sig.ui_name,
                memoized=self.call_option is not None,
                wrapped_inputs=self.wrapped_inputs,
                wrapped_outputs=call.outputs,
            )
        if self.collect_calls:
            self.context._call_buffer.append(call)
        sig = func_op.sig
        self.context._call_uids[(sig.internal_name, sig.version)].append(call.uid)
        self.context._call_depth -= 1
        if self.context._attach_call_to_outputs:
            for output in call.outputs:
                output._call = call.detached()
        if output_format == "list":
            return call.outputs
        elif output_format == "returns":
            return format_as_outputs(outputs=call.outputs)

    def _get_connection(self) -> Connection:
        return self.storage.rel_storage._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.storage.rel_storage._end_transaction(conn=conn)
