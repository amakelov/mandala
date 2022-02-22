from .context import Context
from .storage import Storage

from ..common_imports import *
from ..util.common_ut import group_like, ungroup_like, rename_dict_keys, get_collection_atoms, concat_homogeneous_lists
from ..util.common_ut import transpose_lists, transpose_listdict, transpose_returns
from ..core.config import CoreConfig, MODES, SuperopWrapping
from ..core.utils import BackwardCompatible, AnnotatedObj
from ..core.bases import (
    Operation, Call, ValueRef, is_member_of_tp, is_instance, is_deeply_persistable,
    get_vrefs_from_calls, overwrite_vref, set_uid_pure, unwrap, GlobalContext,
    contains_any_vref
)
from ..core.tps import Type
from ..core.sig import BaseSignature
from ..core.impl import BaseCall, SimpleFunc, AtomRef
from ..core.wrap import (
    wrap_constructive, wrap_structure, get_deconstruction_calls, wrap_detached
)
from ..storages.calls import CallLocation
from ..adapters.vals import BaseValAdapter
from ..adapters.calls import BaseCallAdapter
from ..queries.rel_weaver import FuncQuery, ValQuery

CONTEXT_KW = '__context__'

default_type_dict = {}

def wrap_inputs(args:tuple, kwargs:dict, signature:BaseSignature,
                typecheck:bool=True, strict_typecheck:bool=False,
                ) -> TTuple[TDict[str, ValueRef], TList[Call], TDict[str, TAny]]:
    """
    Bind args and kwargs to a dictionary, add defaults and deal with
    backward-compatible arguments.

    Returns:
        - a dictionary of wrapped inputs that ignores backward-compatible inputs
        that equal their default value
        - list of constructive calls induced by wrapping these inputs
        - dictionary of any bound backward-compatible inputs (where values are
        just their default values).
    """
    inputs = signature.bind_args(args=args, kwargs=kwargs,
                                 apply_defaults=CoreConfig.enable_defaults)
    # ignore BackwardCompatible instances, or default values of
    # backward-compatible arguments
    filtered_inputs = {}
    compat_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, BackwardCompatible):
            compat_inputs[k] = v.default
            continue
        elif k in signature.defaults:
            default = signature.defaults[k]
            if isinstance(default, BackwardCompatible):
                if default.default == unwrap(v):
                    compat_inputs[k] = v
                    continue
        filtered_inputs[k] = v
    input_types = signature.bind_types(args=args, kwargs=kwargs)
    wrapped = {}
    calls = []
    for k, v in filtered_inputs.items():
        wrapped_input, input_calls = wrap_constructive(
            obj=v, reference=input_types[k],
            type_dict=default_type_dict
        )
        if typecheck:
            checker = is_member_of_tp if strict_typecheck else is_instance
            assert checker(vref=wrapped_input, tp=input_types[k]),\
                   f'Got input of {wrapped_input.get_type()}, expected {input_types[k]}'
            if wrapped_input.get_type() != input_types[k]:
                logging.debug(f'Encountered unequal types for input {k}')
        wrapped[k] = wrapped_input
        calls += input_calls
    return wrapped, calls, compat_inputs

def wrap_outputs(outputs:TDict[str, TAny], uids:TDict[str, str], 
                 output_types:TDict[str, TOption[Type]], typecheck:bool=True, 
                 strict_typecheck:bool=False,
                 ) -> TTuple[TDict[str, ValueRef], TList[Call]]:
    """
    Wrap outputs of a non-superop deconstructively
    """
    wrapped = {}
    calls = []
    for k, v in outputs.items():
        uid = uids[k]
        reference = output_types[k]
        wrapped_output = wrap_structure(obj=v, reference=reference, 
                                        type_dict=default_type_dict)
        wrapped_output._set_uid(uid=uid)
        if typecheck:
            if reference is not None:
                checker = is_member_of_tp if strict_typecheck else is_instance
                assert checker(vref=wrapped_output, tp=reference),\
                       f'Got output of {wrapped_output.get_type()}, expected {output_types[k]}'
                if wrapped_output.get_type() != output_types[k]:
                    logging.debug(f'Encountered unequal types for output {k}')
        output_calls = get_deconstruction_calls(obj=wrapped_output,
                                                write_uids=True)
        wrapped[k] = wrapped_output
        calls += output_calls
    return wrapped, calls

def wrap(obj:TAny, c:Context=None, reference:Type=None,
         type_dict:TDict[type, Type]=None, annotation:TAny=None) -> ValueRef:
    """
    Wrap an object (constructively) and save all resulting calls.
    """
    if c is None:
        c = GlobalContext.get()
        assert c is not None
    type_dict = {} if type_dict is None else type_dict
    mode = c.mode
    if mode == MODES.noop:
        return obj
    vref, calls = wrap_detached(obj=obj, reference=reference, annotation=annotation,
                                type_dict=type_dict, through_collections=False,
                                return_calls=True)
    if mode == MODES.delete:
        assert c.storage is not None
        c.deletion_buffer.minsert(calls=calls)
    elif mode == MODES.capture:
        assert c.captured_calls is not None
        c.captured_calls.minsert(calls=calls)
    elif mode == MODES.transient:
        pass
    elif mode == MODES.run:
        # must save calls and vrefs
        storage = c.storage
        assert storage is not None
        partition = c.partition
        assert partition is not None
        # don't forget that there may be no calls, just a single value
        storage.val_adapter.set_shallow(vref=vref)
        store_vrefs_from_calls(calls=calls, val_adapter=storage.val_adapter)
        store_calls(calls=calls, call_adapter=storage.call_adapter,
                    on_exists='skip', 
                    partition=partition)
    else:
        raise ValueError()
    return vref
    
################################################################################
### most complex logic in the entire thing 
################################################################################
def store_vrefs_from_calls(calls:TList[Call], val_adapter:BaseValAdapter,
                           skip_delayed:bool=False):
    unique_vrefs = get_vrefs_from_calls(calls=calls, drop_duplicates=True)
    val_adapter.mset(vrefs=unique_vrefs, shallow=True,
                     skip_delayed=skip_delayed)

def store_calls(calls:TList[Call], call_adapter:BaseCallAdapter, 
                on_exists:str, partition:str):
    detached_calls = [call.detached() for call in calls]
    call_locs = [call_adapter.get_location(uid=call.uid, 
                                           metadata={'partition': partition})
                 for call in detached_calls]
    if on_exists == 'skip':
        set_func = call_adapter.call_storage.mset_if_not_exists
    elif on_exists == 'raise':
        set_func = call_adapter.call_storage.mcreate
    else:
        raise ValueError()
    set_func(locs=call_locs, call_datas=detached_calls)

class OpCaller(object):
    RETURNS_KW = '__returns__'
    CONTEXT_KW = '__context__'
    OPTIMIZE_ATTACHMENT = False
    def __init__(self, op:SimpleFunc, c:Context=None, typecheck:bool=True):
        self.c = Context() if c is None else c
        self.op = op
        self.typecheck = typecheck
        
    ### helpers for different stages of execution
    def process_returns_kwarg(self, kwargs:TDict[str, TAny]):
        if self.RETURNS_KW in kwargs:
            returns = kwargs.pop(self.RETURNS_KW)
            apply_returns = True
        else:
            returns = None
            apply_returns = False
        return apply_returns, returns
    
    def get_context_from_kwargs(self, kwargs:TDict[str, TAny]) -> Context:
        if self.CONTEXT_KW in kwargs:
            c = kwargs[self.CONTEXT_KW]
        else:
            c = self.c
        return c
    
    def process_context_kwarg(self,
                              kwargs:TDict[str, TAny]) -> TTuple[bool, Context]:
        #! TODO:  modifies kwargs inplace :( - 
        if self.CONTEXT_KW in kwargs:
            pass_context_to_func = True
            context = kwargs.pop(self.CONTEXT_KW)
        else:
            pass_context_to_func = False
            context = self.c
        return pass_context_to_func, context
    
    def preprocess_inputs(self, args, kwargs, apply_defaults:bool,
                          ) -> TTuple[TDict[str, TAny], bool, TAny, Context, bool]:
        """
        Put inputs in a dict named by internal argnames, and parse out optional 
        return override and optional context override
        ! changes kwargs in-place
        """
        apply_returns, returns = self.process_returns_kwarg(kwargs=kwargs)
        pass_context_to_func, context_data = self.process_context_kwarg(kwargs=kwargs)
        ui_inputs_dict = self.op.orig_sig.bind_args(args=args, kwargs=kwargs, 
                                                    apply_defaults=apply_defaults)
        name_mapping = self.op.sig_map.map_input_names(input_names=ui_inputs_dict.keys())
        inputs_dict = rename_dict_keys(dct=ui_inputs_dict, mapping=name_mapping)
        return (inputs_dict, apply_returns, returns,
                context_data, pass_context_to_func)

    def get_call_uid(self, wrapped_inputs:TDict[str, ValueRef], c:Context):
        call_metadata = {
            '__version__': self.op.version,
            '__tag__': c.tag
        }
        input_uids = {k: v.uid for k, v in wrapped_inputs.items()}
        call_uid = self.op.get_call_uid(input_uids=input_uids,
                                        metadata=call_metadata)
        return call_uid
    
    def mget_recoverable(self, call_uids:TList[str], c:Context,
                         ) -> TTuple[TList[bool], TList[TOption[Call]]]:
        """
        Figure out if this call needs to be recomputed, or we can get away with
        either loading the values or returning lazy references. 
        
        NOTE: some settings can change this behavior:
            - if c.lazy and not c.force, existing calls are marked as
              `must_compute`
            - if c.force, a call that exists and is marked for recomputation 
            is marked as `must_compute`, even if c.lazy

        Returns:
            - mrecoverable: indicator for which calls' outputs have been
            persisted
            - call_datas: loaded data for existing call, None for non-existing
        """
        logging.debug('Entering mget_must_compute...')
        call_st, call_adapter = c.storage.call_st, c.storage.call_adapter
        call_locs = [call_adapter.get_location(
            uid=uid,
            metadata={'partition': c.partition}
        ) for uid in call_uids]
        #! optimization
        mexists_mask = call_st.mexists(locs=call_locs, allow_fallback=True)
        logging.debug('Got mexists mask')
        loc_groups = group_like(objs=call_locs, labels=mexists_mask)
        #! optimization
        existing_call_datas = call_st.mget(locs=loc_groups[True], allow_fallback=True)
        logging.debug('Got existing call datas')
        mmust_compute_existing = []
        for call_data in existing_call_datas:
            needs_recomputation_recorded = call_data.metadata.get('needs_recomputation', False)
            if c.lazy:
                if c.force and needs_recomputation_recorded:
                    must_compute = True
                else:
                    must_compute = False
            else:
                must_compute = needs_recomputation_recorded
            mmust_compute_existing.append(must_compute)
        mmust_compute_new = [True for _ in loc_groups[False]]
        new_call_datas = [None for _ in loc_groups[False]]
        mmust_compute = ungroup_like(
            groups={True: mmust_compute_existing, False: mmust_compute_new},
            labels=mexists_mask
        )
        call_datas = ungroup_like(
            groups={True: existing_call_datas, False: new_call_datas},
            labels=mexists_mask
        )
        mrecoverable = [not elt for elt in mmust_compute]
        return mrecoverable, call_datas

    def get_recoverable(self,
                        call_uid, c:Context) -> TTuple[bool, TOption[Call]]:
        """
        Single-call version of mget_recoverable
        """
        mrecoverable, call_datas = self.mget_recoverable(call_uids=[call_uid],
                                                         c=c)
        return mrecoverable[0], call_datas[0]
    
    def mattach_inputs(self, mwrapped_inputs:TList[TDict[str, ValueRef]],
                       c:Context):
        flat = get_collection_atoms(collection=mwrapped_inputs)
        c.storage.val_adapter.mattach(vrefs=flat, shallow=False)
    
    def attach_inputs(self, wrapped_inputs:TDict[str, ValueRef], c:Context):
        self.mattach_inputs(mwrapped_inputs=[wrapped_inputs], c=c)
    
    def wrap_inputs(self, args, kwargs, include_deconstructive:bool=False,
                    ) -> TTuple[TDict[str, ValueRef], TList[Call], TDict[str, TAny]]:
        """
        Wraps inputs for self.op and produces corresponding calls. It:
            - builds up calls to wrap the values if necessary;
            - *optionally* gives deconstructive calls for these values. This can
             be used to propagate relations inside a superop's body
        """
        wrapped_inputs, input_calls, compat_inputs = wrap_inputs(args=args,
                                    kwargs=kwargs, signature=self.op.sig, 
                                    typecheck=self.typecheck)
        if include_deconstructive:
            for _, v in wrapped_inputs.items():
                input_calls += get_deconstruction_calls(obj=v, write_uids=False)
        return wrapped_inputs, input_calls, compat_inputs

    def wrap_superop_outputs(self, outputs:TDict[str, TAny], 
                             include_deconstructive:bool,
                             ) -> TTuple[TDict[str, ValueRef], TList[Call], TList[Call]]:
        """
        Given outputs of a superop (which may be wrapped or not), this builds up
        calls to wrap these values if necessary
        
        also (optionally) gives deconstructive calls for these values to be
        consistent with how ordinary ops are treated.
        """
        if self.op.sig.has_fixed_outputs:
            output_types = self.op.sig.outputs
            assert output_types is not None
        else:
            raise NotImplementedError()
        wrapped_outputs = {}
        constructive_calls = []
        for k, v in outputs.items():
            vref, calls = wrap_constructive(obj=v, reference=output_types[k])
            wrapped_outputs[k] = vref
            constructive_calls += calls
        deconstructive_calls = []
        if include_deconstructive:
            for k, v in wrapped_outputs.items():
                deconstructive_calls += get_deconstruction_calls(
                    obj=v,
                    write_uids=False
                )
        return wrapped_outputs, constructive_calls, deconstructive_calls
        
    def wrap_outputs(self, outputs:TDict[str, TAny],
                     call_uid:str,
                     ) -> TTuple[TDict[str, ValueRef], TList[Call]]:
        if self.op.sig.has_fixed_outputs:
            output_types = self.op.sig.outputs
            assert output_types is not None
        else:
            # outputs must be annotated
            output_types = {}
            for k, v in outputs.items():
                assert isinstance(v, AnnotatedObj)
                assert v.target_type is not None
                output_types[k] = v.target_type
        # compute content uids if needed
        output_uids = Operation.get_output_uids(call_uid=call_uid, 
                                                raw_outputs=outputs, 
                                                output_types=output_types)
        return wrap_outputs(outputs=outputs, uids=output_uids, 
                            output_types=output_types,
                            typecheck=self.typecheck)

    def call_query(self, inputs_dict:TDict[str, ValQuery], c:Context):
        if c.mode == MODES.query_delete:
            tags = {'delete': True}
        else:
            tags = {}
        func_query = FuncQuery(op=self.op, tags=tags)
        results = func_query(**inputs_dict)
        return self.postprocess_outputs_as_returns(outputs_dict=results)
    
    def _returns_to_dict(self, sig:BaseSignature,
                         returns:TAny) -> TDict[str, TAny]:
        """
        Format the return values as a dictionary of {output_name: value} with
        respect to the given signature.
        """
        if sig.has_fixed_outputs:
            assert sig.output_names is not None
            output_names = sig.output_names
            if len(output_names) == 0:
                assert returns is None
                func_outputs = {}
            elif len(output_names) == 1:
                func_outputs = {output_names[0]: returns}
            else:
                assert isinstance(returns, tuple)
                func_outputs = {name: returns[i] 
                                for i, name in enumerate(output_names)}
        else:
            if isinstance(returns, tuple):
                func_outputs = {f'output_{i}': returns[i]
                                for i in range(len(returns))}
            elif returns is None:
                logging.warning('Interpreting __returns__=None as 0 outputs')
                func_outputs = {}
            else:
                func_outputs = {'output_0': returns}
        return func_outputs

    def compute(self, call_uid:str, wrapped_inputs:TDict[str, ValueRef],
                returns:TUnion[tuple, TAny], apply_returns:bool,
                pass_context_to_func:bool, compat_inputs:TDict[str, TAny],
                c:Context, is_retracing_superop:bool):
        """
        Responsible for computing the function (or simulating its computation
        via returns). 
        
        Must handle the following corner cases:
            - provide unwrapped inputs for normal ops and keep them wrapped for 
            superops or low-level ops;
            - handle wrapping of inputs/outputs of superops separately 
            - figure out if the call will need to be recomputed in the future
        """
        input_uids = {k: v.uid for k, v in wrapped_inputs.items()}
        allow_calls = c.allow_calls
        call_metadata = {
            '__version__': self.op.version,
            '__tag__': c.tag
        }
        if apply_returns:
            func_outputs = self._returns_to_dict(sig=self.op.sig,
                                                 returns=returns)
            # ensure that values are fully unwrapped 
            if CoreConfig.require_returns_unwrapped:
               assert all(not contains_any_vref(obj) 
                          for obj in func_outputs.values())
        else:
            if not is_retracing_superop:
                assert allow_calls
            if (not self.op.is_super) and (self.op.unwrap_inputs):
                func_inputs = {k: v.unwrap() 
                               for k, v in wrapped_inputs.items()}
            else:
                func_inputs = wrapped_inputs
            if pass_context_to_func:
                func_outputs = self.op.compute(inputs={**func_inputs,
                                                       **compat_inputs}, 
                                        context_arg=CONTEXT_KW,
                                        context_representation=c)
            else:
                func_outputs = self.op.compute(inputs={**func_inputs, **compat_inputs})
            
        if not self.op.is_super:
            wrapped_outputs, output_calls = self.wrap_outputs(
                outputs=func_outputs,
                call_uid=call_uid
            )
        else:
            # this block produces (wrapped_outputs, output_calls)
            if CoreConfig.superop_wrapping_style == SuperopWrapping.construct_and_deconstruct:
                (wrapped_outputs, constructive_calls,
                 deconstructive_calls) = self.wrap_superop_outputs(outputs=func_outputs, include_deconstructive=True)
                output_calls = constructive_calls + deconstructive_calls
            elif CoreConfig.superop_wrapping_style == SuperopWrapping.legacy:
                if not all([isinstance(output, ValueRef) for output in func_outputs.values()]):
                    raise RuntimeError(f'Superop {self.op.ui_name} did not return vrefs')
                # all calls are accounted for by internal ops, supposedly
                wrapped_outputs, output_calls = func_outputs, []
            elif CoreConfig.superop_wrapping_style == SuperopWrapping.construct_only:
                assert not CoreConfig.decompose_struct_as_many
                (wrapped_outputs, constructive_calls,
                 deconstructive_calls) = self.wrap_superop_outputs(outputs=func_outputs, include_deconstructive=False)
                output_calls = constructive_calls
            else:
                raise ValueError()
        if not all(is_deeply_persistable(vref=output) for output in wrapped_outputs.values()):
            logging.debug('New call; NOT all outputs are deeply persistable')
            needs_recomputation = True
        else:
            logging.debug('New call; all outputs are deeply persistable')
            needs_recomputation = False
        call_metadata['needs_recomputation'] = needs_recomputation
        if self.op.is_super:
            # in case there are mutations to the inputs
            original_inputs = {k: set_uid_pure(v, new_uid=input_uids[k])
                               for k, v in wrapped_inputs.items()}
        else:
            original_inputs = wrapped_inputs
        op_call = BaseCall.from_execution_data(
            op=self.op,
            inputs=original_inputs,
            outputs=wrapped_outputs,
            metadata=call_metadata, uid=call_uid
        )
        return wrapped_outputs, output_calls, op_call
    
    def mload_outputs(self, mcall_data:TList[Call],
                      c:Context, deeplazy:bool=False,
                      ) -> TList[TDict[str, ValueRef]]:
        assert c.storage is not None
        moutput_locations = [
            c.storage.call_adapter.get_output_locs(call=call_data)
            for call_data in mcall_data
        ]
        mwrapped_outputs = c.storage.val_adapter.mget_collection(
            collection=moutput_locations,
            lazy=c.lazy,
            deeplazy=deeplazy
        )
        return mwrapped_outputs

    def load_outputs(self, call_data,
                     c:Context, deeplazy:bool=False) -> TDict[str, ValueRef]:
        return self.mload_outputs([call_data], c=c, deeplazy=deeplazy)[0]
    
    def msave_resulting_calls(self, mop_calls:TList[Call], 
                              moutput_calls:TList[TList[Call]],
                              minput_calls:TList[TList[Call]], 
                              c:Context, skip_delayed:bool=False):
        """
        Saves the op calls themselves last to make it more difficult for broken
        state to exist.
        """
        buff_val_adapter = (c.buffer.val_adapter if c.buffer is not None
                            else c.storage.val_adapter)
        all_output_calls = concat_homogeneous_lists(lists=moutput_calls)
        all_input_calls = concat_homogeneous_lists(minput_calls)
        # store vrefs BEFORE calls
        store_vrefs_from_calls(calls=all_input_calls +
                               all_output_calls +
                               mop_calls, 
                               val_adapter=buff_val_adapter, 
                               skip_delayed=skip_delayed)
        ### store calls
        # input calls: may already exist from decompositions of the same struct
        buff_call_adapter = (c.buffer.call_adapter if c.buffer is not None
                             else c.storage.call_adapter)
        store_calls(calls=all_input_calls,
                    call_adapter=buff_call_adapter,
                    on_exists='skip', partition=c.partition)
        # op call and output calls may already exist, if we re-computed
        # transient values
        store_calls(calls=all_output_calls,
                    call_adapter=buff_call_adapter,
                    on_exists='skip', partition=c.partition)
        #! store op calls LAST to avoid broken state
        store_calls(calls=mop_calls,
                    call_adapter=buff_call_adapter,
                    on_exists='skip', partition=c.partition)
    
    def save_resulting_calls(self, c:Context, op_call:Call=None, 
                             output_calls:TList[Call]=None, 
                             input_calls:TList[Call]=None,
                             skip_delayed:bool=False):
        mop_calls = [] if op_call is None else [op_call]
        moutput_calls = [] if output_calls is None else [output_calls]
        minput_calls = [] if input_calls is None else [input_calls]
        self.msave_resulting_calls(mop_calls=mop_calls, moutput_calls=moutput_calls, 
                                   minput_calls=minput_calls, c=c,
                                   skip_delayed=skip_delayed)

    def postprocess_outputs_as_returns(self, outputs_dict:TDict[str, TAny]):
        if self.op.sig.is_fixed:
            assert self.op.output_names is not None
            if len(self.op.output_names) == 1:
                assert len(outputs_dict) == 1
                key = list(outputs_dict.keys())[0]
                return outputs_dict[key]
            else:
                return tuple([outputs_dict[k] for k in self.op.output_names])
        else:
            output_names = outputs_dict.keys()
            expected_names = [f'output_{i}' for i in range(len(output_names))]
            assert set(output_names) == set(expected_names)
            n_out = len(output_names)
            if n_out == 0:
                return None
            elif n_out == 1:
                return outputs_dict[expected_names[0]]
            else:
                return tuple([outputs_dict[name] for name in expected_names])

    ############################################################################ 
    ### putting it together
    ############################################################################ 
    def mpreprocess_inputs(self, marg_cols, mkwarg_cols,
                           ) -> TTuple[TList[tuple], TList[dict], TList[bool], 
                                       TUnion[TList[TAny], TList[TTuple[TAny,...]]]]:
        if self.RETURNS_KW in mkwarg_cols:
            returns_given = True
            returns_cols = mkwarg_cols.pop(self.RETURNS_KW)
            mreturns = transpose_returns(returns_cols=returns_cols)
            mapply_returns = [True for _ in mreturns]
        else:
            returns_given = False
        margs = [tuple(elt) for elt in transpose_lists(lists=list(marg_cols))]
        mkwargs = transpose_listdict(dictlist=mkwarg_cols)
        if not margs:
            margs = [() for _ in mkwargs]
        if not mkwargs:
            mkwargs = [{} for _ in margs]
        if not returns_given:
            mreturns = [None for _ in margs]
            mapply_returns = [False for _ in margs]
        return margs, mkwargs, mapply_returns, mreturns

    def run_is_recoverable(self, args, kwargs) -> bool:
        (inputs_dict, apply_returns, returns, c,
         pass_context_to_func) = self.preprocess_inputs(args=args,
                                                        kwargs=kwargs,
                                                        apply_defaults=True)
        wrapped_inputs, _, _ = self.wrap_inputs(args=(), kwargs=inputs_dict,
                                                include_deconstructive=False)
        call_uid = self.get_call_uid(wrapped_inputs=wrapped_inputs, c=c)
        recoverable, _ = self.get_recoverable(call_uid=call_uid, c=c)
        return recoverable
    
    def mrun_is_recoverable(self, marg_cols:TTuple[TList[TAny]], 
             mkwarg_cols:TDict[str, TList[TAny]], c:Context) -> TList[bool]:
        margs, mkwargs, _, _ = self.mpreprocess_inputs(marg_cols=marg_cols,
                                                       mkwarg_cols=mkwarg_cols)
        call_uids = []
        for args, kwargs in zip(margs, mkwargs):
            (inputs_dict, _, _, c, _) = self.preprocess_inputs(args=args, 
                                                               kwargs=kwargs,
                                                               apply_defaults=True)
            wrapped_inputs, _, _ = self.wrap_inputs(args=(), kwargs=inputs_dict,
                                                    include_deconstructive=False)
            call_uid = self.get_call_uid(wrapped_inputs=wrapped_inputs, c=c)
            call_uids.append(call_uid)
        mrecoverable, _ = self.mget_recoverable(call_uids=call_uids, c=c)
        return mrecoverable
    
    def track_calls(self, call_uid:str, input_calls:TList[Call], 
                    deeplazy_outputs:TDict[str, ValueRef], 
                    output_calls:TList[Call], c:Context, 
                    destination:str):
        assert c.storage is not None
        loc_getter = lambda uid: c.storage.call_adapter.get_location(
            uid=uid,
            metadata={'partition': c.partition}
        )
        call_loc = loc_getter(call_uid)
        op_call = c.storage.call_st.get(loc=call_loc, allow_fallback=True)
        # figure out if we need to include decomposition calls
        if self.op.is_super:
            if (CoreConfig.superop_wrapping_style ==
                SuperopWrapping.construct_and_deconstruct):
                include_decomp = True
            else:
                include_decomp = False
        else:
            include_decomp = True
        if include_decomp:
            decomposition_calls = concat_homogeneous_lists(
                [get_deconstruction_calls(obj=output, write_uids=False)
                 for output in deeplazy_outputs.values()])
        else:
            decomposition_calls = []
        if destination == 'deletion_buffer':
            assert c.deletion_buffer is not None
            c.deletion_buffer.minsert(calls=[op_call] + 
                                      input_calls +
                                      decomposition_calls +
                                      output_calls)
        elif destination == 'captured_calls':
            assert c.captured_calls is not None
            c.captured_calls.minsert(calls=[op_call] +
                                     input_calls +
                                     decomposition_calls +
                                     output_calls)   
        else:
            raise ValueError()
    
    def track_deletions(self, call_uid:str, input_calls:TList[Call], 
                        deeplazy_outputs:TDict[str, ValueRef], 
                        output_calls:TList[Call], c:Context):
        """
        Track call locations to be deleted in deletion buffer. 
        
        It needs the following data:
            - any input wrapping calls that happened in this context;
            - the *deeply lazily loaded* outputs, so that we can take into
            account all calls for the outputs of a function;
            - the call uid for the function's own call
        """
        assert c.storage is not None
        loc_getter = lambda uid: c.storage.call_adapter.get_location(
            uid=uid,
            metadata={'partition': c.partition}
        )
        call_loc = loc_getter(call_uid)
        op_call = c.storage.call_st.get(loc=call_loc, allow_fallback=True)
        decomposition_calls = concat_homogeneous_lists(
            [get_deconstruction_calls(obj=output, write_uids=False)
             for output in deeplazy_outputs.values()])
        c._deletion_buffer.minsert(calls=[op_call] +
                                   input_calls +
                                   decomposition_calls +
                                   output_calls)
        
    def locate(self, args, kwargs) -> CallLocation:
        """
        Return the location of the call for these arguments
        """
        (inputs_dict, apply_returns, returns, c,
         pass_context_to_func) = self.preprocess_inputs(
             args=args,
            kwargs=kwargs,
            apply_defaults=True
        )
        wrapped_inputs, _, _ = self.wrap_inputs(args=(), kwargs=inputs_dict, 
                                                include_deconstructive=False)
        call_uid = self.get_call_uid(wrapped_inputs=wrapped_inputs, c=c)
        call_loc = c.storage.call_adapter.get_location(
            uid=call_uid,
            metadata={'partition': c.partition}
        )
        return call_loc
    
    ############################################################################ 
    ### mutations
    ############################################################################ 
    def get_internal_mutations(self) -> TDict[str, int]:
        """
        Get the dictionary of {internal input name: output index } mutation
        pairs. 
        """
        assert self.op.mutations is not None
        mapping = self.op.sig_map.fixed_inputs_map()
        return {mapping[k]: v for k, v in self.op.mutations.items()}

    def verify_mutations(self, wrapped_inputs:TDict[str, ValueRef], 
                        returns:TUnion[ValueRef, TTuple[ValueRef,...]], 
                        first_time:bool=False):
        """
        Check that mutations are valid

        Assumptions:
            - any mutation pair must be atoms of the same type
            - if both are in memory and this is the first time we are running
            this call, their underlying objects must have the same id
        """
        if self.op.mutations is None:
            return
        internal_mutations = self.get_internal_mutations()
        outputs_tuple = returns if isinstance(returns, tuple) else (returns,)
        for input_name, output_index in internal_mutations.items():
            mut_input = wrapped_inputs[input_name]
            mut_output = outputs_tuple[output_index]
            if not isinstance(mut_input, AtomRef):
                raise NotImplementedError()
            if not isinstance(mut_output, AtomRef):
                raise NotImplementedError()
            if not mut_input.get_type() == mut_output.get_type():
                raise RuntimeError()
            if mut_input.in_memory and mut_output.in_memory and first_time:
                if id(mut_input.obj()) != id(mut_output.obj()):
                    raise RuntimeError()
    
    def apply_mutations(self, wrapped_inputs:TDict[str, ValueRef], 
                        returns:TUnion[ValueRef, TTuple[ValueRef,...]]):
        """
        Overwrite the state of the op's mutated inputs with their matching
        outputs.
        """
        if self.op.mutations is None:
            return
        outputs_tuple = returns if isinstance(returns, tuple) else (returns,)
        internal_mutations = self.get_internal_mutations()
        for input_name, output_index in internal_mutations.items():
            mut_input = wrapped_inputs[input_name]
            mut_output = outputs_tuple[output_index]
            overwrite_vref(vref=mut_input, overwrite_from=mut_output)
    
    ############################################################################ 
    ### high-level run methods
    ############################################################################ 
    def _infer_include_deconstructive_in_inputs(self) -> bool:
        if (CoreConfig.superop_wrapping_style ==
            SuperopWrapping.construct_and_deconstruct):
            inputs_include_deconstructive = self.op.is_super
        elif (CoreConfig.superop_wrapping_style ==
              SuperopWrapping.legacy):
            inputs_include_deconstructive = False
        elif (CoreConfig.superop_wrapping_style ==
              SuperopWrapping.construct_only):
            inputs_include_deconstructive = False
        else:
            raise ValueError()
        return inputs_include_deconstructive
    
    def _cg_precall(self, c:Context, will_compute:bool):
        # update call graph and call stack before a call
        if not CoreConfig.build_call_graph:
            return
        if c.mode == MODES.run:
            storage:Storage = c.storage
            cg_storage = storage.call_graph_st
            op_id = self.op.qualified_name
            if len(c.call_stack) > 0:
                caller = c.call_stack[-1]
                cg_storage.add_edge(source=caller, target=op_id)
            if self.op.is_super and will_compute:
                c.push_call(op_id)
    
    def _cg_postcall(self, c:Context, did_compute:bool):
        # update call stack after a call
        if not CoreConfig.build_call_graph:
            return
        if c.mode == MODES.run:
            storage:Storage = c.storage
            if self.op.is_super and did_compute:
                op_id = self.op.qualified_name
                assert c.call_stack[-1] == op_id
                c.pop_call()
        
    def run(self, args, kwargs):
        ### preprocess inputs, remove special kwargs, and update context data,
        ### if given explicitly
        logging.debug('Entered run')
        ### figure out if we should apply defaults
        c = self.get_context_from_kwargs(kwargs=kwargs)
        mode = c.mode
        if mode == MODES.query:
            apply_defaults = CoreConfig.bind_defaults_in_queries
        else:
            apply_defaults = CoreConfig.enable_defaults
        ### 
        (inputs_dict, apply_returns, returns, c,
         pass_context_to_func) = self.preprocess_inputs(
             args=args,
            kwargs=kwargs,
            apply_defaults=apply_defaults
        )
        mode = c.mode
        logging.debug(f'Mode is {mode}')
        if mode in MODES.noop:
            return self.op.func(*args, **kwargs)
        if mode in (MODES.query, MODES.query_delete):
            return self.call_query(inputs_dict=inputs_dict, c=c)
        inputs_include_deconstructive =\
            self._infer_include_deconstructive_in_inputs()
        wrapped_inputs, input_calls, compat_inputs = self.wrap_inputs(
            args=(),
            kwargs=inputs_dict,
            include_deconstructive=inputs_include_deconstructive
        )
        logging.debug('Wrapped inputs')
        call_uid = self.get_call_uid(wrapped_inputs=wrapped_inputs, c=c)
        if mode in (MODES.run, MODES.delete, MODES.capture):
            recoverable, call_data = self.get_recoverable(call_uid=call_uid, c=c)
            logging.debug(f'Got call status: is_recoverable={recoverable}')
        elif mode == MODES.transient:
            recoverable, call_data = False, None
        else:
            raise NotImplementedError()
        # we proceed to the call if:
        # - outputs are not recoverable, or
        # - this is a superop, and we are in delete/capture mode 
        is_retracing_superop = (self.op.is_super and mode in 
                                (MODES.delete, MODES.capture))
        compute_condition = (not recoverable) or is_retracing_superop
        ### case: go inside op
        if compute_condition:
            if mode in (MODES.delete, MODES.capture) and not recoverable:
                raise NotImplementedError()
            if c.storage is not None:
                if self.op.is_super and self.OPTIMIZE_ATTACHMENT:
                    pass
                else:
                    # an optimization to prevent superops from loading large
                    # structs when this is not necessary
                    attach_condition = (not apply_returns) and (not self.op.is_super)
                    if attach_condition:
                        self.attach_inputs(wrapped_inputs=wrapped_inputs, c=c)
                        logging.debug(f'Inputs attached')
            if mode == MODES.run:
                #! these calls must be saved before function call in case of mutations
                if call_data is None:
                    # save calls only for calls that have not happened
                    # (otherwise it's a "transient" call that recorded its data before)
                    self.save_resulting_calls(c=c, input_calls=input_calls, 
                                            skip_delayed=True)
            self._cg_precall(c=c, will_compute=True)
            start_time = time.time() # an approximation really
            wrapped_outputs, output_calls, op_call = self.compute(
                call_uid=call_uid,
                wrapped_inputs=wrapped_inputs,
                returns=returns,
                apply_returns=apply_returns,
                compat_inputs=compat_inputs,
                c=c, is_retracing_superop=is_retracing_superop,
                pass_context_to_func=pass_context_to_func
            )
            end_time = time.time()
            self._cg_postcall(c=c, did_compute=True)
            op_call.exec_interval = (start_time, end_time)
            logging.debug(f'Computed function')
            returns = self.postprocess_outputs_as_returns(outputs_dict=wrapped_outputs)
            self.verify_mutations(wrapped_inputs=wrapped_inputs,
                                  returns=returns, first_time=True)
            if mode == MODES.run and call_data is None:
                # save calls only for calls that have not happened (like above)
                self.save_resulting_calls(
                    op_call=op_call,
                    output_calls=output_calls,
                    input_calls=None,
                    c=c, skip_delayed=True
                )
            #! mutations must be applied AFTER calls have been saved
            self.apply_mutations(wrapped_inputs=wrapped_inputs, returns=returns)
            if mode in (MODES.delete, MODES.capture):
                if mode == MODES.delete:
                    call_destination = 'deletion_buffer'
                elif mode == MODES.capture:
                    call_destination = 'captured_calls'
                else:
                    raise ValueError()
                self.track_calls(call_uid=call_uid,
                                 input_calls=input_calls,
                                 deeplazy_outputs=wrapped_outputs,
                                 output_calls=output_calls,
                                 c=c, destination=call_destination)
        ### case: skip to end of op
        else:
            if apply_returns:
                logging.warning('Call is recoverable, but __returns__ were provided; they will be ignored')
            assert call_data is not None
            deeplazy = (mode in (MODES.delete, MODES.capture))
            self._cg_precall(c=c, will_compute=False)
            wrapped_outputs = self.load_outputs(call_data=call_data,
                                                c=c, deeplazy=deeplazy)
            # TODO redundant
            self._cg_postcall(c=c, did_compute=False) 
            if mode in (MODES.delete, MODES.capture):
                if mode == MODES.delete:
                    call_destination = 'deletion_buffer'
                elif mode == MODES.capture:
                    call_destination = 'captured_calls'
                else:
                    raise ValueError()
                self.track_calls(call_uid=call_uid,
                                 input_calls=input_calls,
                                 output_calls=[],
                                 deeplazy_outputs=wrapped_outputs, c=c, 
                                 destination=call_destination)
            returns = self.postprocess_outputs_as_returns(
                outputs_dict=wrapped_outputs
            )
            self.verify_mutations(wrapped_inputs=wrapped_inputs,
                                  returns=returns, first_time=False)
            self.apply_mutations(wrapped_inputs=wrapped_inputs,
                                 returns=returns)
        return returns

    # needs to be updated
    # def mrun(self, marg_cols:TTuple[TUnion[TList[TAny], TAny],...], 
    #          mkwarg_cols:TDict[str, TUnion[TList[TAny], TTuple[TList[TAny]]]],
    #          c:Context, progress:bool=True):
    #     if self.op.mutations is not None:
    #         raise NotImplementedError()
    #     # process returns separately
    #     margs, mkwargs, mapply_returns, mreturns = self.mpreprocess_inputs(marg_cols=marg_cols, mkwarg_cols=mkwarg_cols)
    #     if not len(set(mapply_returns)) in [0, 1]:
    #         raise NotImplementedError()
    #     if len(mapply_returns) == 0:
    #         apply_returns_global = False
    #     else:
    #         apply_returns_global = set(mapply_returns).pop()
    #     if c.mode in (MODES.noop, MODES.transient, MODES.query):
    #         return [self.run(args=args, kwargs=kwargs) for args, kwargs in zip(margs, mkwargs)]
    #     ### figure out if we should apply defaults
    #     if c.mode == MODES.query:
    #         apply_defaults = CoreConfig.bind_defaults_in_queries
    #     else:
    #         apply_defaults = CoreConfig.enable_defaults
    #     ### 
    #     mcall_uid = []
    #     mcompute_data = []
    #     minput_calls = []
    #     for args, kwargs, apply_returns, returns in zip(margs, mkwargs, mapply_returns, mreturns):
    #         (inputs_dict, _, _, _,
    #         pass_context_to_func) = self.preprocess_inputs(args=args, kwargs=kwargs, apply_defaults=apply_defaults)
    #         if CoreConfig.superop_wrapping_style == CoreConfig.Exp.SuperopWrapping.construct_and_deconstruct:
    #             inputs_include_deconstructive = self.op.is_super
    #         elif CoreConfig.superop_wrapping_style == CoreConfig.Exp.SuperopWrapping.legacy:
    #             inputs_include_deconstructive = False
    #         else:
    #             raise ValueError()
    #         wrapped_inputs, input_calls, compat_inputs = self.wrap_inputs(args=(), kwargs=inputs_dict, include_deconstructive=inputs_include_deconstructive)
    #         call_uid = self.get_call_uid(wrapped_inputs=wrapped_inputs, c=c)
    #         mcall_uid.append(call_uid)
    #         mcompute_data.append((wrapped_inputs, apply_returns, returns, 
    #                               pass_context_to_func, compat_inputs))
    #         minput_calls.append(input_calls)
    #     if c.mode == MODES.delete:
    #         assert c.storage is not None
    #         call_locs = [c.storage.call_adapter.get_location(uid=uid, 
    #                      metadata={'partition': c.partition}) for uid in mcall_uid]
    #         # c._deletion_buffer.minsert(locs=call_locs)
    #         raise NotImplementedError()
    #     #! optimization: check call status in bulk
    #     mrecoverable, mcall_data = self.mget_recoverable(call_uids=mcall_uid, c=c)
    #     mmust_compute = [(not elt) for elt in mrecoverable]
    #     compute_data_groups = group_like(objs=mcompute_data, labels=mmust_compute)
    #     call_data_groups = group_like(objs=mcall_data, labels=mmust_compute)
    #     call_uid_groups = group_like(objs=mcall_uid, labels=mmust_compute)
    #     
    #     ### deal with computations
    #     #! optimization: attach inputs in bulk
    #     if self.op.is_super and self.OPTIMIZE_ATTACHMENT:
    #         pass
    #     else:
    #         if not apply_returns_global:
    #             self.mattach_inputs(mwrapped_inputs=[elt[0] for elt in compute_data_groups[True]], c=c)
    #     mwrapped_outputs_compute = []
    #     moutput_calls_compute = []
    #     mop_call_compute = []
    #     compute_iterator = zip(compute_data_groups[True], call_uid_groups[True])
    #     if progress:
    #         compute_iterator = tqdm.tqdm(compute_iterator)
    #     for compute_data, call_uid in compute_iterator:
    #         wrapped_inputs, apply_returns, returns, pass_context_to_func, compat_inputs = compute_data
    #         call_start = time.time()
    #         wrapped_outputs, output_calls, op_call = self.compute(call_uid=call_uid, 
    #                                                      wrapped_inputs=wrapped_inputs, 
    #                                                      returns=returns, 
    #                                                      apply_returns=apply_returns, 
    #                                                      compat_inputs=compat_inputs,
    #                                                      c=c, 
    #                                                      pass_context_to_func=pass_context_to_func)
    #         call_end = time.time()
    #         op_call.exec_interval = (call_start, call_end)
    #         mwrapped_outputs_compute.append(wrapped_outputs)
    #         moutput_calls_compute.append(output_calls)
    #         mop_call_compute.append(op_call)
    #     #! optimization: save calls and vrefs in bulk
    #     logging.debug('Saving new calls...')
    #     self.msave_resulting_calls(mop_calls=mop_call_compute, moutput_calls=moutput_calls_compute, 
    #                                minput_calls=minput_calls, c=c, 
    #                                skip_delayed=True)
    #     ### deal with loading 
    #     #! optimization: load existing calls in bulk
    #     logging.debug('Loading existing outputs...')
    #     mwrapped_outputs_load = self.mload_outputs(mcall_data=call_data_groups[False], c=c)
    #     mwrapped_outputs = ungroup_like(groups={True: mwrapped_outputs_compute, 
    #                                 False: mwrapped_outputs_load}, 
    #                         labels=mmust_compute)
    #     returns_list = [self.postprocess_outputs_as_returns(outputs_dict=wrapped_outputs) 
    #                     for wrapped_outputs in mwrapped_outputs]
    #     ### reformat as columns
    #     nout = len(self.op.sig.output_names)
    #     if nout == 0:
    #         return None
    #     elif nout == 1:
    #         return returns_list
    #     else:
    #         return tuple([[returns_list[i][j] for i in range(len(returns_list))] for j in range(nout)])
