from networkx import MultiDiGraph
from networkx.algorithms import topological_sort

from ..core.bases import Call, ValueRef, unwrap
from ..adapters.ops import BaseOpAdapter
from ..adapters.vals import BaseValAdapter
from ..util.common_ut import invert_dict
from ..common_imports import *

class CallNode(object):
    def __init__(self, call:Call):
        self._call = call
        self._inputs = {}
        self._outputs = {}

    @property
    def inputs(self) -> TDict[str, 'VRefNode']:
        return self._inputs
    
    @property
    def outputs(self) -> TDict[str, 'VRefNode']:
        return self._outputs
    
    def weave_input(self, name: str, inp:'VRefNode'):
        self._inputs[name] = inp
        inp.add_consumer(consumer=self, consumed_as=name)
        
    def weave_output(self, name: str, outp: 'VRefNode'):
        self._outputs[name] = outp
        outp.add_creator(creator=self, created_as=name)

    @property
    def call(self) -> Call:
        return self._call


class VRefNode(object):
    def __init__(self, vref:ValueRef):
        self._vref = vref
        self._creators = []
        self._created_as = []
        self._consumers = []
        self._consumed_as = []
        self._display_name_data = None
    
    @property
    def creators(self) -> TList[CallNode]:
        return self._creators
    
    @property
    def created_as(self) -> TList[str]:
        return self._created_as
    
    @property
    def consumers(self) -> TList[CallNode]:
        return self._consumers
    
    @property
    def consumed_as(self) -> TList[str]:
        return self._consumed_as
    
    def add_creator(self, creator:CallNode, created_as:str):
        self._creators.append(creator)
        self._created_as.append(created_as)
    
    def add_consumer(self, consumer:CallNode, consumed_as:str):
        self._consumers.append(consumer)
        self._consumed_as.append(consumed_as)

    @property
    def vref(self) -> ValueRef:
        return self._vref
    
    @property
    def display_name_data(self) -> TOption[TTuple[str, int]]:
        return self._display_name_data
    
    @property
    def display_name(self) -> str:
        assert self.display_name_data is not None
        name, suffix = self.display_name_data
        if suffix == 0:
            return name
        else:
            return f'{name}_{suffix}'


def get_or_create_call_node(container:TDict[str, CallNode], 
                            call:Call) -> CallNode:
    call_uid = call.uid
    if call_uid not in container:
        call_node = CallNode(call=call)
        container[call_uid] = call_node
    else:
        call_node = container[call_uid]
    return call_node

def get_or_create_vref_node(container:TDict[str, VRefNode],
                            vref:ValueRef) -> VRefNode:
    vref_uid = vref.uid
    if vref_uid not in container:
        vref_node = VRefNode(vref=vref)
        container[vref_uid] = vref_node
    else:
        vref_node = container[vref_uid]
    return vref_node

def weave_calls(calls:TList[Call],
                ) -> TTuple[TDict[str, VRefNode], TDict[str, CallNode]]:
    vref_nodes = {}
    call_nodes = {}
    for call in calls:
        call_node = get_or_create_call_node(container=call_nodes, call=call)
        for input_name, input_vref in call.inputs.items():
            input_node = get_or_create_vref_node(container=vref_nodes,
                                                 vref=input_vref)
            call_node.weave_input(name=input_name, inp=input_node)
        for output_name, output_vref in call.outputs.items():
            output_node = get_or_create_vref_node(container=vref_nodes,
                                                  vref=output_vref)
            call_node.weave_output(name=output_name, outp=output_node)
    return vref_nodes, call_nodes

def weave_to_nx(vref_nodes, call_nodes) -> nx.MultiDiGraph:
    nx_graph = MultiDiGraph()
    for vref_uid, vref_node in vref_nodes.items():
        nx_graph.add_node(vref_uid)
    for call_uid, call_node in call_nodes.items():
        call = call_node.call
        nx_graph.add_node(call_uid)
        for input_name, input_vref in call.inputs.items():
            # input ---> call
            nx_graph.add_edge(u_for_edge=input_vref.uid, 
                              v_for_edge=call_uid, key=input_name)
        for output_name, output_vref in call.outputs.items():
            # call ---> output
            nx_graph.add_edge(u_for_edge=call_uid,
                              v_for_edge=output_vref.uid, key=output_name)
    return nx_graph

def get_call_line(call_node:CallNode, op_adapter:BaseOpAdapter) -> str:
    call = call_node.call
    op = call.op
    output_names = op.sig.output_names
    output_display_names = [call_node.outputs[output_name].display_name
                            for output_name in output_names]
    return op.repr_call(call=call, output_names=output_display_names, 
                        input_reprs={k: v.display_name 
                                     for k, v in call_node.inputs.items()})

def sanitize_output_name(name:str) -> str:
    if name == 'list':
        return 'lst'
    if name == 'dict':
        return 'dct'
    else:
        return name

def print_history(calls:TList[Call], op_adapter:BaseOpAdapter,
                  val_adapter:BaseValAdapter) -> str:
    vref_nodes, call_nodes = weave_calls(calls=calls)
    nx_graph = weave_to_nx(vref_nodes=vref_nodes, call_nodes=call_nodes)
    topsort = list(topological_sort(G=nx_graph))
    # traverse and generate names
    name_data = {} # name: current largest prefix
    for uid in topsort:
        if uid in call_nodes:
            call_node = call_nodes[uid]
            op = call_node.call.op
            if not op.is_builtin:
                internal_to_ui_outputs = invert_dict(
                    op_adapter.get_ui_to_internal_interface(
                        op=op, which='outputs'
                    )
                )
            else:
                internal_to_ui_outputs = {k: k for k in 
                                          call_node.call.outputs.keys()}
            inputs = call_node.inputs
            for _, input_node in inputs.items():
                if input_node.display_name_data is None:
                    # this is a global input
                    vref = input_node.vref
                    loc = val_adapter.get_vref_location(vref=vref)
                    value = unwrap(val_adapter.get(loc))
                    input_node._display_name_data = (repr(value), 0)
            outputs = call_node.outputs
            for output_name, output_node in outputs.items():
                if output_node.display_name_data is None:
                    name = internal_to_ui_outputs[output_name]
                    name = sanitize_output_name(name=name)
                    if name in name_data:
                        current_suffix = name_data[name]
                        name_data[name] = current_suffix + 1 
                        output_node._display_name_data = (name, 
                                                          current_suffix + 1)
                    else:
                        name_data[name] = 0
                        output_node._display_name_data = (name, 0)
    # generate strings 
    lines = []
    for uid in topsort:
        if uid in call_nodes:
            lines.append(get_call_line(call_node=call_nodes[uid],
                                       op_adapter=op_adapter))
    return '\n'.join(lines)