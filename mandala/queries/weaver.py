from abc import ABC, abstractmethod
from ..common_imports import *
from ..core.model import FuncOp, Ref, wrap_atom
from ..core.wrapping import causify_atom
from ..core.config import dump_output_name, parse_output_idx
from ..core.tps import Type, ListType, DictType, SetType, AnyType, StructType
from ..core.builtins_ import Builtins, ListRef, DictRef, SetRef
from ..core.utils import Hashing, concat_lists, invert_dict
from typing import Literal


class StructOrientations:
    # at runtime only
    construct = "construct"
    destruct = "destruct"


class Node(ABC):
    @abstractmethod
    def neighbors(
        self, direction: Literal["backward", "forward", "both"] = "both"
    ) -> List["Node"]:
        raise NotImplementedError


class ValNode(Node):
    def __init__(
        self,
        constraint: Optional[List[str]],
        tp: Optional[Type],
        creators: Optional[List["CallNode"]] = None,
        created_as: Optional[List[str]] = None,
        _label: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.creators = [] if creators is None else creators
        self.created_as = [] if created_as is None else created_as
        self.consumers: List["CallNode"] = []
        self.consumed_as: List[str] = []
        self.name = name
        self.constraint = constraint
        self.tp = tp
        self._label: Optional[str] = _label

    def add_consumer(self, consumer: "CallNode", consumed_as: str):
        self.consumers.append(consumer)
        self.consumed_as.append(consumed_as)

    def add_creator(self, creator: "CallNode", created_as: str):
        assert isinstance(created_as, str)
        self.creators.append(creator)
        self.created_as.append(created_as)

    def neighbors(
        self, direction: Literal["backward", "forward", "both"] = "both"
    ) -> Set["CallNode"]:
        backward = self.creators
        forward = self.consumers
        if direction == "backward":
            return set(backward)
        elif direction == "forward":
            return set(forward)
        elif direction == "both":
            return set(backward + forward)

    def named(self, name: str) -> Any:
        self.name = name
        return self

    def __getitem__(self, idx: Union[int, str, "ValNode"]) -> "ValNode":
        tp = self.tp or _infer_type(self)
        if isinstance(tp, ListType):
            return BuiltinQueries.GetListItemQuery(
                lst=self, idx=qwrap(idx, tp=tp.elt_type)
            )
        elif isinstance(tp, DictType):
            assert isinstance(idx, (ValNode, str))
            return BuiltinQueries.GetDictItemQuery(
                dct=self, key=qwrap(idx, tp=tp.elt_type)
            )
        else:
            raise NotImplementedError(f"Cannot index into query of type {tp}")

    def __repr__(self):
        if self.name is not None:
            return f"ValNode({self.name})"
        else:
            return f"ValNode({self.tp})"

    def get_constraint(self, *values) -> List[str]:
        wrapped = [wrap_atom(v) for v in values]
        for w in wrapped:
            causify_atom(w)
        return [w.full_uid for w in wrapped]

    def pin(self, *values):
        if len(values) == 0:
            raise ValueError("Must pin to at least one value")
        else:
            self.constraint = self.get_constraint(*values)

    def unpin(self):
        self.constraint = None


def copy(vq: ValNode, label: Optional[str] = None) -> ValNode:
    return ValNode(
        constraint=vq.constraint,
        tp=vq.tp,
        creators=vq.creators,
        created_as=vq.created_as,
        _label=label,
    )


class CallNode(Node):
    def __init__(
        self,
        inputs: Dict[str, ValNode],
        func_op: FuncOp,
        outputs: Dict[str, ValNode],
        constraint: Optional[List[str]],
        orientation: Optional[str] = None,
    ):
        self.func_op = func_op
        self.inputs = inputs
        self.outputs = outputs
        self.orientation = orientation
        self.constraint = constraint

    def set_outputs(self, outputs: Dict[str, ValNode]):
        self.outputs = outputs

    @property
    def returns(self) -> List[ValNode]:
        if self.func_op.is_builtin:
            raise NotImplementedError()
        else:
            ord_outputs = {parse_output_idx(k): v for k, v in self.outputs.items()}
            ord_outputs = [ord_outputs[i] for i in range(len(ord_outputs))]
            return ord_outputs

    @property
    def returns_interp(self) -> List[Optional[ValNode]]:
        if self.func_op.is_builtin:
            raise NotImplementedError()
        else:
            ord_outputs = {parse_output_idx(k): v for k, v in self.outputs.items()}
            res = []
            for i in range(self.func_op.sig.n_outputs):
                if i in ord_outputs:
                    res.append(ord_outputs[i])
                else:
                    res.append(None)
            return res

    @property
    def displayname(self) -> str:
        data = {
            "__list__": {"construct": "__list__", "destruct": "__getitem__"},
            "__dict__": {"construct": "__dict__", "destruct": "__getitem__"},
            "__set__": {"construct": "__set__", "destruct": "__getitem__"},
        }
        if self.func_op._is_builtin:
            return data[self.func_op.sig.ui_name][self.orientation]
        else:
            return self.func_op.sig.ui_name

    def neighbors(
        self, direction: Literal["backward", "forward", "both"] = "both"
    ) -> Set[ValNode]:
        backward = list(self.inputs.values())
        forward = list(self.outputs.values())
        if direction == "backward":
            return set(backward)
        elif direction == "forward":
            return set(forward)
        elif direction == "both":
            return set(backward + forward)

    @staticmethod
    def link(
        inputs: Dict[str, ValNode],
        func_op: FuncOp,
        outputs: Dict[str, ValNode],
        constraint: Optional[List[str]],
        orientation: Optional[str],
        include_indexing: bool = True,
    ) -> "CallNode":
        """
        Link a func query into the graph
        """
        if func_op._is_builtin:
            struct_id = func_op.sig.ui_name
            assert orientation is not None
            joined = {**inputs, **outputs}
            if not include_indexing:
                if struct_id == "__list__":
                    joined = {k: v for k, v in joined.items() if k != "idx"}
                if struct_id == "__dict__":
                    joined = {k: v for k, v in joined.items() if k != "key"}
            input_keys = Builtins.IO[orientation][struct_id]["in"] & joined.keys()
            output_keys = Builtins.IO[orientation][struct_id]["out"] & joined.keys()
            effective_inputs = {k: joined[k] for k in input_keys}
            effective_outputs = {k: joined[k] for k in output_keys}
        else:
            assert orientation is None
            effective_inputs = inputs
            effective_outputs = outputs
        result = CallNode(
            inputs=effective_inputs,
            func_op=func_op,
            outputs=effective_outputs,
            orientation=orientation,
            constraint=constraint,
        )
        for name, inp in effective_inputs.items():
            inp.add_consumer(consumer=result, consumed_as=name)
        for name, out in effective_outputs.items():
            out.add_creator(creator=result, created_as=name)
        return result

    def unlink(self):
        """
        Remove this `FuncQuery` from the graph.
        """
        for inp in self.inputs.values():
            idxs = [i for i, x in enumerate(inp.consumers) if x is self]
            inp.consumers = [x for i, x in enumerate(inp.consumers) if i not in idxs]
            inp.consumed_as = [
                x for i, x in enumerate(inp.consumed_as) if i not in idxs
            ]
        for out in self.outputs.values():
            idxs = [i for i, x in enumerate(out.creators) if x is self]
            out.creators = [x for i, x in enumerate(out.creators) if i not in idxs]
            out.created_as = [x for i, x in enumerate(out.created_as) if i not in idxs]

    def __repr__(self):
        args_string = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
        if self.orientation is not None:
            args_string += f", orientation={self.orientation}"
        return f"CallNode({self.func_op.sig.ui_name}, {args_string})"


def traverse_all(
    vqs: Set[ValNode],
    direction: Literal["backward", "forward", "both"] = "both",
) -> Tuple[Set[ValNode], Set[CallNode]]:
    """
    Extend the given `ValQuery` objects to all objects connected to them through
    function inputs and/or outputs.
    """
    vqs_ = {_ for _ in vqs}
    fqs_: Set[CallNode] = set()
    found_new = True
    while found_new:
        found_new = False
        val_neighbors = concat_lists([v.neighbors(direction=direction) for v in vqs_])
        op_neighbors = concat_lists([o.neighbors(direction=direction) for o in fqs_])
        if any(k not in fqs_ for k in val_neighbors):
            found_new = True
            for neigh in val_neighbors:
                if neigh not in fqs_:
                    fqs_.add(neigh)
        if any(k not in vqs_ for k in op_neighbors):
            found_new = True
            for neigh in op_neighbors:
                if neigh not in vqs_:
                    vqs_.add(neigh)
    return vqs_, fqs_


class BuiltinQueries:
    @staticmethod
    def ListQ(
        elts: List[ValNode], idxs: Optional[List[Optional[ValNode]]] = None
    ) -> ValNode:
        result = ValNode(
            creators=[], created_as=[], tp=ListType(elt_type=AnyType()), constraint=None
        )
        if idxs is None:
            idxs = [
                ValNode(constraint=None, tp=AnyType(), creators=[], created_as=[])
                for _ in elts
            ]
        for elt, idx in zip(elts, idxs):
            CallNode.link(
                inputs={"lst": result, "elt": elt, "idx": idx},
                func_op=Builtins.list_op,
                outputs={},
                constraint=None,
                orientation=StructOrientations.construct,
            )
        return result

    @staticmethod
    def DictQ(dct: Dict[ValNode, ValNode]) -> ValNode:
        result = ValNode(
            creators=[], created_as=[], tp=DictType(elt_type=AnyType()), constraint=None
        )
        for key, val in dct.items():
            CallNode.link(
                inputs={"dct": result, "key": key, "val": val},
                func_op=Builtins.dict_op,
                outputs={},
                constraint=None,
                orientation=StructOrientations.construct,
            )
        return result

    @staticmethod
    def SetQ(elts: Set[ValNode]) -> ValNode:
        result = ValNode(
            creators=[], created_as=[], tp=SetType(elt_type=AnyType()), constraint=None
        )
        for elt in elts:
            CallNode.link(
                inputs={"st": result, "elt": elt},
                func_op=Builtins.set_op,
                outputs={},
                constraint=None,
                orientation=StructOrientations.construct,
            )
        return result

    @staticmethod
    def GetListItemQuery(lst: ValNode, idx: Optional[ValNode] = None) -> ValNode:
        elt_tp = lst.tp.elt_type if isinstance(lst.tp, ListType) else None
        result = ValNode(creators=[], created_as=[], tp=elt_tp, constraint=None)
        CallNode.link(
            inputs={"lst": lst, "elt": result, "idx": idx},
            func_op=Builtins.list_op,
            outputs={},
            orientation=StructOrientations.destruct,
            constraint=None,
        )
        return result

    @staticmethod
    def GetDictItemQuery(dct: ValNode, key: Optional[ValNode] = None) -> ValNode:
        val_tp = dct.tp.elt_type if isinstance(dct.tp, DictType) else None
        result = ValNode(creators=[], created_as=[], tp=val_tp, constraint=None)
        CallNode.link(
            inputs={"dct": dct, "key": key, "val": result},
            func_op=Builtins.dict_op,
            outputs={},
            orientation=StructOrientations.destruct,
            constraint=None,
        )
        return result

    ############################################################################
    ### syntactic sugar
    ############################################################################
    @staticmethod
    def is_pattern(obj: Any) -> bool:
        if type(obj) is list and Ellipsis in obj:
            return all(
                BuiltinQueries.is_pattern(elt) or isinstance(elt, ValNode)
                for elt in obj
                if elt is not Ellipsis
            )
        elif type(obj) is dict and Ellipsis in obj:
            return all(
                BuiltinQueries.is_pattern(elt) or isinstance(elt, ValNode)
                for elt in obj.values()
                if elt is not Ellipsis
            )
        elif type(obj) is set:
            return all(
                BuiltinQueries.is_pattern(elt) or isinstance(elt, ValNode)
                for elt in obj
                if elt is not Ellipsis
            )
        else:
            return False

    @staticmethod
    def link_pattern(obj: Union[list, dict, set, ValNode]) -> ValNode:
        if isinstance(obj, ValNode):
            return obj
        elif type(obj) is list:
            elts = [
                BuiltinQueries.link_pattern(elt) for elt in obj if elt is not Ellipsis
            ]
            result = ValNode(
                creators=[],
                created_as=[],
                tp=ListType(elt_type=AnyType()),
                constraint=None,
            )
            for elt in elts:
                CallNode.link(
                    inputs={
                        "lst": result,
                        "elt": elt,
                        "idx": ValNode(
                            constraint=None, tp=AnyType(), creators=[], created_as=[]
                        ),
                    },
                    func_op=Builtins.list_op,
                    outputs={},
                    constraint=None,
                    orientation=StructOrientations.construct,
                )
        elif type(obj) is dict:
            elts = {
                k: BuiltinQueries.link_pattern(v)
                for k, v in obj.items()
                if k is not Ellipsis
            }
            result = ValNode(
                creators=[],
                created_as=[],
                tp=DictType(elt_type=AnyType()),
                constraint=None,
            )
            for k, v in elts.items():
                CallNode.link(
                    inputs={"dct": result, "key": k, "val": v},
                    func_op=Builtins.dict_op,
                    outputs={},
                    constraint=None,
                    orientation=StructOrientations.construct,
                )
        elif type(obj) is set:
            elts = {
                BuiltinQueries.link_pattern(elt) for elt in obj if elt is not Ellipsis
            }
            result = ValNode(
                creators=[],
                created_as=[],
                tp=SetType(elt_type=AnyType()),
                constraint=None,
            )
            for elt in elts:
                CallNode.link(
                    inputs={"st": result, "elt": elt},
                    func_op=Builtins.set_op,
                    outputs={},
                    constraint=None,
                    orientation=StructOrientations.construct,
                )
        else:
            raise ValueError
        return result


def qwrap(obj: Any, tp: Optional[Type] = None, strict: bool = False) -> ValNode:
    """
    Produce a ValQuery from an object.
    """
    if isinstance(obj, ValNode):
        return obj
    elif isinstance(obj, Ref):
        assert obj._query is not None, "Ref must be linked to a query"
        return obj.query
    elif BuiltinQueries.is_pattern(obj=obj):
        if not strict:
            return BuiltinQueries.link_pattern(obj=obj)
        else:
            raise ValueError
    else:
        if strict:
            raise ValueError("value must be a `ValQuery` or `Ref`")
        if tp is None:
            tp = AnyType()
        # wrap a raw value as a pointwise constraint
        uid = obj.uid if isinstance(obj, Ref) else Hashing.get_content_hash(obj)
        return ValNode(
            tp=tp,
            creators=[],
            created_as=[],
            constraint=[uid],
        )


def call_query(
    func_op: FuncOp, inputs: Dict[str, Union[list, dict, set, ValNode, Ref, Any]]
) -> List[ValNode]:
    for k in inputs.keys():
        inputs[k] = qwrap(obj=inputs[k])
    assert all(isinstance(inp, ValNode) for inp in inputs.values())
    ord_outputs = [
        ValNode(creators=[], created_as=[], tp=tp, constraint=None)
        for tp in func_op.output_types
    ]
    outputs = {dump_output_name(index=i): o for i, o in enumerate(ord_outputs)}
    CallNode.link(
        inputs=inputs,
        func_op=func_op,
        outputs=outputs,
        orientation=None,
        constraint=None,
    )
    return ord_outputs


################################################################################
### introspection
################################################################################
def _infer_type(val_query: ValNode) -> Type:
    consumer_op_names = [c.func_op.sig.ui_name for c in val_query.consumers]
    mapping = {"__list__": ListType(), "__dict__": DictType(), "__set__": SetType()}
    tps = [mapping.get(x, None) for x in consumer_op_names]
    struct_tps = [x for x in tps if x is not None]
    if len(struct_tps) == 0:
        return AnyType()
    elif len(struct_tps) == 1:
        return struct_tps[0]
    else:
        raise RuntimeError(f"Multiple types for {val_query}: {struct_tps}")


def get_vq_orientation(vq: ValNode) -> str:
    if not isinstance(vq.tp, StructType):
        raise ValueError
    if (
        len(vq.creators) == 1
        and vq.creators[0].orientation == StructOrientations.destruct
    ):
        return StructOrientations.destruct
    elif len(vq.creators) == 1 and vq.creators[0].orientation is None:
        return StructOrientations.destruct
    else:
        return StructOrientations.construct


def is_idx(vq: ValNode) -> bool:
    for consumer, consumed_as in zip(vq.consumers, vq.consumed_as):
        if consumed_as == "idx" and consumer.func_op.sig.ui_name == "__list__":
            return True
    return False


def is_key(vq: ValNode) -> bool:
    for consumer, consumed_as in zip(vq.consumers, vq.consumed_as):
        if consumed_as == "key" and consumer.func_op.sig.ui_name == "__dict__":
            return True
    return False


def get_elt_fqs(vq: ValNode) -> List[CallNode]:
    assert isinstance(vq.tp, StructType)
    struct_id = vq.tp.struct_id
    orientation = get_vq_orientation(vq)
    fqs_to_search = (
        vq.consumers if orientation == StructOrientations.destruct else vq.creators
    )
    fqs = [
        fq
        for fq in fqs_to_search
        if fq.func_op.sig.ui_name == struct_id and fq.orientation == orientation
    ]
    return fqs


def get_elts(vq: ValNode) -> Dict[CallNode, ValNode]:
    """
    Get the constituent element queries of a set, as a dictionary of {fq: vq}
    pairs.
    """
    orientation = get_vq_orientation(vq)
    elt_fqs = get_elt_fqs(vq)
    return {
        fq: fq.inputs["elt"]
        if orientation == StructOrientations.construct
        else fq.outputs["elt"]
        for fq in elt_fqs
    }


def get_items(vq: ValNode) -> Dict[CallNode, Tuple[Optional[ValNode], ValNode]]:
    """
    Get the constituent elements and indices of a list or dict in the form of
    {fq: (idx_vq, elt_vq)} pairs.
    """
    assert isinstance(vq.tp, (ListType, DictType))
    orientation = get_vq_orientation(vq)
    elt_fqs = get_elt_fqs(vq)
    elt_key = "elt" if isinstance(vq.tp, ListType) else "val"
    idx_key = "idx" if isinstance(vq.tp, ListType) else "key"
    return {
        fq: (fq.inputs.get(idx_key), fq.inputs[elt_key])
        if orientation == StructOrientations.destruct
        else (fq.inputs.get(idx_key), fq.inputs[elt_key])
        for fq in elt_fqs
    }


def get_elt_and_struct(fq: CallNode) -> Tuple[ValNode, ValNode]:
    assert fq.func_op.is_builtin
    struct_id = fq.func_op.sig.ui_name
    elt_target = (
        fq.outputs if fq.orientation == StructOrientations.destruct else fq.inputs
    )
    struct_target = (
        fq.inputs if fq.orientation == StructOrientations.destruct else fq.outputs
    )
    if struct_id == "__list__":
        return elt_target["elt"], struct_target["lst"]
    elif struct_id == "__dict__":
        return elt_target["val"], struct_target["dct"]
    elif struct_id == "__set__":
        return elt_target["elt"], struct_target["st"]
    else:
        raise NotImplementedError()


def get_idx(fq: CallNode) -> Optional[ValNode]:
    assert fq.func_op.is_builtin
    struct_id = fq.func_op.sig.ui_name
    idx_target = fq.inputs
    if struct_id == "__list__":
        return idx_target.get("idx", None)
    elif struct_id == "__dict__":
        return idx_target.get("key", None)
    else:
        raise ValueError


def prepare_query(ref: Ref, tp: Type):
    if ref._query is None:
        ref._query = ValNode(tp=tp, constraint=None, creators=[], created_as=[])
