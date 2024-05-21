import textwrap
from collections import deque
from common_imports import *
from mandala.common_imports import sess
from typing import Literal
from tps import *
from config import *
from utils import parse_args, dump_args, parse_output_name, dump_output_name, parse_returns
from utils import serialize, deserialize, get_content_hash

################################################################################
### model
################################################################################

class Ref:
    """
    Base class, should not be instantiated directly.
    """
    def __init__(self, cid: str, hid: str, in_memory: bool, obj: Optional[Any]) -> None:
        self.cid = cid
        self.hid = hid
        self.obj = obj
        self.in_memory = in_memory
    
    def with_hid(self, hid: str) -> "Ref":
        return type(self)(cid=self.cid, hid=hid, in_memory=self.in_memory, obj=self.obj)

    def __repr__(self) -> str:
        if self.in_memory:
            return f"Ref({self.obj}, hid={self.hid[:3]}...)"
        else:
            return f"Ref(hid={self.hid[:3]}..., in_memory={self.in_memory})"
    
    def __hash__(self) -> int:
        return hash(self.hid)
    
    def detached(self) -> "Ref":
        return type(self)(cid=self.cid, hid=self.hid, in_memory=False, obj=None)
    
    def attached(self, obj: Any) -> "Ref":
        return type(self)(cid=self.cid, hid=self.hid, in_memory=True, obj=obj)
    
    def shallow_copy(self) -> "Ref":
        return type(self)(cid=self.cid, hid=self.hid, in_memory=self.in_memory, obj=self.obj)


class AtomRef(Ref):
    def __repr__(self) -> str:
        return "Atom" + super().__repr__()


class Op:
    def __init__(self, name: str, f: Callable, 
                 nout: Union[Literal['var', 'auto'], int] = 'auto',
                 output_names: Optional[List[str]] = None,
                 skip_inputs: Optional[List[str]] = None,
                 skip_outputs: Optional[List[str]] = None,
                 version: Optional[int] = 0,
                 __structural__: bool = False,
                 ) -> None:
        self.name = name
        self.nout = nout
        self.version = version
        self.output_names = output_names
        self.__structural__ = __structural__
        self.f = f
    
    def __repr__(self) -> str:
        return f"Op({self.name}, nout={self.nout}, output_names={self.output_names}, version={self.version})"
    
    @property
    def id(self) -> str:
        return self.name
    
    def get_call_history_id(self, inputs: Dict[str, Ref]) -> str:
        return get_content_hash((
            {k: v.hid for k, v in inputs.items()},
            self.name,
            self.version
        ))
    
    def get_call_content_id(self, inputs: Dict[str, Ref]) -> str:
        return get_content_hash((
            {k: v.cid for k, v in inputs.items()},
            self.name,
            self.version
        ))
    
    def get_output_history_ids(self, call_history_id: str, output_names: List[str]) -> Dict[str, str]:
        return {k: get_content_hash((call_history_id, k)) for k in output_names}
    
    def get_ordered_outputs(self, output_dict: Dict[str, Any]) -> Tuple[Any, ...]:
        if self.output_names is None: # output names must be generic output_0, output_1, etc.
            output_dict_by_int = {parse_output_name(name): value for name, value in output_dict.items()}
            return tuple([output_dict_by_int[i] for i in range(len(output_dict))])
        else: # we use the order of the keys in self.output_names
            return tuple([output_dict[k] for k in self.output_names])
    
    def detached(self) -> "Op":
        return Op(name=self.name, f=None, nout=self.nout, output_names=self.output_names, version=self.version, __structural__=self.__structural__)

    def __call__(self, *args, **kwargs) -> Union[Tuple[Ref, ...], Ref]:
        if Context.current_context is None: # act as noop
            return self.f(*args, **kwargs)
        else:
            storage = Context.current_context.storage
            return storage.call(self, *args, __config__={"save_calls": True}, **kwargs)
    


class Call:
    def __init__(self, op: Op, cid: str, hid: str, inputs: Dict[str, Ref], outputs: Dict[str, Ref]) -> None:
        self.op = op
        self.cid = cid
        self.hid = hid
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self) -> str:
        return f"Call({self.op.name}, cid={self.hid[:3]}..., hid={self.hid[:3]}...)"
    
    def detached(self) -> "Call":
        return Call(op=self.op,
                    cid=self.cid,
                    hid=self.hid,
                    inputs={k: v.detached() for k, v in self.inputs.items()},
                    outputs={k: v.detached() for k, v in self.outputs.items()})


def wrap_atom(obj: Any, history_id: Optional[str] = None) -> AtomRef:
    if isinstance(obj, Ref):
        assert history_id is None
        return obj
    uid = get_content_hash(obj)
    if history_id is None:
        history_id = get_content_hash(uid)
    return AtomRef(cid=uid, hid=history_id, in_memory=True, obj=obj)


### native support for some collections
class ListRef(Ref):

    def __len__(self) -> int:
        return len(self.obj)
    
    def __getitem__(self, i: int) -> Ref:
        assert self.in_memory
        return self.obj[i]
    
    def __repr__(self) -> str:
        return "List" + super().__repr__()
    
    def shape(self) -> "ListRef":
        return ListRef(cid=self.cid, hid=self.hid, in_memory=True,
                       obj=[elt.detached() for elt in self.obj])
    
    

class DictRef(Ref):
    pass


class TupleRef(Ref):
    pass


class SetRef(Ref):
    
    def __repr__(self) -> str:
        return "Set" + super().__repr__()
    
    def __len__(self) -> int:
        return len(self.obj)
    
    def __iter__(self):
        return iter(self.obj)
    
    def __contains__(self, elt: Ref) -> bool:
        return elt in self.obj


def recurse_on_ref_collections(f: Callable, obj: Any, **kwargs: Any) -> Any:
    if isinstance(obj, AtomRef):
        return f(obj, **kwargs)
    elif isinstance(obj, (list, ListRef)):
        return [recurse_on_ref_collections(f, elt, **kwargs) for elt in obj]
    elif isinstance(obj, dict):
        return {k: recurse_on_ref_collections(f, v, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(recurse_on_ref_collections(f, elt, **kwargs) for elt in obj)
    elif isinstance(obj, set):
        return {recurse_on_ref_collections(f, elt, **kwargs) for elt in obj}
    else:
        return obj

def __make_list__(**kwargs: Any) -> MList[Any]:
    elts = [kwargs[f"elts_{i}"] for i in range(len(kwargs))]
    return ListRef(cid=get_content_hash([elt.cid for elt in elts]), hid=get_content_hash([elt.hid for elt in elts]), in_memory=True, obj=elts)

def __make_set__(**kwargs: Any) -> MSet[Any]:
    elts = [kwargs[f"elts_{i}"] for i in range(len(kwargs))]
    return SetRef(
        cid=get_content_hash(sorted([elt.cid for elt in elts])),
        hid=get_content_hash(sorted([elt.hid for elt in elts])),
        in_memory=True,
        obj=set(elts)
    )

def __make_dict__(**items: Any) -> dict:
    keys = [items[f'key_{i}'] for i in range(len(items))]
    values = [items[f'value_{i}'] for i in range(len(items))]
    obj = {k: v for k, v in zip(keys, values)}
    return DictRef(
        cid=get_content_hash(sorted([(k.cid, v.cid) for k, v in obj.items()])),
        hid=get_content_hash(sorted([(k.hid, v.hid) for k, v in obj.items()])),
        in_memory=True,
        obj=obj
    )

def __make_tuple__(*elts: Any) -> tuple:
    return tuple(elts)

def __get_item__(obj: MList[Any], attr: Any) -> Any:
    return obj[attr.obj]


__make_list__ = Op(name=__make_list__.__name__, f=__make_list__, __structural__=True)
__make_dict__ = Op(name=__make_dict__.__name__, f=__make_dict__, __structural__=True)
__make_set__ = Op(name=__make_set__.__name__, f=__make_set__, __structural__=True)
__make_tuple__ = Op(name=__make_tuple__.__name__, f=__make_tuple__, __structural__=True)
__get_item__ = Op(name=__get_item__.__name__, f=__get_item__, __structural__=True)


def make_ref_set(resf: Iterable[Ref]) -> SetRef:
    return __make_set__.f(**{f"elts_{i}": elt for i, elt in enumerate(resf)})




from ui import Context