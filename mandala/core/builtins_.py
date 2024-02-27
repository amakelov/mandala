from collections.abc import Sequence, Mapping, Set as SetABC
from ..common_imports import *

from .config import MODES
from .model import Ref, FuncOp, Call, wrap_atom, ValueRef
from .utils import Hashing
from .tps import AnyType

HASHERS = {
    "__list__": Hashing.hash_list,
    "__set__": Hashing.hash_multiset,
    "__dict__": Hashing.hash_dict,
}


class StructRef(Ref):
    builtin_id = None

    def __init__(
        self,
        uid: Optional[str],
        obj: Optional[Any],
        in_memory: bool,
        transient: bool = False,
    ):
        if uid is None:
            builtin_id = self.builtin_id
            hasher = HASHERS[builtin_id]
            uids = Builtins.map(func=lambda elt: elt.uid, obj=obj, struct_id=builtin_id)
            uid = Builtins._make_builtin_uid(uid=hasher(uids), builtin_id=builtin_id)
        super().__init__(uid=uid, obj=obj, in_memory=in_memory, transient=transient)

    def as_inputs_list(self) -> List[Dict[str, Ref]]:
        raise NotImplementedError

    def causify_up(self):
        raise NotImplementedError

    def get_call(self, wrapped_inputs: Dict[str, Ref]) -> Call:
        call_uid = Builtins.OPS[self.builtin_id].get_pre_call_uid(
            input_uids={k: v.uid for k, v in wrapped_inputs.items()}
        )
        return Call(
            uid=call_uid,
            inputs=wrapped_inputs,
            outputs=[],
            func_op=Builtins.OPS[self.builtin_id],
            transient=False,
        )

    def get_calls(self) -> List[Call]:
        inputs_list = self.as_inputs_list()
        res = []
        for wrapped_inputs in inputs_list:
            res.append(self.get_call(wrapped_inputs=wrapped_inputs))
        return res

    @staticmethod
    def map(obj: Iterable, func: Callable) -> Iterable:
        raise NotImplementedError

    @staticmethod
    def elts(obj: Iterable) -> Iterable:
        raise NotImplementedError

    def children(self) -> Iterable[Ref]:
        return self.elts(self.obj)

    def unlinked(self, keep_causal: bool) -> "StructRef":
        if not self.in_memory:
            res = type(self)(uid=self.uid, obj=None, in_memory=False)
            if keep_causal:
                res._causal_uid = self._causal_uid
            return res
        else:
            unlinked_elts = type(self).map(
                obj=self.obj, func=lambda elt: elt.unlinked(keep_causal=keep_causal)
            )
            res = type(self)(uid=self.uid, obj=unlinked_elts, in_memory=True)
            if keep_causal:
                res._causal_uid = self._causal_uid
            return res


class ListRef(StructRef, Sequence):
    """
    Immutable list of Refs.
    """

    builtin_id = "__list__"

    def as_inputs_list(self) -> List[Dict[str, Ref]]:
        idxs = [wrap_atom(idx) for idx in range(len(self))]
        for idx in idxs:
            causify_atom(idx)
        wrapped_inputs_list = [
            {"lst": self, "elt": elt, "idx": idx} for elt, idx in zip(self.obj, idxs)
        ]
        return wrapped_inputs_list

    def causify_up(self):
        assert all(elt.causal_uid is not None for elt in self.obj)
        self.causal_uid = Hashing.hash_list([elt.causal_uid for elt in self.obj])

    @staticmethod
    def map(obj: list, func: Callable) -> list:
        return [func(elt) for elt in obj]

    @staticmethod
    def elts(obj: list) -> list:
        return obj

    def dump(self) -> "ListRef":
        return ListRef(
            uid=self.uid, obj=[vref.detached() for vref in self.obj], in_memory=True
        )

    ############################################################################
    ### list interface
    ############################################################################
    def __getitem__(
        self, idx: Union[int, "ValNode", Ref, slice]
    ) -> Union[Ref, "ValNode"]:
        self._auto_attach()
        if isinstance(idx, Ref):
            prepare_query(ref=idx, tp=AnyType())
            res_query = BuiltinQueries.GetListItemQuery(lst=self.query, idx=idx.query)
            res = self.obj[idx.obj].unlinked(keep_causal=True)
            res._query = res_query
            return res
        elif isinstance(idx, ValNode):
            res = BuiltinQueries.GetListItemQuery(lst=self.query, idx=idx)
            return res
        elif isinstance(idx, int):
            if (
                GlobalContext.current is not None
                and GlobalContext.current.mode != MODES.run
            ):
                raise ValueError
            res: Ref = self.obj[idx]
            res = res.unlinked(keep_causal=True)
            wrapped_idx = wrap_atom(obj=idx)
            causify_atom(ref=wrapped_idx)
            call = self.get_call(
                wrapped_inputs={"lst": self, "idx": wrapped_idx, "elt": res}
            )
            call.link(orientation=StructOrientations.destruct)
            return res
        elif isinstance(idx, slice):
            if (
                GlobalContext.current is not None
                and GlobalContext.current.mode != MODES.run
            ):
                raise ValueError
            res = self.obj[idx]
            res = [elt.unlinked(keep_causal=True) for elt in res]
            wrapped_idxs = [wrap_atom(obj=i) for i in range(*idx.indices(len(self)))]
            for wrapped_idx in wrapped_idxs:
                causify_atom(ref=wrapped_idx)
            for wrapped_idx, elt in zip(wrapped_idxs, res):
                call = self.get_call(
                    wrapped_inputs={"lst": self, "idx": wrapped_idx, "elt": elt}
                )
                call.link(orientation=StructOrientations.destruct)
            return res

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class DictRef(StructRef):  # don't inherit from Mapping because it's not hashable
    """
    Immutable string-keyed dict of Refs.
    """

    builtin_id = "__dict__"

    def as_inputs_list(self) -> List[Dict[str, Ref]]:
        keys = {k: wrap_atom(k) for k in self.obj.keys()}
        for k in keys.values():
            causify_atom(k)
        wrapped_inputs_list = [
            {"dct": self, "key": keys[k], "val": v} for k, v in self.obj.items()
        ]
        return wrapped_inputs_list

    def causify_up(self):
        assert all(elt.causal_uid is not None for elt in self.obj.values())
        self.causal_uid = Hashing.hash_dict(
            {k: v.causal_uid for k, v in self.obj.items()}
        )

    @staticmethod
    def map(obj: dict, func: Callable) -> dict:
        return {k: func(v) for k, v in obj.items()}

    @staticmethod
    def elts(obj: dict) -> Iterable:
        return iter(obj.values())

    def dump(self) -> "DictRef":
        assert self.in_memory
        return DictRef(
            uid=self.uid,
            obj={k: vref.detached() for k, vref in self.obj.items()},
            in_memory=True,
        )

    ############################################################################
    ### dict interface
    ############################################################################
    def __getitem__(self, key: Union[str, "ValNode", Ref]) -> Union[Ref, "ValNode"]:
        self._auto_attach()
        if isinstance(key, str):
            if (
                GlobalContext.current is not None
                and GlobalContext.current.mode != MODES.run
            ):
                raise ValueError
            return self.obj[key]
        if isinstance(key, Ref):
            prepare_query(ref=key, tp=AnyType())
            res_query = BuiltinQueries.GetDictItemQuery(dct=self.query, key=key.query)
            res = self.obj[key.obj].unlinked(keep_causal=True)
            res._query = res_query
            return res
        elif isinstance(key, ValNode):
            res = BuiltinQueries.GetDictItemQuery(dct=self.query, key=key)
            return res
        else:
            raise ValueError

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class SetRef(StructRef):  # don't subclass from set, because it's not hashable
    """
    Immutable set of Refs.
    """

    builtin_id = "__set__"

    def as_inputs_list(self) -> List[Dict[str, Ref]]:
        wrapped_inputs_list = [{"st": self, "elt": elt} for elt in self.obj]
        return wrapped_inputs_list

    def causify_up(self):
        assert all(elt.causal_uid is not None for elt in self.obj)
        self.causal_uid = Hashing.hash_multiset([elt.causal_uid for elt in self.obj])

    @staticmethod
    def map(obj: set, func: Callable) -> list:
        return [func(elt) for elt in obj]

    @staticmethod
    def elts(obj: set) -> Iterable:
        return iter(obj)

    def dump(self) -> "SetRef":
        assert self.in_memory
        return SetRef(
            uid=self.uid, obj={vref.detached() for vref in self.obj}, in_memory=True
        )

    ############################################################################
    ### set interface
    ############################################################################
    def __contains__(self, item: Ref) -> bool:
        from .wrapping import unwrap

        if not self.in_memory:
            logging.warning(
                "Checking membership in a lazy SetRef requires loading the entire set into memory."
            )
            self._auto_attach(shallow=False)
            return item in unwrap(self.obj)
        else:
            return item in unwrap(self.obj)

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class Builtins:
    IDS = ("__list__", "__dict__", "__set__")

    @staticmethod
    def list_func(lst: List[Any], elt: Any, idx: Any):
        assert lst[idx] is elt

    @staticmethod
    def dict_func(dct: Dict[str, Any], key: str, val: Any):
        assert dct[key] is val

    @staticmethod
    def set_func(st: Set[Any], elt: Any):
        assert elt in st

    list_op = FuncOp(
        func=list_func.__func__, _is_builtin=True, version=0, ui_name="__list__"
    )
    dict_op = FuncOp(
        func=dict_func.__func__, _is_builtin=True, version=0, ui_name="__dict__"
    )
    set_op = FuncOp(
        func=set_func.__func__, _is_builtin=True, version=0, ui_name="__set__"
    )

    OPS = {
        "__list__": list_op,
        "__dict__": dict_op,
        "__set__": set_op,
    }

    REF_CLASSES = {
        "__list__": ListRef,
        "__dict__": DictRef,
        "__set__": SetRef,
    }

    PY_TYPES = {
        "__list__": list,
        "__dict__": dict,
        "__set__": set,
    }

    IO = {
        "construct": {
            "__list__": {"in": {"elt", "idx"}, "out": {"lst"}},
            "__dict__": {"in": {"key", "val"}, "out": {"dct"}},
            "__set__": {"in": {"elt"}, "out": {"st"}},
        },
        "destruct": {
            "__list__": {"in": {"lst", "idx"}, "out": {"elt"}},
            "__dict__": {"in": {"dct", "key"}, "out": {"val"}},
            "__set__": {"in": {"st"}, "out": {"elt"}},
        },
    }

    @staticmethod
    def _make_builtin_uid(uid: str, builtin_id: str) -> str:
        return f"{builtin_id}.{uid}"

    @staticmethod
    def is_builtin_uid(uid: str) -> bool:
        return ("." in uid) and (uid.split(".")[0] in Builtins.IDS)

    @staticmethod
    def parse_builtin_uid(uid: str) -> Tuple[str, str]:
        assert Builtins.is_builtin_uid(uid)
        builtin_id, uid = uid.split(".", 1)
        return builtin_id, uid

    @staticmethod
    def spawn_builtin(
        builtin_id: str, uid: str, causal_uid: Optional[str] = None
    ) -> Ref:
        assert builtin_id in Builtins.IDS
        uid = Builtins._make_builtin_uid(uid=uid, builtin_id=builtin_id)
        res = Builtins.REF_CLASSES[builtin_id](uid=uid, obj=None, in_memory=False)
        if causal_uid is not None:
            res._causal_uid = causal_uid
        return res

    @staticmethod
    def map(
        func: Callable, obj: Union[List, Dict, Set], struct_id: str
    ) -> Union[List, Dict, Set]:
        if struct_id == "__list__":
            return [func(elt) for elt in obj]
        elif struct_id == "__dict__":
            return {key: func(val) for key, val in obj.items()}
        elif struct_id == "__set__":
            return {func(elt) for elt in obj}
        else:
            raise ValueError(f"Invalid struct_id: {struct_id}")

    @staticmethod
    def collect_all_calls(ref: Ref) -> List[Call]:
        if isinstance(ref, ValueRef):
            return []
        elif isinstance(ref, StructRef):
            if not ref.in_memory:
                return []
            else:
                calls = ref.get_calls()
                for elt in ref.elts(ref.obj):
                    calls.extend(Builtins.collect_all_calls(elt))
                return calls
        else:
            raise ValueError(f"Unexpected ref type: {type(ref)}")


from ..queries.weaver import ValNode, BuiltinQueries, prepare_query, StructOrientations
from ..ui.contexts import GlobalContext
from .wrapping import causify_atom
