from collections.abc import Sequence, Mapping, Set as SetABC
from ..common_imports import *

from .model import Ref, FuncOp, Call, wrap
from .utils import Hashing


class ListRef(Ref, Sequence):
    """
    Immutable list of Refs.
    """

    def __init__(self, uid: str, obj: Optional[List[Ref]], in_memory: bool):
        assert uid.startswith("__list__.")
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory

    def dump(self) -> "ListRef":
        return ListRef(
            uid=self.uid, obj=[vref.detached() for vref in self.obj], in_memory=True
        )

    ############################################################################
    ### list interface
    ############################################################################
    def __getitem__(self, idx: int) -> Ref:
        self._auto_attach()
        return self.obj[idx]

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class DictRef(Ref, Mapping):
    """
    Immutable string-keyed dict of Refs.
    """

    def __init__(self, uid: str, obj: Optional[Dict[str, Ref]], in_memory: bool):
        assert uid.startswith("__dict__.")
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory

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
    def __getitem__(self, key: str) -> Ref:
        self._auto_attach()
        return self.obj[key]

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class SetRef(Ref, SetABC):
    """
    Immutable set of Refs.
    """

    def __init__(self, uid: str, obj: Optional[Set[Ref]], in_memory: bool):
        assert uid.startswith("__set__.")
        self.uid = uid
        self._obj = obj
        self.in_memory = in_memory

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

    @staticmethod
    def _make_builtin_uid(uid: str, builtin_id: str) -> str:
        return f"{builtin_id}.{uid}"

    @staticmethod
    def is_builtin_uid(uid: str) -> bool:
        return ("." in uid) and (uid.split(".")[0] in Builtins.IDS)

    @staticmethod
    def parse_builtin_uid(uid: str) -> Tuple[str, str]:
        assert Builtins.is_builtin_uid(uid)
        builtin_id, uid = uid.split(".")
        return builtin_id, uid

    @staticmethod
    def spawn_builtin(builtin_id: str, uid: str) -> Ref:
        assert builtin_id in Builtins.IDS
        uid = Builtins._make_builtin_uid(uid=uid, builtin_id=builtin_id)
        return Builtins.REF_CLASSES[builtin_id](uid=uid, obj=None, in_memory=False)

    @staticmethod
    def _make_calls(
        builtin_id: str, wrapped_inputs_list: List[Dict[str, Ref]]
    ) -> List[Call]:
        calls = []
        for wrapped_inputs in wrapped_inputs_list:
            call_uid = Builtins.OPS[builtin_id].get_call_uid(
                wrapped_inputs=wrapped_inputs
            )
            calls.append(
                Call(
                    uid=call_uid,
                    inputs=wrapped_inputs,
                    outputs=[],
                    func_op=Builtins.OPS[builtin_id],
                    transient=False,
                )
            )
        return calls

    @staticmethod
    def construct_list(elts: List[Ref]) -> Tuple[ListRef, List[Call]]:
        uid = Builtins._make_builtin_uid(
            uid=Hashing.hash_list([elt.uid for elt in elts]), builtin_id="__list__"
        )
        lst = ListRef(uid=uid, obj=elts, in_memory=True)
        idxs = [wrap(idx) for idx in range(len(elts))]
        wrapped_inputs_list = [
            {"lst": lst, "elt": elt, "idx": idx} for elt, idx in zip(elts, idxs)
        ]
        return lst, Builtins._make_calls("__list__", wrapped_inputs_list)

    @staticmethod
    def construct_dict(elts: Dict[str, Ref]) -> Tuple[DictRef, List[Call]]:
        uid = Builtins._make_builtin_uid(
            uid=Hashing.hash_dict({k: v.uid for k, v in elts.items()}),
            builtin_id="__dict__",
        )
        dct = DictRef(uid=uid, obj=elts, in_memory=True)
        wrapped_inputs_list = [
            {"dct": dct, "key": wrap(key), "val": val} for key, val in elts.items()
        ]
        return dct, Builtins._make_calls("__dict__", wrapped_inputs_list)

    @staticmethod
    def construct_set(elts: Set[Ref]) -> Tuple[SetRef, List[Call]]:
        uid = Builtins._make_builtin_uid(
            uid=Hashing.hash_set({elt.uid for elt in elts}), builtin_id="__set__"
        )
        st = SetRef(uid=uid, obj=elts, in_memory=True)
        wrapped_inputs_list = [{"st": st, "elt": elt} for elt in elts]
        return st, Builtins._make_calls("__set__", wrapped_inputs_list)
