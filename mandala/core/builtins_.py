from collections.abc import Sequence, Mapping, Set as SetABC
from ..common_imports import *

from .config import MODES
from .model import Ref, FuncOp, Call, wrap_atom, ValueRef
from .utils import Hashing
from .tps import AnyType

HASHERS = {
    "__list__": Hashing.hash_list,
    "__set__": Hashing.hash_set,
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
        self._calls = None
        if self._obj is not None:
            self.set_calls()

    def set_calls(self):
        assert self._calls is None
        self._calls = type(self).make_calls(self)

    def get_calls(self) -> Any:
        if self._calls is None:
            self.set_calls()
        return self._calls

    @staticmethod
    def make_calls(ref: "StructRef") -> Any:
        """
        This step - generating calls based on an existing collection - has the
        benefit of separating the construction of the calls from the
        construction of the Refs, in particular, the assignment of uids to the
        Refs. This allows you to easily swap in a different uid assignment
        strategy.
        """
        raise NotImplementedError

    @staticmethod
    def map(obj: Iterable, func: Callable) -> Iterable:
        raise NotImplementedError

    @staticmethod
    def elts(obj: Iterable) -> Iterable:
        raise NotImplementedError

    def unlinked(self) -> "StructRef":
        if not self.in_memory:
            return type(self)(uid=self.uid, obj=None, in_memory=False)
        else:
            unlinked_elts = type(self).map(
                obj=self.obj, func=lambda elt: elt.unlinked()
            )
            return type(self)(uid=self.uid, obj=unlinked_elts, in_memory=True)


class ListRef(StructRef, Sequence):
    """
    Immutable list of Refs.
    """

    builtin_id = "__list__"

    def get_calls(self) -> List[Call]:
        return super().get_calls()

    @staticmethod
    def make_calls(ref: "ListRef") -> List[Call]:
        res = []
        for i, elt in enumerate(ref.obj):
            wrapped_inputs = {"lst": ref, "elt": elt, "idx": wrap_atom(i)}
            call_uid = Builtins.list_op.get_call_uid(wrapped_inputs=wrapped_inputs)
            res.append(
                Call(
                    uid=call_uid,
                    inputs=wrapped_inputs,
                    outputs=[],
                    func_op=Builtins.list_op,
                    transient=False,
                )
            )
        return res

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
        self, idx: Union[int, "ValQuery", Ref, slice]
    ) -> Union[Ref, "ValQuery"]:
        self._auto_attach()
        if isinstance(idx, Ref):
            prepare_query(ref=idx, tp=AnyType())
            res_query = BuiltinQueries.GetListItemQuery(lst=self.query, idx=idx.query)
            res = self.obj[idx.obj].unlinked()
            res._query = res_query
            return res
        elif isinstance(idx, ValQuery):
            res = BuiltinQueries.GetListItemQuery(lst=self.query, idx=idx)
            return res
        else:
            if (
                GlobalContext.current is not None
                and GlobalContext.current.mode != MODES.run
            ):
                raise ValueError
            return self.obj[idx]

    def __iter__(self):
        self._auto_attach()
        return iter(self.obj)

    def __len__(self) -> int:
        self._auto_attach()
        return len(self.obj)


class DictRef(StructRef, Mapping):
    """
    Immutable string-keyed dict of Refs.
    """

    builtin_id = "__dict__"

    @staticmethod
    def map(obj: dict, func: Callable) -> dict:
        return {k: func(v) for k, v in obj.items()}

    @staticmethod
    def elts(obj: dict) -> Iterable:
        return obj.values()

    def make_calls(self) -> Dict[str, Call]:
        res = {}
        for k, v in self.obj.items():
            wrapped_inputs = {"dct": self, "key": wrap_atom(k), "val": v}
            call_uid = Builtins.dict_op.get_call_uid(wrapped_inputs=wrapped_inputs)
            res[k] = Call(
                uid=call_uid,
                inputs=wrapped_inputs,
                outputs=[],
                func_op=Builtins.dict_op,
                transient=False,
            )
        return res

    def get_calls(self) -> Dict[str, Call]:
        return super().get_calls()

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
    def __getitem__(self, key: Union[str, "ValQuery", Ref]) -> Union[Ref, "ValQuery"]:
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
            res = self.obj[key.obj].unlinked()
            res._query = res_query
            return res
        elif isinstance(key, ValQuery):
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


class SetRef(StructRef, SetABC):
    """
    Immutable set of Refs.
    """

    builtin_id = "__set__"

    @staticmethod
    def map(obj: set, func: Callable) -> list:
        return [func(elt) for elt in obj]

    @staticmethod
    def elts(obj: set) -> Iterable:
        return obj

    def make_calls(self) -> Set[Call]:
        res = set()
        for elt in self.obj:
            wrapped_inputs = {"st": self, "elt": elt}
            call_uid = Builtins.set_op.get_call_uid(wrapped_inputs=wrapped_inputs)
            res.add(
                Call(
                    uid=call_uid,
                    inputs=wrapped_inputs,
                    outputs=[],
                    func_op=Builtins.set_op,
                    transient=False,
                )
            )
        return res

    def get_calls(self) -> Set[Call]:
        return super().get_calls()

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
    def get_inputs_list(ref: Union[ListRef, DictRef, SetRef]) -> List[Dict[str, Ref]]:
        assert ref.in_memory
        if isinstance(ref, ListRef):
            idxs = [wrap_atom(idx) for idx in range(len(ref))]
            wrapped_inputs_list = [
                {"lst": ref, "elt": elt, "idx": idx} for elt, idx in zip(ref, idxs)
            ]
        elif isinstance(ref, DictRef):
            wrapped_inputs_list = [
                {"dct": ref, "key": wrap_atom(k), "val": v} for k, v in ref.items()
            ]
        elif isinstance(ref, SetRef):
            wrapped_inputs_list = [{"st": ref, "elt": elt} for elt in ref]
        else:
            raise ValueError(f"Unexpected ref type: {type(ref)}")
        return wrapped_inputs_list

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
                calls = list(ref.elts(obj=ref.get_calls()))
                for elt in ref.elts(ref.obj):
                    calls.extend(Builtins.collect_all_calls(elt))
                return calls
        else:
            raise ValueError(f"Unexpected ref type: {type(ref)}")


from ..queries.weaver import ValQuery, BuiltinQueries, prepare_query
from ..ui.contexts import GlobalContext
