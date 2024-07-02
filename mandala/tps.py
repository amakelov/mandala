from .common_imports import *
import typing
from typing import Hashable

################################################################################
### types
################################################################################
from typing import Generic

T = TypeVar("T")
# Subclassing List
class MList(List[T], Generic[T]):
    def identify(self):
        return "Type annotation for `mandala` lists"


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
class MDict(Dict[_KT, _VT], Generic[_KT, _VT]):
    def identify(self):
        return "Type annotation for `mandala` dictionaries"


class MSet(Set[T], Generic[T]):
    def identify(self):
        return "Type annotation for `mandala` sets"


class MTuple(Tuple, Generic[T]):
    def identify(self):
        return "Type annotation for `mandala` tuples"


class Type:
    @staticmethod
    def from_annotation(annotation: Any) -> "Type":
        if (annotation is None) or (annotation is inspect._empty):
            return AtomType()
        elif annotation is typing.Any:
            return AtomType()
        elif hasattr(annotation, "__origin__"):
            if annotation.__origin__ is MList:
                elt_annotation = annotation.__args__[0]
                return ListType(elt=Type.from_annotation(annotation=elt_annotation))
            elif annotation.__origin__ is MDict:
                key_annotation = annotation.__args__[0]
                value_annotation = annotation.__args__[1]
                return DictType(
                    key=Type.from_annotation(annotation=key_annotation),
                    val=Type.from_annotation(annotation=value_annotation),
                )
            elif annotation.__origin__ is MSet:
                elt_annotation = annotation.__args__[0]
                return SetType(elt=Type.from_annotation(annotation=elt_annotation))
            elif annotation.__origin__ is MTuple:
                if len(annotation.__args__) == 2 and annotation.__args__[1] == Ellipsis:
                    return TupleType(
                        Type.from_annotation(annotation=annotation.__args__[0])
                    )
                else:
                    return TupleType(
                        *(
                            Type.from_annotation(annotation=elt_annotation)
                            for elt_annotation in annotation.__args__
                        )
                    )
            else:
                return AtomType()
        elif isinstance(annotation, Type):
            return annotation
        else:
            return AtomType()

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        elif isinstance(self, AtomType):
            return True
        else:
            raise NotImplementedError


class AtomType(Type):
    def __repr__(self):
        return "AnyType()"


class ListType(Type):
    struct_id = "__list__"
    model = list

    def __init__(self, elt: Type):
        self.elt = elt

    def __repr__(self):
        return f"ListType(elt_type={self.elt})"


class DictType(Type):
    struct_id = "__dict__"
    model = dict

    def __init__(self, val: Type, key: Type = None):
        self.key = key
        self.val = val

    def __repr__(self):
        return f"DictType(val_type={self.val})"


class SetType(Type):
    struct_id = "__set__"
    model = set

    def __init__(self, elt: Type):
        self.elt = elt

    def __repr__(self):
        return f"SetType(elt_type={self.elt})"


class TupleType(Type):
    struct_id = "__tuple__"
    model = tuple

    def __init__(self, *elt_types: Type):
        self.elt_types = elt_types

    def __repr__(self):
        return f"TupleType(elt_types={self.elt_types})"
