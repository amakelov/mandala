from ..common_imports import *
import typing

################################################################################
### types
################################################################################
class Type:
    @staticmethod
    def from_annotation(annotation: Any) -> "Type":
        if (annotation is None) or (annotation is inspect._empty):
            return AnyType()
        if annotation is typing.Any:
            return AnyType()
        elif isinstance(annotation, Type):
            return annotation
        elif isinstance(annotation, type):
            if annotation is list:
                return ListType()
            elif annotation is dict:
                return DictType()
            elif annotation is set:
                return SetType()
            else:
                return AnyType()
        elif hasattr(annotation, "__origin__"):
            if annotation.__origin__ is list:
                elt_annotation = annotation.__args__[0]
                return ListType(
                    elt_type=Type.from_annotation(annotation=elt_annotation)
                )
            elif annotation.__origin__ is dict:
                value_annotation = annotation.__args__[1]
                return DictType(
                    elt_type=Type.from_annotation(annotation=value_annotation)
                )
            elif annotation.__origin__ is set:
                elt_annotation = annotation.__args__[0]
                return SetType(elt_type=Type.from_annotation(annotation=elt_annotation))
            elif annotation.__origin__ is tuple:
                return AnyType()
            else:
                return AnyType()
        else:
            return AnyType()

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        if isinstance(self, StructType):
            return self.struct_id == other.struct_id and self.elt_type == other.elt_type
        elif isinstance(self, AnyType):
            return True
        else:
            raise NotImplementedError


class AnyType(Type):
    def __repr__(self):
        return "AnyType()"


class StructType(Type):
    struct_id = None

    def __init__(self, elt_type: Optional[Type] = None):
        self.elt_type = AnyType() if elt_type is None else elt_type


class ListType(StructType):
    struct_id = "__list__"
    model = list

    def __repr__(self):
        return f"ListType(elt_type={self.elt_type})"


class DictType(StructType):
    struct_id = "__dict__"
    model = dict

    def __repr__(self):
        return f"DictType(elt_type={self.elt_type})"


class SetType(StructType):
    struct_id = "__set__"
    model = set

    def __repr__(self):
        return f"SetType(elt_type={self.elt_type})"
