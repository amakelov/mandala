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
            else:
                return AnyType()
        elif hasattr(annotation, "__origin__"):
            if annotation.__origin__ is list:
                elt_annotation = annotation.__args__[0]
                return ListType(
                    elt_type=Type.from_annotation(annotation=elt_annotation)
                )
            elif annotation.__origin__ is tuple:
                return AnyType()
            else:
                return AnyType()
        else:
            return AnyType()


class AnyType(Type):
    pass


class ListType(Type):
    def __init__(self, elt_type: Optional[Type] = None):
        self.elt_type = AnyType() if elt_type is None else elt_type
