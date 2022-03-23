from abc import abstractmethod, ABC

from .config import CoreConfig

from ..common_imports import *
from ..util.common_ut import concat_lists, ContentHashing

HASH_METHODS = ('causal', 'content')
COLLECTION_HASH_METHOD = 'causal'

################################################################################
### bases
################################################################################
class Type(ABC):

    def __init__(self, name:str=None):
        self._name = name

    ###################################### 
    ### small but important
    ###################################### 
    @property
    def name(self) -> TOption[str]:
        return self._name
    
    def set_name(self, name:str):
        # can be set to at most a single value
        if self._name is not None:
            if name != self._name:
                raise ValueError()
        if name is None:
            raise ValueError()
        self._name = name
    
    @property
    def is_named(self) -> bool:
        return self._name is not None
    
    def _reset_name(self):
        self._name = None

    ###################################### 
    ### 
    ###################################### 
    @property
    def annotation(self) -> TAny:
        raise NotImplementedError()
    
    @staticmethod
    def from_tp_or_wrapper(obj:TUnion['Type', 'TypeWrapper']) -> 'Type':
        if isinstance(obj, Type):
            return obj
        elif isinstance(obj, TypeWrapper):
            return obj.tp
        else:
            raise TypeError(f'Got type {type(obj)}')
    
    @staticmethod
    def from_annotation(annotation:TAny) -> 'Type':
        if (annotation is None) or (annotation is inspect._empty):
            return AnyType()
        if annotation == typing.Any:
            return AnyType()
        elif isinstance(annotation, Type):
            return annotation
        elif isinstance(annotation, TypeWrapper):
            return annotation.tp
        elif isinstance(annotation, type):
            if annotation is list:
                return ListType()
            elif annotation is dict:
                return DictType()
            else:
                return AtomType(annotation=annotation)
        elif isinstance(annotation, typing.TypeVar):
            return TypeVar(_id=annotation.__name__, 
                           constraints=annotation.__constraints__)
        elif hasattr(annotation, '__origin__'):
            if annotation.__origin__ is list:
                elt_annotation = annotation.__args__[0]
                return ListType(
                    elt_type=Type.from_annotation(annotation=elt_annotation)
                    )
            elif annotation.__origin__ is dict:
                key_annotation = annotation.__args__[0]
                assert key_annotation is str
                value_annotation = annotation.__args__[1]
                return DictType(
                    value_type=Type.from_annotation(value_annotation)
                    )
            elif annotation.__origin__ is typing.Union:
                return get_union(
                    tps=[Type.from_annotation(x) for x in annotation.__args__]
                    )
            elif annotation.__origin__ is tuple:
                return AtomType(annotation=annotation)
            else:
                raise NotImplementedError(f'Got annotation {annotation}')
        else:
            raise TypeError(f'Got value {annotation}')

    @property
    @abstractmethod
    def is_compound(self) -> bool:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def is_concrete(self) -> bool:
        """
        Whether this type can be the type of a value
        """
        raise NotImplementedError()

    ### storage interface 
    @abstractmethod
    def dump(self) -> TAny:
        """
        Return JSON-serializable unique structural description of this type
        """
        raise NotImplementedError()
    
    @staticmethod
    def load(residue) -> 'Type':
        raise NotImplementedError()

    def __eq__(self, other:'Type') -> bool:
        if not isinstance(other, Type):
            return False
        return self.dump() == other.dump()
    
    def hash(self) -> str:
        return ContentHashing.DEFAULT(self.dump())

    @abstractmethod
    def subtypes(self) -> TIter['Type']:
        raise NotImplementedError()
    
    ### 
    def __repr__(self) -> str:
        data = self.dump()
        data_str = ', '.join([f'{k}={v}' for k, v in data.items()])
        return f'Type({data_str})'
        # return prettify_obj(self.dump())
    
    ############################################################################ 
    ### hashing for values of this type
    ############################################################################ 
    @property
    @abstractmethod
    def hash_method(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def set_hash_method(self, method:str):
        raise NotImplementedError()


class TypeWrapper(ABC):
    # used for interfaces
    
    @property
    @abstractmethod
    def tp(self) -> Type:
        raise NotImplementedError()
    
################################################################################
### implementations
################################################################################
class AnyType(Type):

    def __init__(self):
        self._name = None
        self._hash_method = CoreConfig.default_hash_method

    @property
    def annotation(self) -> TAny:
        return typing.Any

    @property
    def is_compound(self) -> bool:
        return False
    
    @property
    def is_concrete(self) -> bool:
        return True
    
    def dump(self) -> TAny:
        return {'type': type(self).__name__}
    
    def subtypes(self) -> TIter['Type']:
        return [self]
    
    @property
    def hash_method(self) -> str:
        if hasattr(self, '_hash_method'):
            return self._hash_method
        else:
            return CoreConfig.default_hash_method

    def set_hash_method(self, method:str):
        self._hash_method = method

    def __repr__(self) -> str:
        return 'AnyType()'


class AtomType(Type):

    def __init__(self, annotation:TAny):
        self._name = None
        self._annotation = annotation
        self._hash_method = CoreConfig.default_hash_method
    
    @property
    def hash_method(self) -> str:
        if hasattr(self, '_hash_method'):
            return self._hash_method
        else:
            return CoreConfig.default_hash_method

    def set_hash_method(self, method:str):
        self._hash_method = method

    @property
    def annotation(self) -> TAny:
        return self._annotation

    @property
    def is_compound(self) -> bool:
        return False
    
    @property
    def is_concrete(self) -> bool:
        return True
    
    def dump(self) -> TAny:
        return {'type': type(self).__name__, 
                'annotation': str(self.annotation),
                'name': self.name
                }

    def subtypes(self) -> TIter['Type']:
        return [self]
    
    def __repr__(self) -> str:
        if isinstance(self.annotation, type):
            anno_repr = self.annotation.__name__
        else:
            anno_repr = self.annotation
        if self.name is None:
            return f'AtomType({anno_repr})'
        else:
            return f'AtomType({anno_repr}, name={self.name})'


class ListType(Type):

    def __init__(self, elt_type:Type=None):
        self._name = None
        self._ui_name = 'list'
        self._elt_type = AnyType() if elt_type is None else elt_type
        self._hash_method = None
    
    @property
    def hash_method(self) -> str:
        return COLLECTION_HASH_METHOD

    def set_hash_method(self, method:str):
        assert method == COLLECTION_HASH_METHOD

    @property
    def annotation(self) -> TAny:
        return typing.List[self.elt_type.annotation]

    @property
    def elt_type(self) -> Type:
        return self._elt_type
    
    @property
    def is_compound(self) -> bool:
        return True
    
    @property
    def is_concrete(self) -> bool:
        return self.elt_type.is_concrete

    def dump(self) -> TAny:
        return {'type': type(self).__name__, 
                'name': self.name,
                'elt_type': self.elt_type.dump()}

    def subtypes(self) -> TIter['Type']:
        return [self] + list(self.elt_type.subtypes())

    def __repr__(self) -> str:
        return f'ListType(elt_type={self.elt_type}, name={self.name})'


class DictType(Type):
    
    def __init__(self, value_type:Type=None) -> None:
        self._name = None
        self._ui_name = 'dict'
        self._value_type = AnyType() if value_type is None else value_type
        self._hash_method = None
    
    @property
    def hash_method(self) -> str:
        return COLLECTION_HASH_METHOD

    def set_hash_method(self, method:str):
        assert method == COLLECTION_HASH_METHOD

    @property
    def value_type(self) -> Type:
        return self._value_type
    
    @property
    def annotation(self) -> TAny:
        return typing.Dict[str, self.value_type.annotation]
    
    @property
    def is_compound(self) -> bool:
        return True
    
    @property
    def is_concrete(self) -> bool:
        return self.value_type.is_concrete

    def dump(self) -> TAny:
        return {'type': type(self).__name__, 
                'name': self.name,
                'value_type': self.value_type.dump()}

    def subtypes(self) -> TIter['Type']:
        return [self] + list(self.value_type.subtypes())

    def __repr__(self) -> str:
        return f'DictType(value_type={self.value_type}, name={self.name})'

################################################################################
### type unions
################################################################################
class UnionType(Type):
    """
    Object representing a non-trivial union of types.
    """
    def __init__(self, operands:TList[Type]=None) -> None:
        """
        Inductive invariant: 

        Makes sure that the invariants of self.operands are satisfied, assuming
        any union types in the input already satisfy the invariants.
        """
        self._name = None
        operands = [] if operands is None else operands
        expanded_operands = []
        for op in operands:
            if isinstance(op, UnionType):
                expanded_operands += op.operands
            else:
                expanded_operands.append(op)
        unique_operands = remove_duplicates(tps=expanded_operands)
        if not len(unique_operands) > 1:
            raise ValueError('Cannot form union of fewer than 2 types')
        self._operands = unique_operands
    
    @property
    def hash_method(self) -> str:
        return COLLECTION_HASH_METHOD

    def set_hash_method(self, method:str):
        assert method == COLLECTION_HASH_METHOD

    @property
    def annotation(self) -> TAny:
        things = tuple(elt.annotation for elt in self.operands)
        return typing.Union[things]
    
    @property
    def operands(self) -> TList[Type]:
        """
        Invariants:
            - no two types are the same
            - no types are union types themselves
        """
        return self._operands
    
    def associated(self) -> 'UnionType':
        assoc_ops = []
        for op in self.operands:
            if isinstance(op, UnionType):
                assoc_ops += op.associated().operands
            else:
                assoc_ops.append(op)
        return UnionType(operands=assoc_ops)
    
    @property
    def is_compound(self) -> bool:
        raise NotImplementedError()
    
    @property
    def is_concrete(self) -> bool:
        return False
    
    def dump(self) -> TAny:
        return {'type': type(self).__name__, 
                'name': self.name,
                'operands': [op.dump() for op in self.operands]}
    
    def subtypes(self) -> TIter['Type']:
        return concat_lists([list(op.subtypes()) for op in self.operands])

    def __repr__(self) -> str:
        return f'UnionType(operands=[{", ".join([repr(elt) for elt in self.operands])}])'


def remove_duplicates(tps:TList[Type]) -> TList[Type]:
    df = pd.DataFrame({'tp': tps, 'hash': [tp.hash() for tp in tps]})
    return df.groupby('hash').first()['tp'].values.tolist()

def get_union(tps:TList[Type]) -> Type:
    """
    Use to avoid cases when operands reduce to 0 or 1 unique types
    """
    if not tps:
        logging.warning('Type union of empty collection')
        return AnyType()
    dedup = remove_duplicates(tps)
    if len(dedup) == 1:
        return dedup[0]
    else:
        return UnionType(operands=dedup)

################################################################################
### builtins
################################################################################
class BuiltinTypes(object): # todo - this only exists for backward compat
    INT_NAME = '__int__'
    STR_NAME = '__str__'
    
    @staticmethod
    def get(py_type:type) -> Type:
        if py_type is int:
            res = AtomType(annotation=int)
            res.set_name(name=BuiltinTypes.INT_NAME)
            return res
        elif py_type is str:
            res = AtomType(annotation=str)
            res.set_name(BuiltinTypes.STR_NAME)
            return res
        else:
            raise NotImplementedError()

################################################################################
### type utils
################################################################################
def is_subtype(s:Type, t:Type) -> bool:
    """
    Return True if and only if any value that is an instance of s is also
    an instance of t.
    """
    if isinstance(t, AnyType):
        res = True
    elif isinstance(t, AtomType):
        if not isinstance(s, AtomType):
            res = False
        else:
            # importantly, *both* name and annotation must agree
            res = (s.name == t.name) and (s.annotation is t.annotation)
    elif isinstance(t, ListType):
        if not isinstance(s, ListType):
            res = False
        else:
            # immutable lists are covariant
            res = is_subtype(s.elt_type, t.elt_type)
    elif isinstance(t, DictType):
        if not isinstance(s, DictType):
            res = False
        else:
            # immutable mapping are covariant
            res = is_subtype(s=s.value_type, t=t.value_type)
    elif isinstance(t, UnionType):
        if isinstance(s, UnionType):
            res = all(is_subtype(s_operand, t) for s_operand in s.operands)
        else:
            res = any(is_subtype(s, t_operand) for t_operand in t.operands)
    else:
        raise NotImplementedError()
    return res

def is_member(s:Type, t:Type) -> bool:
    """
    Checks whether a concrete type precisely matches a member type in a (not
    necessarily concrete) target type.
    """
    assert s.is_concrete
    if isinstance(t, AnyType):
        return isinstance(s, AnyType)
    elif isinstance(t, AtomType):
        if not isinstance(s, AtomType):
            return False
        return (s.name == t.name)
    elif isinstance(t, ListType):
        if not isinstance(s, ListType):
            return False
        return is_member(s.elt_type, t.elt_type)
    elif isinstance(t, DictType):
        if not isinstance(s, DictType):
            return False
        return is_member(s=s.value_type, t=t.value_type)
    elif isinstance(t, UnionType):
        return any(is_member(s, op) for op in t.operands)
    else:
        raise NotImplementedError()
    
def intersection(u:Type, v:Type) -> TOption[Type]:
    """
    Return the type of values belonging to both u and v, if there exist such
    values. 
    """
    ### deal with Any's
    if isinstance(u, AnyType):
        return v
    if isinstance(v, AnyType):
        return u
    ### deal with v being a union recursively
    if isinstance(v, UnionType):
        op_intersections = [intersection(u, operand) for operand in v.operands]
        nonempty_intersections = [x for x in op_intersections if x is not None]
        if nonempty_intersections:
            return get_union(nonempty_intersections)
        else:
            return 
    ### cases for u 
    if isinstance(u, AtomType):
        if isinstance(v, AtomType) and v == u:
            return u
    elif isinstance(u, ListType):
        if isinstance(v, ListType):
            return intersection(u.elt_type, v.elt_type)
    elif isinstance(u, DictType):
        if isinstance(v, DictType):
            return intersection(u.value_type, v.value_type)
    elif isinstance(u, UnionType):
        return intersection(v, u)

def isinstance_annotation(obj:TAny, annotation:TAny) -> bool:
    """
    Check if an object matches a type annotation. 
    """
    if isinstance(annotation, type):
        return isinstance(obj, annotation)
    if hasattr(annotation, '__origin__'):
        if annotation.__origin__ is list:
            elt_annotation = annotation.__args__[0]
            return (isinstance(obj, list) and 
                    all(isinstance_annotation(elt, elt_annotation) 
                        for elt in obj))
        elif annotation.__origin__ is dict:
            key_annotation, value_annotation = annotation.__args__
            return (isinstance(obj, dict) and 
                    all(isinstance_annotation(k, key_annotation)
                        for k in obj.keys()) and
                    all(isinstance_annotation(v, value_annotation)
                        for v in obj.values()))
        elif annotation.__origin__ is tuple:
            elt_annotations = annotation.__args__
            if len(elt_annotations) == 2 and elt_annotations[1] == Ellipsis:
                elt_annotation = annotation.__args__[0]
                return (isinstance(obj, tuple) and
                        all(isinstance_annotation(elt, elt_annotation)
                            for elt in obj))
            else:
                return (isinstance(obj, tuple) and
                        all(isinstance_annotation(elt, elt_annotation) 
                            for elt, elt_annotation in 
                            zip(obj, annotation.__args__)))
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

################################################################################
### future
################################################################################
class TypeVar(Type):

    def __init__(self, _id:str, constraints):
        self._id, self._constraints = _id, constraints
        self._name = None

    @property
    def annotation(self) -> TAny:
        return typing.TypeVar(self._id, self._constraints)
    
    @property
    def constraints(self) -> TAny:
        return self._constraints

    @property
    def is_compound(self) -> bool:
        raise NotImplementedError()
    
    @property
    def is_concrete(self) -> bool:
        return False
    
    def dump(self) -> TAny:
        return {'type': type(self).__name__,
                'annotation': str(self.annotation),
                'name': self.name}
    
    def subtypes(self) -> TIter['Type']:
        return [self]


is_subtype(UnionType([AnyType(), ListType(AnyType())]), 
           UnionType([AnyType(), ListType(AnyType())]), )

a = UnionType([AnyType(), ListType(AnyType())]) 
