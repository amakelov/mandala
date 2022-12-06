from .a import *
import numpy as np
from sklearn.datasets import make_classification

# a function that calls a function from a different module
def compound_func(x, y):
    return func_add(x, y) + func_inc(y)


def bad_global_var_func(x):
    return CONST_LIST.append(CONST_1)


def good_global_var_func(x):
    return sum(CONST_LIST) + CONST_1 + x


# a function that calls class methods
def method_caller(x, y, z):
    simple = SimpleClass(x)
    another = AnotherClass(y)
    return simple.method_inc(y) + simple.method_add(y, z)


### two functions that call each other
def recursive_a(x):
    if x in [0, 1]:
        return x
    return recursive_b(x - 2) + 2


def recursive_b(x):
    if x in [0, 1]:
        return x
    return recursive_a(x - 2) + 2


### functions illustrating ast aspects
def types_of_calls():
    # func()
    func_inc(1)
    # obj.method()
    obj = SimpleClass(1)
    obj.method_inc(1)
    # obj.attr.method()
    obj = AnotherClass(1)
    obj.obj.method_inc(1)
    # obj.method().method()
    obj = AnotherClass(1)
    obj.create_obj(1).method_inc(1)
    # obj[key].method()
    objs = [SimpleClass(1), SimpleClass(2)]
    objs[0].method_inc(1)
    # combining two calls via a binop
    obj = AnotherClass(1).create_obj(1).method_inc(10) + SimpleClass(1).method_inc(1)


def from_inside_lambda(x):
    return x


def types_of_dependencies():
    """
    A zoo of the different types of dependencies that can be found in a
    function.
    """
    types_of_calls()
    good_global_var_func(x=23)
    a = CONST_1 + CONST_2
    # b = (lambda a: from_inside_lambda(a))(23)
    print(23)
    import numpy as np

    np.random.uniform()
    make_classification()
