from .a import *

# a function that calls a function from a different module
def compound_func(x, y):
    return func_add(x, y) + func_inc(y)


def global_var_func(x):
    return CONST_1 + x


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
