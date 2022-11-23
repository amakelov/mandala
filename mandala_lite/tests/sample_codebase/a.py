### some global constants
CONST_1 = 23
CONST_2 = 42

### elementary functions
def func_inc(x) -> int:
    return x


def func_add(x, y) -> int:
    return x + y


### a class with elementary methods
class SimpleClass:
    def __init__(self, x):
        self.x = x

    def method_inc(self, y):
        return self.x + y

    def method_add(self, y, z):
        return self.x + y + z


class AnotherClass:
    def __init__(self, y):
        self.y = y


### function closures
def closure_func(x):
    LOCAL_CONST = 23

    def inner_func(y):
        return x + y + LOCAL_CONST

    return inner_func(23)
