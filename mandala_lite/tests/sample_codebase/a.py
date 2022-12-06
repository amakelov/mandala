### some global constants
CONST_1 = 23
CONST_2 = 43
CONST_LIST = [1, 2, 3]
CONST_4 = CONST_1

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
        return self.x + y + 1

    def method_add(self, y, z):
        return self.x + y + z


class AnotherClass:
    def __init__(self, y):
        self.y = y
        # useful for creating calls of the form `obj.attr.method()`
        self.obj = SimpleClass(y)

    def create_obj(self, x):
        # useful for creating calls of the form `obj.method().method()`
        return SimpleClass(x)


# a class with a nested class inside
class OuterClass:
    class InnerClass:
        def __init__(self, x):
            self.x = x

        def method_inc(self, y):
            return self.x + y + 1

        def method_add(self, y, z):
            return self.x + y + z


# a function using the inner class
def outer_func(x):
    return OuterClass.InnerClass(x).method_inc(1)


### function closures
def closure_func(x):
    LOCAL_CONST = 23

    def inner_func(y):
        return x + y + LOCAL_CONST

    return inner_func(23)
