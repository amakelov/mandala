from mandala_lite.all import *
from mandala_lite.tests.utils import *


### some code is set up at the top level to replicate how the user would use the library
MODULE_PATH = os.path.dirname(__file__)
MODULE_NAME = "mandala_lite.tests.test_deps"
storage = Storage(deps_root=Path(MODULE_PATH))

################################################################################
### the below illustrates different configurations of dependencies
################################################################################
CONST_INT = 23  # a scalar global variable
CONST_DICT = {
    "a": 1,
    "b": 2,
}  # a data structure global variable. The entire structure is tracked as a dependency.
CONST_LIST = [1, 2, 3]
CONST_ARRAY = np.random.uniform(size=(1000,))


def func_1(x):  # uses global variable
    return x + CONST_INT


def func_2(x):  # calls another function, calls function on global variable
    return func_1(x) + func_1(CONST_DICT["a"])


def func_3(x):  # uses a lambda that calls a function, etc.
    f = lambda y: func_2(y) + func_1(CONST_DICT["b"]) + sum(CONST_ARRAY)
    return f(x)


def closure_func(x):  # uses a closure
    closure_const = 42

    def inner(y):  # references local variables
        return x + y + closure_const

    return inner


# a function that references a closure's local variables
f_with_closure_deps = closure_func(23)


def _setup_class_state():  # a function invoked in the __init__ of a class
    return 23


class DepsCls:
    # a class to test dependencies on methods and static/class methods
    def __init__(self, x):  # __init__ calls a method
        self.x = x + _setup_class_state()

    def method_1(self, y):
        return self.x + y

    def method_2(self, y):
        return func_1(self.x) + self.method_1(y)

    @staticmethod
    def static_method_1(y):
        return func_1(y) + CONST_ARRAY

    @classmethod
    def class_method_1(cls, y):
        return cls.static_method_1(y) + cls.static_method_1(y)


class DepsClsChild(DepsCls):
    # a child class to test inheritance and resolving dependencies to the
    # correct class
    def method_1(self, y):  # overrides a method, uses different dependencies
        return func_1(23) + sum(CONST_LIST)


with storage.define():
    # ops with different kinds of dependencies

    @op
    def f_call_func(x: int) -> int:  # calls a func, uses a global variable
        return func_1(x) + CONST_DICT["a"]

    @op
    def f_call_func_with_deps(
        x: int, y: int
    ) -> int:  # calls a func with dependencies of its own
        return func_2(x)

    @op
    def f_make_obj(x: int) -> int:  # creates a class instance, calls a method
        obj = DepsCls(x)
        return obj.method_1(x)

    @op
    def use_closure_implicit(
        x: int,
    ) -> int:  # implicitly has dependencies to closure's local variables
        return f_with_closure_deps(x)

    @op
    def f_make_child_obj(x: int) -> int:  # use a child class
        obj = DepsClsChild(x)
        return obj.method_1(x)


def test_deps_tracking():

    # check dependencies start out empty
    deps = storage.sig_adapter.deps_adapter.load_state()
    for k, v in deps.items():
        assert v.size == 0

    with storage.run():
        f_call_func(23)

    deps = storage.get_deps(f_call_func)
    assert deps.globals_.keys() == {MODULE_NAME}
    assert deps.globals_[MODULE_NAME].keys() == {"CONST_DICT", "CONST_INT"}
    assert deps.sources.keys() == {MODULE_NAME}
    assert deps.sources[MODULE_NAME].keys() == {"func_1", "f_call_func"}

    with storage.run():
        f_call_func_with_deps(23, 42)

    deps = storage.get_deps(f_call_func_with_deps)
    assert deps.globals_.keys() == {MODULE_NAME}
    assert deps.globals_[MODULE_NAME].keys() == {"CONST_DICT", "CONST_INT"}
    assert deps.sources.keys() == {MODULE_NAME}
    assert deps.sources[MODULE_NAME].keys() == {
        "func_1",
        "func_2",
        "f_call_func_with_deps",
    }

    with storage.run():
        f_make_obj(23)

    deps = storage.get_deps(f_make_obj)
    assert deps.num_globals == 0
    assert deps.sources.keys() == {MODULE_NAME}
    assert deps.sources[MODULE_NAME].keys() == {
        "_setup_class_state",
        "DepsCls.__init__",
        "f_make_obj",
        "DepsCls.method_1",
    }

    try:
        with storage.run():
            # this should fail because the closure's local variables cannot be
            # traked by the dependency tracker at import time
            use_closure_implicit(23)
        assert False
    except Exception as e:
        assert True

    with storage.run():
        f_make_child_obj(23)

    deps = storage.get_deps(f_make_child_obj)
    assert deps.globals_.keys() == {MODULE_NAME}
    assert deps.globals_[MODULE_NAME].keys() == {"CONST_LIST", "CONST_INT"}
    assert deps.sources.keys() == {MODULE_NAME}
    assert deps.sources[MODULE_NAME].keys() == {
        "_setup_class_state",
        "DepsClsChild.__init__",
        "f_make_child_obj",
        "DepsClsChild.method_1",
        "func_1",
    }


def test_changes():

    # change the dependencies
    global CONST_INT, CONST_DICT, func_1, func_2, func_3, _setup_class_state, DepsCls, DepsClsChild

    CONST_INT = 24  # change value
    CONST_DICT = {"a": 23, "b": 2}  # change single key only

    def func_1(x):  # source code unchanged, but global variable referenced has changed
        return x + CONST_INT

    def func_2(x):  # change the key of the dict referenced
        return func_1(x) + func_1(CONST_DICT["b"])

    def func_3(x):  # change body of the lambda
        f = lambda y: func_2(y) + func_1(CONST_DICT["b"]) + sum(CONST_ARRAY) + y**2
        return f(x)

    def _setup_class_state():  # change return value
        return 43

    class DepsCls:
        def __init__(self, x):  # REMOVE dependency on _setup_class_state
            self.x = x

        def method_1(self, y):
            return self.x + y

        def method_2(self, y):
            return func_1(self.x) + self.method_1(y)

        @staticmethod
        def static_method_1(y):
            return func_1(y) + CONST_ARRAY

        @classmethod
        def class_method_1(cls, y):
            return cls.static_method_1(y) + cls.static_method_1(y)

    class DepsClsChild(DepsCls):
        # a child class to test inheritance and resolving dependencies to the
        # correct class
        def method_1(self, y):  # overrides a method, uses different dependencies
            return func_1(23) + sum(CONST_LIST)

    # todo: more precise tests with diffs to see if we detect the correct changes
    with storage.define():
        # ops with different kinds of responses to changes

        @op(on_change=OnChange.new_version)
        def f_call_func(x: int) -> int:  # calls a func, uses a global variable
            print("Changed the source code!")
            return func_1(x) + CONST_DICT["a"]

        @op(on_change=OnChange.new_version)
        def f_call_func_with_deps(
            x: int, y: int
        ) -> int:  # calls a func with dependencies of its own
            return func_2(x)

        @op(on_change=OnChange.ignore)
        def f_make_obj(x: int) -> int:  # creates a class instance, calls a method
            obj = DepsCls(x)
            return obj.method_1(x)

        @op(on_change=OnChange.ignore)
        def f_make_child_obj(x: int) -> int:  # use a child class
            obj = DepsClsChild(x)
            return obj.method_1(x)

    assert f_call_func.func_op.sig.version == 1
    assert f_call_func_with_deps.func_op.sig.version == 1
    assert f_make_obj.func_op.sig.version == 0
    assert f_make_child_obj.func_op.sig.version == 0
