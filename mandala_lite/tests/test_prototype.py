from mandala_lite.all import *


def test_func_creation():
    storage = Storage()

    @op(storage=storage)
    def f(x:int) -> int:
        return x + 1