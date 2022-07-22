from core import *


def test():
    storage = Storage(root=Path("./test_storage/"))

    @op(storage=storage)
    def f(x: int) -> int:
        return x + 1
