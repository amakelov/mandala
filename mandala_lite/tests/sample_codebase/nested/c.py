from ..b import *


def compound_func_2(x, y) -> int:
    return compound_func(x, y) + bad_global_var_func(y)
