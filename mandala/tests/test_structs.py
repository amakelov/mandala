from mandala.all import *
from mandala.tests.utils import *
from mandala.core.weaver import *


def test_unit():
    storage = Storage()

    @op
    def repeat(x: int, n_times: int = None) -> List[int]:
        return [x] * n_times

    @op
    def mean(nums: List[float]) -> float:
        return sum(nums) / len(nums)

    with storage.run():
        a = repeat(x=23, n_times=5)
        b = mean(nums=a)
        assert isinstance(a, ListRef)

    with storage.query() as q:
        x = Q()
        a = repeat(x)
        idx = Q()
        elt = BuiltinQueries.GetItemQuery(lst=a, idx=idx)
        df = q.get_table(x, a, elt, idx)


def test_nested():
    storage = Storage()

    @op
    def sum_rows(mat: List[List[float]]) -> List[float]:
        return [sum(row) for row in mat]

    with storage.run():
        mat = [[1, 2, 3], [4, 5, 6]]
        sums = sum_rows(mat=mat)
        mat_2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        sums_2 = sum_rows(mat=mat_2)
        assert sums_2[0].uid == sums[0].uid
        assert sums_2[1].uid == sums[1].uid

    with storage.query() as q:
        x = Q().named("x")
        row = BuiltinQueries.ListQuery(elt=x).named("row")
        mat = BuiltinQueries.ListQuery(elt=row).named("mat")
        row_sums = sum_rows(mat=mat).named("row_sums")
        df = q.get_table(x, row, mat, row_sums)
