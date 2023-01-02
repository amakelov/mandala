from mandala.all import *
from mandala.tests.utils import *
from mandala.core.weaver import *


def test_unit():
    storage = Storage()

    ### lists
    @op
    def repeat(x: int, times: int = None) -> List[int]:
        return [x] * times

    @op
    def get_list_mean(nums: List[float]) -> float:
        return sum(nums) / len(nums)

    with storage.run():
        lst = repeat(x=42, times=23)
        x = get_list_mean(nums=lst)
        y = get_list_mean(nums=lst[:10])
        assert unwrap(lst[1]) == 42
        assert len(lst) == 23
    storage.rel_adapter.obj_get(uid=lst.uid)

    with storage.query() as q:
        x = Q().named("x")
        lst = repeat(x).named("lst")
        idx = Q().named("idx")
        elt = BuiltinQueries.GetListItemQuery(lst=lst, idx=idx).named("elt")
        df = q.get_table(x, lst, elt, idx)
    assert df.shape == (23, 4)
    assert all(df["elt"] == 42)
    assert sorted(df["idx"]) == list(range(23))

    ### dicts
    @op
    def get_dict_mean(nums: Dict[str, float]) -> float:
        return sum(nums.values()) / len(nums)

    @op
    def describe_sequence(seq: List[int]) -> Dict[str, float]:
        return {
            "min": min(seq),
            "max": max(seq),
            "mean": sum(seq) / len(seq),
        }

    with storage.run():
        dct = describe_sequence(seq=[1, 2, 3])
        dct_mean = get_dict_mean(nums=dct)
        assert unwrap(dct_mean) == 2.0
        assert unwrap(dct["min"]) == 1
        assert len(dct) == 3
    storage.rel_adapter.obj_get(uid=dct.uid)

    with storage.query() as q:
        seq = Q().named("seq")
        dct = describe_sequence(seq=seq).named("dct")
        dct_mean = get_dict_mean(nums=dct).named("dct_mean")
        df = q.get_table(seq, dct, dct_mean)

    ### sets
    @op
    def mean_set(nums: Set[float]) -> float:
        return sum(nums) / len(nums)

    @op
    def get_prime_factors(num: int) -> Set[int]:
        factors = set()
        for i in range(2, num):
            while num % i == 0:
                factors.add(i)
                num /= i
        return factors

    with storage.run():
        factors = get_prime_factors(num=42)
        factors_mean = mean_set(nums=factors)
        assert unwrap(factors_mean) == 4.0
        assert len(factors) == 3
    storage.rel_adapter.obj_get(uid=factors.uid)

    with storage.query() as q:
        num = Q().named("num")
        factors = BuiltinQueries.SetQuery(elt=num).named("factors")
        factors_mean = mean_set(nums=factors).named("factors_mean")
        df = q.get_table(num, factors, factors_mean)


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
