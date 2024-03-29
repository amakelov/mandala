from mandala.all import *
from mandala.tests.utils import *
from mandala.queries.weaver import *


@pytest.mark.parametrize("storage", generate_storages())
def test_unit(storage):
    Config.query_engine = "_test"

    ### lists
    @op
    def repeat(x: int, times: int = None) -> List[int]:
        return [x] * times

    @op
    def get_list_mean(nums: List[float]) -> float:
        return sum(nums) / len(nums)

    with storage.run():
        lst = repeat(x=42, times=23)
        a = lst[0]
        x = get_list_mean(nums=lst)
        y = get_list_mean(nums=lst[:10])
        assert unwrap(lst[1]) == 42
        assert len(lst) == 23
    storage.rel_adapter.obj_get(uid=lst.uid)

    with storage.query():
        x = Q().named("x")
        lst = repeat(x).named("lst")
        idx = Q().named("idx")
        elt = BuiltinQueries.GetListItemQuery(lst=lst, idx=idx).named("elt")
        df = storage.df(x, lst, elt, idx)
    assert df.shape == (23, 4)
    assert all(df["elt"] == 42)
    assert sorted(df["idx"]) == list(range(23))

    # test list constructor
    with storage.query():
        # a query for all the lists whose mean we've taken
        lst = BuiltinQueries.ListQ(elts=[Q()]).named("lst")
        x = get_list_mean(nums=lst).named("x")
        df = storage.df(lst, x)

    # test syntax sugar
    with storage.query():
        x = Q().named("x")
        lst = repeat(x).named("lst")
        first_elt = lst[Q()].named("first_elt")
        df = storage.df(x, lst, first_elt)

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
        dct_mean_2 = get_dict_mean(nums={"a": dct["min"]})
        assert unwrap(dct_mean) == 2.0
        assert unwrap(dct["min"]) == 1
        assert len(dct) == 3
    storage.rel_adapter.obj_get(uid=dct.uid)

    with storage.query():
        seq = Q().named("seq")
        dct = describe_sequence(seq=seq).named("dct")
        dct_mean = get_dict_mean(nums=dct).named("dct_mean")
        df = storage.df(seq, dct, dct_mean)

    # test dict constructor
    with storage.query():
        # a query for all the dicts whose mean we've taken
        dct = BuiltinQueries.DictQ(dct={Q(): Q()}).named("dct")
        dct_mean = get_dict_mean(nums=dct).named("dct_mean")
        df = storage.df(dct, dct_mean)

    # test syntax sugar
    # with storage.query():
    #     seq = Q().named("seq")
    #     dct = describe_sequence(seq=seq).named("dct")
    #     dct_mean = dct["mean"].named("dct_mean")
    #     df = storage.df(seq, dct, dct_mean)

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

    with storage.query():
        num = Q().named("num")
        factors = BuiltinQueries.SetQ(elts={num}).named("factors")
        factors_mean = mean_set(nums=factors).named("factors_mean")
        df = storage.df(num, factors, factors_mean)


@pytest.mark.parametrize("storage", generate_storages())
def test_nested(storage):
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

    with storage.query():
        x = Q().named("x")
        row = BuiltinQueries.ListQ(elts=[x]).named("row")
        mat = BuiltinQueries.ListQ(elts=[row]).named("mat")
        row_sums = sum_rows(mat=mat).named("row_sums")
        df = storage.df(x, row, mat, row_sums)
