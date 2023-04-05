from mandala.all import *
from mandala.tests.utils import *
from mandala.queries.weaver import *


@pytest.mark.parametrize("storage", generate_storages())
def test_unit(storage):
    @op
    def f(x: int, y: int) -> int:
        return x + y

    @op
    def mean(nums: List[int]) -> float:
        return sum(nums) / len(nums)

    @op
    def dictmean(nums: Dict[str, Any]) -> float:
        return sum(nums.values()) / len(nums)

    @op
    def repeat(num: int, times: int) -> List[int]:
        return [num for something in range(times)]

    @op
    def make_dict(a: int, b: int) -> Dict[str, int]:
        return {"a": a, "b": b}

    @op
    def swap(x: int, y: int) -> Tuple[int, int]:
        return y, x

    @op
    def concat_lists(a: List[Any], b: List[Any]) -> List[Any]:
        return a + b

    @superop
    def workflow(a: int, b: int, c: int) -> List[Any]:
        dct = {"a": a, "b": b}
        dct_mean = dictmean(nums=dct)
        things = repeat(num=a, times=b)
        things_mean = mean(things)
        new = {"x": dct_mean, "y": things_mean}
        final = dictmean(nums=new)
        x, y = swap(x=final, y=c)
        return [x, y]

    @superop
    def super_workflow(a: int, b: int) -> List[Any]:
        things = workflow(a, a, a)
        other_things = workflow(b, b, b)
        return concat_lists(things, other_things)

    with storage.run():
        a = f(23, 42)
        b = f(4, 8)
        c = f(15, 16)
        avg = mean([a, b, c])
        sames = repeat(num=23, times=5)
        elt = sames[3]
        dict_avg = dictmean({"a": 23, "b": 42})
        dct = make_dict(a=23, b=42)
        z = workflow(a=23, b=42, c=10)
        w = super_workflow(a=a, b=b)


@pytest.mark.parametrize("storage", generate_storages())
def test_mutual_recursion(storage):
    @superop
    def f(x: int) -> int:
        if unwrap(x) == 0:
            return 0
        else:
            return g(unwrap(x) - 1)

    @superop
    def g(x: int) -> int:
        if unwrap(x) == 0:
            return 0
        else:
            return f(unwrap(x) - 1)

    with storage.run():
        a = f(23)
