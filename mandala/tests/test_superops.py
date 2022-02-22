from .utils import *
from mandala.adapters.rels import Prov

def test_nesting():
    storage = Storage()
    
    @op(storage)
    def f(x:int, y:int) -> int:
        return x + y 
    
    @op(storage)
    def mean(nums:TList[int]) -> float:
        return sum(nums) / len(nums)
    
    @op(storage)
    def dictmean(nums:TDict[str, TAny]) -> float:
        return sum(nums.values()) / len(nums)
    
    @op(storage)
    def repeat(num:int, times:int) -> TList[int]:
        return [num for _ in range(times)]
    
    @op(storage)
    def make_dict(a:int, b:int) -> TDict[str, int]:
        return {'a': a, 'b': b}
    
    @op(storage)
    def swap(x:int, y:int) -> TTuple[int, int]:
        return y, x
    
    @op(storage)
    def concat_lists(a:TList[TAny], b:TList[TAny]) -> TList[TAny]:
        return a + b 
    
    @superop(storage)
    def workflow(a:int, b:int, c:int) -> TList[TAny]:
        dct = {'a': a, 'b': b}
        dct_mean = dictmean(nums=dct)
        things = repeat(num=a, times=b)
        things_mean = mean(things)
        new = {'x': dct_mean, 'y': things_mean}
        final = dictmean(nums=new)
        return wrap(obj=[final, c], annotation=TList[TAny])
    
    @superop(storage)
    def super_workflow(a:int, b:int) -> TList[TAny]:
        things = workflow(a, a, a)
        other_things = workflow(b, b, b)
        return concat_lists(things, other_things)

    with run(storage, autocommit=False):
        a = f(23, 42)
        b = f(4, 8)
        c = f(15, 16)
        avg = mean([a, b, c])
        sames = repeat(num=23, times=5)
        elt = sames[3]
        dict_avg = dictmean({'a': 23, 'b': 42})
        dct = make_dict(a=23, b=42)
        z = workflow(a=23, b=42, c=10)
        w = super_workflow(a=a, b=b)
    storage.drop_instance_data(answer=True)