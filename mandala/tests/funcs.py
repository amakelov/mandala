from .utils import *

Any = Var(annotation=typing.Any)
AnyList = Var(annotation=list)
Int = Var(annotation=int)
Float = Var(annotation=float)
Str = Var(annotation=str)
Array = Var(annotation=np.ndarray, name='Array')
ArrayList = Var(annotation=TList[Array], name='ArrayList')
FloatDict = Var(annotation=TDict[str, float])
IntList = Var(annotation=TList[int])
AnyDict = Var(annotation=TDict[str, TAny])
IntMatrix = Var(annotation=TList[TList[int]])
DictOfIntLists = Var(annotation=TDict[str, TList[int]])
Frame = Var(annotation=pd.DataFrame)


VARS = (
    Any, AnyList, Int, Float, Str, Array, ArrayList,
    FloatDict, IntList, AnyDict, IntMatrix, DictOfIntLists, Frame
)


################################################################################ 
### ops
################################################################################ 
@op()
def inc(x:Int) -> Int:
    return x + 1 

@op()
def add(x:Int, y:Int) -> Int:
    return x + y 

@op()
def mean(x:AnyList) -> Any:
    return sum(x) / len(x)

@op()
def add_int(x:Int, y:Int) -> Int:
    return x + y 

@superop()
def add_three(x:Int, y:Int, z:Int) -> Int:
    intermediate = add_int(x=x, y=y)
    return add_int(intermediate, z)

@op()
def int_mean(x:IntList) -> Float:
    return sum(x) / len(x)

@op()
def dict_mean(x:AnyDict) -> Any:
    return sum(x.values()) / len(x)

@op()
def get_prime_factors(x:Int) -> IntList:
    if x < 2:
        return []
    nums = list(range(2, x + 1))
    primes = [a for a in nums if x % a ==0
              and all([a % div != 0 for div in nums if 1 < div and div < a])]
    return primes

@op()
def mean_2d(arr:IntMatrix) -> Float:
    means = [sum(x) / len(x) for x in arr]
    return sum(means) / len(means)

@op()
def make_frame(columns:DictOfIntLists) -> Frame:
    return pd.DataFrame(columns)

### an operation with multiple outputs
@op()
def inc_and_dec(x:Int) -> TTuple[Int, Int]:
    return x + 1, x - 1

### an operation with no outputs
@op()
def log_some_things(x:Int, y:FloatDict, z:DictOfIntLists):
    return

### an operation with dict outputs
@op()
def get_some_metrics(x:Int, y:IntList) -> FloatDict:
    res = {
        'a': 0.3,
        'b': len(y) / 10
    }
    return res

################################################################################ 
### superops, unnamed types
################################################################################ 
@op()
def get_divisors(num:int) -> TList[int]:
    return [x for x in range(1, num) if num % x == 0]

@op()
def sum_nums(nums:TList[int]) -> int:
    return sum(nums)
    
@superop()
def get_max_len_divs(nums:TList[int]) -> TList[int]:
    # return the divisors of the number with the most divisors among `nums`
    all_divs = [get_divisors(num) for num in nums]
    lengths = [len(x) for x in all_divs]
    i = np.argmax(lengths)
    return all_divs[i]

@superop()
def divisor_prefix(num:int, how_many:int) -> TList[int]:
    # return a prefix of the number's divisors of the given length
    divisors = get_divisors(num)
    return divisors[:unwrap(how_many)]

### 
OPS = (
    inc, add, mean, add_int, add_three, int_mean, dict_mean, get_prime_factors,
    mean_2d, make_frame, inc_and_dec, log_some_things, get_some_metrics,
    get_divisors, sum_nums, get_max_len_divs, divisor_prefix,
)
