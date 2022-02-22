from .utils import *
from .funcs import *

def test_superops():
    storage = Storage()
    import random

    @op(storage)
    def get_five_nums(seed:int) -> TList[int]:
        return [random.choice(range(100)) for i in range(5)]
    
    @superop(storage)
    def concat_nums(seeds:TList[int]) -> TList[int]:
        nums_list = [get_five_nums(seed) for seed in seeds]
        return [num for nums in nums_list for num in nums]
    
    # confirm mental model of calls that happen
    for num_seeds in (5, 10, 20):
        with run(storage, autocommit=True):
            seeds = list(range(num_seeds))
            res = concat_nums(seeds=seeds)
        
        call_buffer = CallBuffer()
        with capture(storage, captured_calls=call_buffer):
            seeds = list(range(num_seeds))
            res = concat_nums(seeds=seeds)
        expected_num_calls = (
            1 + # construct list entering concat_nums
            num_seeds + # get_five_nums for each
            num_seeds + # deconstructs at exit of get_five_nums
            1 + # construct list exiting concat_nums
            1 # call to concat_nums itself
            ) 
        assert len(call_buffer.unique_calls()) == expected_num_calls
            
    # check the manual capture function
    all_calls = storage.call_st.mget(locs=storage.call_st.locs())
    super_calls = [c for c in all_calls if c.op.is_super]
    with capture(storage):
        captured = capture.capture_calls(func=concat_nums, call=super_calls[0])
