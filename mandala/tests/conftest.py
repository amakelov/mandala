from .funcs import *
from .utils import *

class ComputationTest(object):
    
    def __init__(self, func:TCallable):
        self.func = func    
        self.test_io:TList[TTuple[TDict[str, TAny], TTuple[TAny,...]]] = []
    
    def add_test(self, inputs:TDict[str, TAny], outputs:TTuple[TAny,...]):
        self.test_io.append((inputs, outputs))

    def run_all(self):
        logging.info(f'Running tests for function {self.func}...')
        for inputs, outputs in self.test_io:
            wrapped_outputs:TTuple[ValueRef,...] = self.func(**inputs)
            actual_outputs = tuple(unwrap(obj) for obj in wrapped_outputs)
            assert eq_objs(x=actual_outputs, y=outputs)


def _setup_tests(default_kv:TType[KVStore], 
                 db_backend:str=CoreConfig.db_backend):

    storage = Storage(call_kv=default_kv, obj_kv=default_kv, 
                      db_backend=db_backend)
    for var in VARS:
        var.synchronize(storage=storage)
    
    for func in OPS:
        storage.synchronize(funcop=func)

    ############################################################################ 
    ### defining computations
    ############################################################################ 
    computation_tests = []

    def single_addition(x, y):
        return tuple([add(x=x, y=y)])
    ct = ComputationTest(func=single_addition)
    ct.add_test(inputs={'x': 23, 'y': 42}, outputs=(65,))
    ct.add_test(inputs={'x': 0, 'y': 0}, outputs=(0,))
    computation_tests.append(ct)
    
    def single_mean(x:TList[int]):
        return tuple([int_mean(x=x)])
    ct = ComputationTest(func=single_mean)
    ct.add_test(inputs={'x': [1, 2, 3]}, outputs=(2.0,))
    computation_tests.append(ct)

    def single_dict_mean(x:TDict[str, TAny]):
        return tuple([dict_mean(x=x)])
    ct = ComputationTest(func=single_dict_mean)
    ct.add_test(inputs={'x': {'a': 23, 'b': 42}}, outputs=(65/2.0,))
    computation_tests.append(ct)

    def single_mean_2d(arr:TList[TList[int]]):
        return tuple([mean_2d(arr=arr)])
    ct = ComputationTest(func=single_mean_2d)
    ct.add_test(inputs={'arr': [[1, 1], [3, 3]]}, outputs=(2.0,))
    computation_tests.append(ct)

    def single_make_frame(columns):
        return (make_frame(columns=columns),)
    ct = ComputationTest(func=single_make_frame)
    cols = {
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    }
    df = pd.DataFrame(cols)
    ct.add_test(inputs={'columns': cols}, outputs=(df,))
    computation_tests.append(ct)

    def prime_factor_mean_over_range(numbers:TList[int]):
        all_factors = []
        for num in numbers:
            all_factors.append(get_prime_factors(x=num))
        result_1 = mean_2d(arr=all_factors)
        result_2 = mean(x=[mean(x=elt) for elt in all_factors])
        assert result_1.obj() == result_2.obj()
        return (result_1, result_2)
    ct = ComputationTest(func=prime_factor_mean_over_range)
    ct.add_test(inputs={'numbers': [4, 5, 6, 7, 8]}, outputs=(3.7, 3.7))
    computation_tests.append(ct)
    
    def inc_dec_prime_factor_pipeline(num:int):
        factors = get_prime_factors(x=num)
        incs = []
        decs = []
        for factor in factors:
            factor_inc, factor_dec = inc_and_dec(x=factor)
            incs.append(factor_inc)
            decs.append(factor_dec)
        incs_mean = int_mean(x=incs)
        decs_mean = int_mean(x=decs)
        return (incs_mean, decs_mean)
    ct = ComputationTest(func=inc_dec_prime_factor_pipeline)
    ct.add_test(inputs={'num': 23}, outputs=(24.0, 22.0))
    ct.add_test(inputs={'num': 100}, outputs=(4.5, 2.5))
    computation_tests.append(ct)

    def superop_workflow(nums:TList[int]):
        divs = get_max_len_divs(nums)
        div_sum = sum_nums(divs)
        final = divisor_prefix(num=div_sum, how_many=2)
        return (divs, div_sum, final)
    ct = ComputationTest(func=superop_workflow)
    ct.add_test(inputs={'nums': list(range(10, 20))}, 
                outputs=([1, 2, 3, 4, 6], 16, [1, 2]))
    computation_tests.append(ct)

    return storage, computation_tests

class SetupConfig(object):
    # storages = [JoblibStorage, SQLiteStorage]
    # storages = [SQLiteStorage]
    storages = [JoblibStorage]
    
class SetupState(object):
    storage:Storage = None
    cts:list = None

# @pytest.fixture(scope='session', params=SetupConfig.storages)
# @pytest.mark.parametrize(argnames=['db_backend'], argvalues=['sqlite',
# 'psql']) doesn't work
# @pytest.fixture(scope='session', params=['sqlite', 'psql'])
@pytest.fixture(scope='session')
def setup_tests(db_backend:str=CoreConfig.db_backend):
    if SetupState.storage is None:
        storage, cts = _setup_tests(default_kv=JoblibStorage,
                                    db_backend=db_backend)
        SetupState.storage = storage
        SetupState.cts = cts
    else:
        storage = SetupState.storage
        cts = SetupState.cts
    yield storage, cts
    # storage.drop(answer=True)
