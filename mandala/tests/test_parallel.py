from .conftest import *
from .utils import *
from mandala.core.config import EnvConfig
if EnvConfig.has_ray:
    import ray
if EnvConfig.has_dask:
    import dask

def get_ray_tests():
    ray_computation_tests = []
    def prime_factor_mean_over_range(numbers:TList[int]):
        all_factors = []
        for num in numbers:
            all_factors.append(get_prime_factors.remote(x=num))
        all_factors = ray.get(all_factors)
        result_1 = mean_2d.remote(arr=all_factors)
        intermediate_means = [mean.remote(x=elt) for elt in all_factors]
        intermediate_means = ray.get(intermediate_means)
        result_2 = int_mean.remote(x=intermediate_means)
        result_1, result_2 = ray.get(result_1), ray.get(result_2)
        assert result_1.obj() == result_2.obj()
        return (result_1, result_2)
    ct = ComputationTest(func=prime_factor_mean_over_range)
    ct.add_test(inputs={'numbers': [4, 5, 6, 7, 8]}, outputs=(3.7, 3.7))
    ray_computation_tests.append(ct)
    def chaining_futures(numbers:TList[int]):
        plus_1 = [inc.remote(x) for x in numbers]
        plus_2 = [inc.remote(x) for x in plus_1]
        results = ray.get(plus_2)
        return (results,)
    ct = ComputationTest(func=chaining_futures)
    ct.add_test(inputs={'numbers': [1, 2, 3]}, outputs=([3, 4, 5],))
    ray_computation_tests.append(ct)
    return ray_computation_tests
    
def get_dask_tests():
    """
    Currently, dask only works correctly when calls to compute are made in the
    same context that defined the values.
    """
    inc_dask = dask.delayed(inc)
    mean_dask = dask.delayed(mean)
    int_mean_dask = dask.delayed(int_mean)
    get_prime_factors_dask = dask.delayed(get_prime_factors)
    mean_2d_dask = dask.delayed(mean_2d)

    dask_computation_tests = []

    def parallel_inc(nums:TList[int]):
        futures = [inc_dask(num) for num in nums]
        results = dask.compute(futures)[0]
        return (results,)
    ct = ComputationTest(func=parallel_inc)
    ct.add_test(inputs={'nums': [1, 2, 3, 4, 5]}, outputs=([2, 3, 4, 5, 6],))
    dask_computation_tests.append(ct)

    def prime_factor_mean_over_range(numbers:TList[int]):
        all_factors = []
        for num in numbers:
            all_factors.append(get_prime_factors_dask(x=num))
        result_1 = mean_2d_dask(arr=all_factors)
        result_2 = int_mean_dask(x=[mean_dask(x=elt) for elt in all_factors])
        result_1 = dask.compute(result_1)[0]
        result_2 = dask.compute(result_2)[0]
        assert result_1.obj() == result_2.obj()
        return (result_1, result_2)
    ct = ComputationTest(func=prime_factor_mean_over_range)
    ct.add_test(inputs={'numbers': [4, 5, 6, 7, 8]}, outputs=(3.7, 3.7))
    dask_computation_tests.append(ct)

    return dask_computation_tests

if EnvConfig.has_dask:
    def test_dask(setup_tests):
        storage, _ = setup_tests
        storage:Storage
        dask_cts = get_dask_tests()
        with run(storage=storage, partition='test') as c:
            for ct in dask_cts:
                ct.run_all()
                c.commit()
                storage.verify_static()
        storage.drop_instance_data(answer=True)

if EnvConfig.has_ray:
    def test_ray(setup_tests):
        storage, _ = setup_tests
        storage:Storage
        ray_cts = get_ray_tests()
        ray.init(ignore_reinit_error=True)

        with run(storage=storage, partition='test') as c:
            for ct in ray_cts:
                ct.run_all()
                c.commit()
                storage.verify_static()
        storage.drop_instance_data(answer=True)

    def test_ray_batching(setup_tests):
        """
        An epic integration test for ray, superops, queries, call checking and
        context passing.
        """
        storage, _ = setup_tests
        storage:Storage
        ray_cts = get_ray_tests()

        with context(storage=storage):
            @op()
            def new_inc(x:Int) -> Int:
                return x + 1 
            
            @superop()
            def inc_batch(elts:IntList, __context__=None):
                for elt in elts:
                    if not new_inc.is_recoverable(x=elt, __context__=__context__):
                        result = elt.unwrap() + 1 
                        new_inc(x=elt, __returns__=result, __context__=__context__)
        
        with run(storage=storage, partition='test') as c:
            c.init_ray()
            __context__ = c.spawn()
            batches = [list(range(10*i, 10*(i+1))) for i in range(10)]
            futures = []
            for batch in batches:
                future = inc_batch.remote(elts=batch, __context__=__context__)
                futures.append(future)
            ray.get(futures)
            c.commit()
        
        with query(storage=storage) as q:
            x = Int()
            incremented = new_inc(x=x)
            df = q.qeval(x, incremented, names=['x', 'incremented'])
            expected_tuples = {(i, i+1) for i in range(100)}
            assert set((x, y) for x, y in df.itertuples(index=False)) == expected_tuples
        storage.drop_instance_data(answer=True)