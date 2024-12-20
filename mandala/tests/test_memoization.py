from mandala.imports import *
import numpy as np


def test_storage():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1
    
    with storage:
        x = 1
        y = inc(x)
        z = inc(2)
        w = inc(y)
    
    assert w.cid == z.cid
    assert w.hid != y.hid
    assert w.cid != y.cid
    assert storage.unwrap(y) == 2
    assert storage.unwrap(z) == 3
    assert storage.unwrap(w) == 3
    for ref in (y, z, w):
        assert storage.attach(ref).in_memory
        assert storage.attach(ref).obj == storage.unwrap(ref)


def test_signatures():
    storage = Storage()

    @op # a function with a wild input/output signature
    def add(x, *args, y: int = 1, **kwargs):
        # just sum everything
        res = x + sum(args) + y + sum(kwargs.values())
        if kwargs:
            return res, kwargs
        elif args:
            return None
        else:
            return res

    with storage:
        # call the func in all the ways
        sum_1 = add(1)
        sum_2 = add(1, 2, 3, 4, )
        sum_3 = add(1, 2, 3, 4, y=5)
        sum_4 = add(1, 2, 3, 4, y=5, z=6)
        sum_5 = add(1, 2, 3, 4, z=5, w=7)
    
    assert storage.unwrap(sum_1) == 2
    assert storage.unwrap(sum_2) == None
    assert storage.unwrap(sum_3) == None
    assert storage.unwrap(sum_4) == (21, {'z': 6})
    assert storage.unwrap(sum_5) == (23, {'z': 5, 'w': 7})


def test_retracing():
    storage = Storage()

    @op 
    def inc(x):
        return x + 1

    ### iterating a function
    with storage:
        start = 1
        for i in range(10):
            start = inc(start)

    with storage:
        start = 1
        for i in range(10):
            start = inc(start)

    ### composing functions
    @op
    def add(x, y):
        return x + y
    
    with storage:
        inp = [1, 2, 3, 4, 5]
        stage_1 = [inc(x) for x in inp]
        stage_2 = [add(x, y) for x, y in zip(stage_1, stage_1)]
            
    with storage:
        inp = [1, 2, 3, 4, 5]
        stage_1 = [inc(x) for x in inp]
        stage_2 = [add(x, y) for x, y in zip(stage_1, stage_1)]


def test_lists():
    storage = Storage()

    @op
    def get_sum(elts: MList[int]) -> int:
        return sum(elts)
    
    @op
    def primes_below(n: int) -> MList[int]:
        primes = []
        for i in range(2, n):
            for p in primes:
                if i % p == 0:
                    break
            else:
                primes.append(i)
        return primes
    
    @op
    def chunked_square(elts: MList[int]) -> MList[int]:
        # a model for an op that does something on chunks of a big thing
        # to prevent OOM errors
        return [x*x for x in elts]
    
    with storage:
        n = 10
        primes = primes_below(n)
        sum_primes = get_sum(primes)
    assert len(primes) == 4
    # check indexing
    assert storage.unwrap(primes[0]) == 2
    assert storage.unwrap(primes[:2]) == [2, 3]

    ### lists w/ overlapping elements
    with storage:
        n = 100
        primes = primes_below(n)
        for i in range(0, len(primes), 2):
            sum_primes = get_sum(primes[:i+1])
    
    with storage:
        elts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        squares = chunked_square(elts)



def test_ignore():

    storage = Storage()

    @op(ignore_args=('irrelevant',))
    def inc(x, irrelevant):
        return x + 1
    

    with storage:
        inc(23, 0)
    
    df = storage.cf(inc).df()
    assert len(df) == 1

    with storage:
        inc(23, 1)
    
    df = storage.cf(inc).df()
    assert len(df) == 1


def test_clear_uncommitted():
    storage = Storage()

    @op
    def inc(x):
        return x + 1
    
    with storage:
        for i in range(10):
            inc(i)
        # attempt to clear the atoms cache without having committed; this should
        # fail by default
        try:
            storage.atoms.clear()
            assert False
        except ValueError:
            pass
    
        # now clear the atoms cache after committing
        storage.commit()
        storage.atoms.clear()



def test_newargdefault():
    storage = Storage()

    @op
    def add(x,):
        return x + 1
    
    with storage:
        add(1)

    @op
    def add(x, y=NewArgDefault(1)):
        return x + y
    
    with storage:
        add(1)
    # check that we didn't make a new call 
    assert len(storage.cf(add).calls) == 1

    with storage:
        add(1, 1)
    # check that we didn't make a new call 
    assert len(storage.cf(add).calls) == 1

    with storage:
        add(1, 2)
    # now this should have made a new call!
    assert len(storage.cf(add).calls) == 2

def test_newargdefault_compound_types():
    storage = Storage()

    @op
    def add_array(x:np.ndarray):
        return x
    with storage:
        add_array(np.array([1, 2, 3]))
    
    @op
    def add_array(x:np.ndarray, y=NewArgDefault(None)):
        return x + y
    # test passing a raw value
    with storage:
        add_array(np.array([1, 2, 3]), y=np.array([4, 5, 6]))
    
    # now test passing a wrapped value
    with storage:
        add_array(np.array([1, 2, 3]), y=wrap_atom(np.array([7, 8, 9])))




def test_value_pointer():
    storage = Storage()

    @op
    def get_mean(x: np.ndarray) -> float:
        return x.mean()
    
    with storage:
        X = np.array([1, 2, 3, 4, 5])
        X_pointer = ValuePointer("X", X)
        mean = get_mean(X_pointer)
    
    assert storage.unwrap(mean) == 3.0
    df = storage.cf(get_mean).df()
    assert len(df) == 1
    assert df['x'].item().id == "X"
