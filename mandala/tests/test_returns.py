from .utils import *
from .funcs import *
from .conftest import _setup_tests, setup_tests

def test_simple():
    storage = Storage()
    
    ### do some work
    with run(storage, partition='temp') as c:
        @op()
        def f(x:int, y:int) -> TTuple[int, int]:
            return x + 23, y + 42
        for i, j in itertools.product(range(3), range(5)):
            z = f(x=i, y=j)
        c.commit()
    
    ### migrate things
    with run(storage, partition='temp') as c:
        @op()
        def g(x:int, y:int) -> TList[int]:
            return [x + 23, y + 42]
        with c(mode=MODES.query) as qc:
            x = Query(int)
            y = Query(int)
            z, w = f(x=x, y=y)
            df = qc.qget(x, y, z, w)
        for x, y, z, w in df.itertuples(index=False):
            new_output = [z, w]
            g(x=x, y=y, __returns__=new_output)
        c.commit()
    
    ### check that it worked and preserved UIDs of the things we moved around
    with query(storage) as c:
        x = Query(int).named('x')
        y = Query(int).named('y')
        z, w = f(x=x, y=y)
        z.named('z')
        w.named('w')
        df_old = c.qget(x, y, z, w)
    old_uids = {(x.uid, y.uid) : (z.uid, w.uid)
                for x, y, z, w in df_old.itertuples(index=False)}
    with query(storage) as c:
        x = Query(int).named('x')
        y = Query(int).named('y')
        z = g(x=x, y=y).named('z')
        df_new = c.qget(x, y, z)
    new_uids = {(x.uid, y.uid): (z[0].uid, z[1].uid)
                for x, y, z in df_new.itertuples(index=False)}
    assert old_uids == new_uids

def test_types(setup_tests):
    storage, cts = setup_tests
    storage:Storage

    with transient():
        x = add(x=23, y=42, __returns__=65)
        x = mean(x=[1, 2, 3], __returns__=2)
        x = get_prime_factors(x=23, __returns__=[42]) # oh no 
        x = mean_2d(arr=[[1, 2], [3, 4]], __returns__=0.0) 
        x = dict_mean(x={'a': 23, 'b': 42}, __returns__=0.0)
    
    ### check the logging of new calls for various types 
    with run(storage=storage, partition='test') as c:
        x = add(x=23, y=42, __returns__=65)
        assert unwrap(x) == 65
        y = mean(x=[1, 2, 3], __returns__=unwrap(x))
        assert y.obj() == 65
        z = get_prime_factors(x=23, __returns__=[unwrap(x)]) 
        assert z.unwrap() == [65]
        c.commit()
    storage.drop_instance_data(answer=True)

    ### check the logging for existing calls: should raise warnings that nothing
    ### is happening
    with run(storage=storage, partition='test') as c:
        x = add(23, 42)
        y = mean(x=[1, 2, 3])
        z = get_prime_factors(x=23)
        c.commit()
    with run(storage=storage, partition='test') as c:
        x = add(23, 42, __returns__=100)
        assert unwrap(x) == 65
        y = mean(x=[1, 2, 3], __returns__=unwrap(x))
        assert unwrap(y) == 2.0
        z = get_prime_factors(x=23, __returns__=[unwrap(x)])
        assert unwrap(z) == [23]
        c.commit()
    storage.drop_instance_data(answer=True)