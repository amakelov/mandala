from mandala.imports import *


def test_single_func():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage:
        for i in range(10):
            inc(i)
    
    cf = storage.cf(inc)
    df = cf.df()
    assert df.shape == (10, 3)
    assert (df['var_0'] == df['x'] + 1).all()


def test_composition():
    storage = Storage()

    @op(output_names=['y'])
    def inc(x):
        return x + 1
    
    @op(output_names=['z'])
    def add(x, y):
        return x + y
    
    with storage:
        for x in range(5):
            y = inc(x)
            if x % 2 == 0:
                z = add(x, y)
        
    cf = storage.cf(add).expand_all()
    df = cf.df()
    assert df.shape[0] == 3
    assert (df['z'] == df['x'] + df['y']).all()

    cf = storage.cf(inc).expand_all()
    df = cf.df()
    assert df.shape[0] == 5
    assert (df['y'] == df['x'] + 1).all()
    assert (df['z'] == df['x'] + df['y'])[df['z'].notnull()].all()

def test_merge():
    storage = Storage()

    @op(output_names=['y'])
    def inc(x):
        return x + 1
    
    @op(output_names=['z'])
    def add(x, y):
        return x + y
    
    @op(output_names=['w'])
    def mul(x, y):
        return x * y
    
    @op(output_names=['v'])
    def final(t):
        return t**2
    
    with storage:
        for x in range(10):
            y = inc(x)
            if x < 5:
                z = add(x, y)
                v = final(z)
            else:
                w = mul(x, y)
                v = final(w)


    cf = storage.cf(final).expand_all().merge_vars()
    df = cf.df()
    assert df.shape[0] == 10