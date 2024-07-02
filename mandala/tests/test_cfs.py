from mandala._next.imports import *


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
    assert (df['output_0'] == df['x'] + 1).all()