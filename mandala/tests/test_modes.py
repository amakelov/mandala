from mandala.imports import *


def test_nesting():
    storage = Storage()
    with storage(mode='noop'):
        assert storage.mode == 'noop'
        with storage():
            assert storage.mode == 'run'
        assert storage.mode == 'noop'
        with storage(mode='noop'):
            assert storage.mode == 'noop'
        assert storage.mode == 'noop'
    assert storage.mode == 'run'


def test_noop_simple():
    @op
    def inc(x: int, *args, y: int = NewArgDefault(2), z: int = 23) -> int:
        return x + y + z + sum(args)
    
    storage = Storage()
    
    with storage(mode='noop'):
        # test various wrapped values
        res = inc(ValuePointer(id='one', obj=1), 1, 1, z=1)
    assert res == 6


def test_noop_composition():

    @op
    def inc(x: int) -> int:
        return x + 1
    
    @op
    def add(x: int, y: int) -> int:
        return x + y
    
    storage = Storage()
    

    # test that wrapped values are unwrapped
    with storage:
        x = inc(20)

    with storage:
        x = inc(20)
        with storage(mode='noop'):
            z = add(x, 21)
        assert z == 42
