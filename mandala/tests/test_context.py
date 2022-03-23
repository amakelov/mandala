from .utils import *

@pytest.fixture(scope='session')
def setup_storage():
    storage = Storage()
    yield storage
    storage.drop(answer=True)

def test_builtin_contexts(setup_storage):
    storage:Storage = setup_storage
    
    with retrace(storage) as c:
        assert c.mode == MODES.run
        assert not c.allow_calls
        assert c.lazy
        assert c.disable_ray

def test_autonesting(setup_storage):
    storage:Storage = setup_storage

    # check config
    if not CoreConfig.enable_autonesting:
        try:
            with noop() as c:
                with run() as d:
                    pass
            assert False
        except ContextError as e:
            assert True
    else:
        with noop() as c:
            with run() as d:
                # check identity
                assert d is c
                assert d.mode == MODES.run
    
    # check mode switch
    with run(storage) as c:
        assert c.mode == MODES.run
        assert c._depth == 1
        # delete
        with delete():
            assert c.mode == MODES.delete
            assert c._depth == 2
        assert c.mode == MODES.run
        assert c._depth == 1
        # query 
        with query():
            assert c.mode == MODES.query
            assert c._depth == 2
        assert c.mode == MODES.run
        assert c._depth == 1
        # check more nesting
        with query():
            assert c.mode == MODES.query
            with qdelete():
                assert c.mode == MODES.query_delete
            assert c.mode == MODES.query
        assert c.mode == MODES.run
    assert not GlobalContext.exists()

    # check attribute overwrites
    with run(storage) as c:
        assert c.storage is storage
        with run(storage) as c:
            assert c.storage is storage


def test_1(setup_storage):
    storage:Storage = setup_storage

    with context(mode=MODES.run) as c:
        assert GlobalContext.exists()
        assert c.mode == MODES.run
    assert c.mode == MODES.noop
    assert not GlobalContext.exists()

    with run(storage) as c:
        assert c.mode == MODES.run
        with c(mode=MODES.delete) as del_c:
            assert del_c.mode == MODES.delete
            assert del_c is c
        assert c.mode == MODES.run

    with context(mode=MODES.run, storage=storage, partition='x') as c:
        assert c.mode == MODES.run
        assert c.storage is storage
        assert c.partition == 'x'
    assert c.mode == MODES.noop
    assert c.storage is None
    assert c.partition == CALLS.default_temp_partition

    ### 

    with run(storage=storage) as c:
        assert c.mode == MODES.run
    assert c.mode == MODES.run

    ### test proper recursive behavior
    with context(mode=MODES.run) as c:
        assert c.mode == MODES.run
        assert c.storage is None
        assert c.partition is CALLS.default_temp_partition
        with c(mode=MODES.transient, storage=storage) as c:
            assert c.mode == MODES.transient
            assert c.storage is storage
        assert c.mode == MODES.run
        assert c.storage is None
        with c(storage=storage) as c:
            # catch a bug we actually had: new *missing* updates shadow old
            # updates
            assert c.mode == MODES.run
        with c(partition='partition_1') as c:
            assert c.partition == 'partition_1'
        assert c.partition == CALLS.default_temp_partition
    assert c.mode == MODES.noop
    assert c.storage is None
    assert c.partition == CALLS.default_temp_partition
    assert not GlobalContext.exists()

    # more interesting: raise errors *inside* context and make sure the original
    # state is recovered
    c = noop()
    try:
        with c(mode=MODES.run):
            raise ValueError()
    except ValueError as e:
        pass
    assert c._depth == 0
    assert c.mode == MODES.DEFAULT
    
    with run(storage, buffered=True) as c:
        assert c.buffer is not None
    assert c.buffer is None

    with run(storage):
        with run(buffered=True) as c:
            assert c.buffer is not None
        assert c.buffer is None
    assert c.buffer is None

    ### test non-existing settings
    try:
        with run(storage, mode='fake_mode'):
            pass
        assert False
    except AssertionError:
        assert True
    
    storage.drop_instance_data(answer=True)

def test_context_passing():
    storage = Storage()
    
    @op(storage)
    def f(x:int) -> int:
        return x + 1 
    
    @superop(storage)
    def g(x:int, __context__:Context) -> int:
        print(__context__.mode)
        return f(f(x))
    
    @op(storage)
    def h(x:int, __context__:Context=None) -> int:
        return x
    
    with run(storage, autocommit=True) as c:
        g(23, __context__=c)
    with run(storage):
        h(23)
    storage.drop_instance_data(answer=True)