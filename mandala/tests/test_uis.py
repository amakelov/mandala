from .utils import *
from .funcs import *
from .conftest import _setup_tests, setup_tests

def test_reprs(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    with run(storage=storage, autocommit=True):
        for ct in cts:
            ct.run_all()
    
    for loc in storage.obj_st.locs():
        repr(loc), str(loc)
        vref = storage.get(loc=loc)
        repr(vref), str(vref)
        repr(vref.get_type()), str(vref.get_type())
    for loc in storage.call_st.locs():
        repr(loc), str(loc)
        call = storage.call_st.get(loc)
        repr(call), str(call)
        repr(call.op), str(call.op)
    storage.drop_instance_data(answer=True)

def test_table_reading(setup_tests):
    storage, cts = setup_tests
    storage:Storage
    
    with run(storage=storage, buffered=True, partition='temp') as c:
        pairs = list(itertools.product(range(3), range(5)))
        for x, y in pairs:
            res = add(x=x, y=y)
        c.commit()
    
    dfs = [storage.rel_adapter.get_op_values(op=add.op,
                                             rename=True),
           add.get_table()]
    for df in dfs:
        assert set([tuple(elt) for elt in 
                    df[['x', 'y']].itertuples(index=False)]) == set(pairs)
        assert np.all(df['x'] + df['y'] == df['output_0'])
    storage.drop_instance_data(answer=True)

def test_table_after_update():
    storage = Storage()

    @op(storage)
    def inc(x:int) -> int:
        return x + 1 

    with run(storage, autocommit=True):
        for i in range(10):
            inc(i)
    df = inc.get_table()

    @op(storage)
    def inc(x:TUnion[int, float],
            how_many_times:int=CompatArg(default=1)) -> TUnion[int, float]:
        return x + how_many_times

    df = inc.get_table()
    storage.drop_instance_data(answer=True)