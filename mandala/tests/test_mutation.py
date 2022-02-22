from .utils import *


def test_basics():
    storage = Storage()
    
    with define(storage):
        Array = Var(annotation=np.ndarray, name='Array')

        @op(mutations={'x': 0})
        def inc_array(x:Array) -> Array:
            x += 1
            return x 
        
    ############################################################################ 
    ### verify that in different modes mutation works as expected
    ############################################################################ 
    ### check UID update and value with actual computation
    with run(storage) as c:
        initial_value = np.zeros(shape=(10, 10))
        a = Array(initial_value.copy())
        uid_before = a.uid
        b = inc_array(x=a)
        uid_after = a.uid
        uid_output = b.uid
        assert uid_before != uid_after
        assert uid_output == uid_after
        assert eq_objs(x=initial_value + 1, y=unwrap(a))
        assert eq_objs(x=initial_value + 1, y=unwrap(b))
        c.commit()
    ### check UID update and value when retracing
    with run(storage) as c:
        initial_value = np.zeros(shape=(10, 10))
        a = Array(initial_value)
        uid_before = a.uid
        b = inc_array(x=a)
        uid_after = a.uid
        uid_output = b.uid
        assert uid_before != uid_after
        assert uid_output == uid_after
        assert eq_objs(x=initial_value + 1, y=unwrap(a))
        assert eq_objs(x=initial_value + 1, y=unwrap(b))
    ### check UID update when retracing LAZILY
    with run(storage, lazy=True) as c:
        initial_value = np.zeros(shape=(10, 10))
        a = Array(initial_value)
        uid_before = a.uid
        b = inc_array(x=a)
        uid_after = a.uid
        uid_output = b.uid
        assert uid_before != uid_after
        assert uid_output == uid_after
    storage.drop_instance_data(answer=True)

    ############################################################################ 
    ### verify that mutation loops work in computation and queries
    ############################################################################ 
    with run(storage) as c:
        a = np.zeros(shape=(10, 10))
        for i in range(10):
            a = inc_array(x=a)
        assert eq_objs(x=unwrap(a), y=np.zeros(shape=(10, 10)) + 10)
        c.commit()
    
    with run(storage) as c:
        initial = Array(np.zeros(shape=(10, 10)))
    
    with query(storage) as c:
        a = Array().identical([initial])
        for i in range(5):
            a = inc_array(x=a).named('array')
        df = c.qeval(a)
        assert df.shape == (1, 1)
        query_res = df['array'].item()
        assert np.all(unwrap(initial) + 5 == query_res)
    storage.drop_instance_data(answer=True)

def test_mock_nn():
    storage = Storage()
    
    with define(storage):
        Int = Var(annotation=int)
        LR = Var(annotation=float)
        Acc = Var(annotation=float)
        AccList = Var(annotation=TList[float])
        Model = Var(annotation=np.ndarray)
        
        @op()
        def make_model(x:Int) -> Model:
            return np.random.uniform(size=(10, 10), low=-x, high=x)

        @op(mutations={'model': 0})
        def train_one_step(model:Model, lr:LR) -> TTuple[Model, Acc]:
            model += lr * np.ones(shape=(10, 10))
            acc = model.mean()
            return model, acc
        
        ### one way to deal with queriability of mutations: wrap in superop
        ### that indexes the iteration. This can always be added later.
        @superop()
        def train_x_steps(model:Model, lr:LR, num_steps:Int) -> TTuple[Model, TList[Acc]]:
            accs = []
            for i in range(unwrap(num_steps)):
                model, acc = train_one_step(model=model, lr=lr)
                accs.append(acc)
            return model, AccList(accs)
    
    with run(storage) as c:
        initial_uids = []
        for x in (1, 2):
            for lr in (0.1, 0.2):
                model = make_model(x=x)
                initial_uids.append(model.uid)
                model, accs = train_x_steps(model=model, lr=lr, num_steps=20)
        c.commit()

    with query(storage) as c:
        x = Int()
        initial_model = make_model(x).named('initial_model')
        num_steps = Int().equals(20)
        final_model, _ = train_x_steps(model=initial_model, num_steps=num_steps)
        df = c.qget(x, initial_model, num_steps, final_model)
        assert df.shape[0] == 4 
        assert set(df['initial_model'].apply(lambda x: x.uid)) == set(initial_uids)