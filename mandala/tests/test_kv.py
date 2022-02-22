from .utils import *
from mandala.storages.kv_impl.dict_impl import DictStorage
from mandala.storages.kv_impl.joblib_impl import JoblibStorage
from mandala.storages.kv_impl.sqlite_impl import SQLiteStorage
from mandala.session import get_scratch_dir


class transition(object):

    def __init__(self, container:list):
        self.container = container
    
    def __call__(self, method:TCallable) -> 'method':
        self.container.append(method)
        return method
    

class precondition(object):

    def __init__(self, transition:TCallable, container:dict):
        self.transition = transition
        self.container = container
    
    def __call__(self, method:TCallable) -> 'method':
        self.container[self.transition] = method
        return method


class KVComparison(object):
    transitions = []
    preconditions = {}
    
    def __init__(self, first:KVStore, second:KVStore, require_empty:bool=False):
        if require_empty:
            assert first.empty
            assert second.empty
        self.first = first
        self.second = second
        self.kvs = [self.first, self.second]
    
    def compare(self):
        first_data = {k: self.first.get(k=k) for k in self.first.keys()}
        second_data = {k: self.second.get(k=k) for k in self.second.keys()}
        assert first_data == second_data
    
    def check_equal_returns(self, *returns:TAny):
        assert returns[0] == returns[1]
    
    def test(self, iterations:int):
        successes = 0
        while successes < iterations:
            transition = np.random.choice(self.transitions)
            # logging.info(f'Sampled transition: {transition.__name__}')
            default_precondition = lambda x: True
            precondition = self.preconditions.get(transition, default_precondition)
            if precondition(self):
                logging.info(f'Found possible transition: {transition.__name__} (current size: {self.first.size})')
                successes += 1
                transition(self)
                self.compare()
    
    ############################################################################ 
    ### KV generators
    ############################################################################ 
    def gen_key(self, exist:bool=True) -> str:
        if exist:
            return np.random.choice(self.first.keys())
        else:
            return get_uid()
    
    def mgen_key(self, exist:bool=True) -> TList[str]:
        """
        Generate keys that either all exist in `self.first` or none of them do.
        Could generate repetitions.

        Args:
            exist (bool, optional): [description]. Defaults to True.

        Returns:
            TList[str]: [description]
        """
        n_total = self.first.size
        size = max(5, int(0.1*n_total))
        if exist:
            if self.first.empty:
                return []
            return np.random.choice(self.first.keys(), size=size).tolist()
        else:
            return [get_uid() for _ in range(size)]
    
    def gen_val(self) -> TList[TAny]:
        return np.random.choice(range(100))
    
    def mgen_val(self, size:int=None) -> TList[TAny]:
        size = self.first.size if size is None else size
        return np.random.choice(range(100), size=size).tolist()
    
    def get_mixed_keys(self) -> TList[str]:
        return self.mgen_key(exist=True) + self.mgen_key(exist=False)
    
    ############################################################################ 
    ### transitions
    ############################################################################ 
    @transition(container=transitions)
    def exists(self):
        k = self.gen_key(exist=True)
        self.check_equal_returns(self.first.exists(k=k), self.second.exists(k=k))
    
    @precondition(transition=exists, container=preconditions)
    def exists_precond(self):
        return not self.first.empty
    
    @transition(container=transitions)
    def set(self):
        k = self.gen_key(exist=False)
        v = self.gen_val()
        for kv in self.kvs:
            kv.set(k=k, v=v)
        
    @precondition(transition=set, container=preconditions)
    def set_precond(self):
        return True

    @transition(container=transitions)
    def get(self):
        k = self.gen_key(exist=True)
        self.check_equal_returns(self.first.get(k=k), self.second.get(k=k))
    
    @precondition(container=preconditions, transition=get)
    def get_precond(self):
        return not self.first.empty
    
    @transition(container=transitions)
    def delete(self):
        k = self.gen_key(exist=True)
        for kv in self.kvs:
            kv.delete(k=k)
    
    @precondition(container=preconditions, transition=delete)
    def delete_precond(self):
        return not self.first.empty

    ### mmethods
    @transition(container=transitions)
    def mexists(self):
        ks = self.get_mixed_keys()
        self.check_equal_returns(self.first.mexists(ks), self.second.mexists(ks))
    
    @precondition(transition=mexists, container=preconditions)
    def mexists_precond(self):
        return True
    
    @transition(container=transitions)
    def mget(self):
        ks = self.mgen_key(exist=True)
        self.check_equal_returns(self.first.mget(ks), self.second.mget(ks))
    
    @precondition(transition=mget, container=preconditions)
    def mget_precond(self):
        return not self.first.empty
    
    @transition(container=transitions)
    def mset(self):
        ks = self.get_mixed_keys()
        vs = self.mgen_val(size=len(ks))
        print(len(ks), len(vs))
        for kv in self.kvs:
            kv.mset(mapping={k: v for k, v in zip(ks, vs)})
    
    @precondition(transition=mset, container=preconditions)
    def mset_precond(self):
        return True

    ### queries
    @transition(container=transitions)
    def where(self):
        if np.random.choice([True, False]):
            keys = self.mgen_key(exist=True)
        else:
            keys = None
        pred = lambda x: x % 2 == 0
        try:
            a = sorted(self.first.where(pred=pred, keys=keys))
            b = sorted(self.second.where(pred=pred, keys=keys))
            self.check_equal_returns(a, b)
        except:
            raise 
    
    @transition(container=transitions)
    def isin(self):
        if np.random.choice([True, False]):
            keys = self.mgen_key(exist=True)
        else:
            keys = None
        rng = list(range(50))
        try:
            a = sorted(self.first.isin(rng=rng, keys=keys))
            b = sorted(self.second.isin(rng=rng, keys=keys))
            self.check_equal_returns(a, b)
        except:
            raise 


@pytest.mark.parametrize('kv_cls', [JoblibStorage, SQLiteStorage])
def test_generative(kv_cls:TType[KVStore]):
    first = DictStorage()
    first.attach(root=get_scratch_dir())
    second = kv_cls()
    second.attach(root=get_scratch_dir())
    cmp = KVComparison(first=first, second=second)
    cmp.test(iterations=200)
    
def test_move():
    storages = [JoblibStorage(), SQLiteStorage(), DictStorage()]
    for x in storages:
        x.attach(root=get_scratch_dir())
        new_root = get_scratch_dir(create=False)
        data = {get_uid(): get_uid() for _ in range(100)}
        x.mset(mapping=data)
        x.move(new_root=new_root)
        ks = x.keys()
        recovered_vs = x.mget(ks)
        recovered = {k: v for k, v in zip(ks, recovered_vs)}
        assert data == recovered