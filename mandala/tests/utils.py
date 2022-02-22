"""
Set up things needed for testing
"""
import pytest

from mandala.common_imports import *
from mandala.util.common_ut import get_uid, eq_objs
from mandala.util.logging_ut import set_logging_level
from mandala.core.utils import CompatArg, BackwardCompatible, AsType, AsTransient, AsDelayedStorage
from mandala.core.bases import unwrap, detached, is_instance, is_deeply_persistable, is_deeply_in_memory
from mandala.core.tps import Type, AnyType, AtomType, ListType, DictType, UnionType, get_union, TypeWrapper
from mandala.core.impl import AtomRef, ListRef, DictRef, ConstructDict, ConstructList, GetItemList, GetKeyDict, BaseCall
from mandala.core.wrap import wrap_constructive, wrap_structure, wrap_as_atom, wrap_detached
from mandala.core.config import CoreConfig, PSQLConfig, EnvConfig, MODES, CALLS
from mandala.core.exceptions import SynchronizationError, VRefNotInMemoryError
from mandala.storages.objects import BaseObjStorage, ObjStorage, BaseObjLocation, PartitionedObjLocation
from mandala.storages.calls import *
from mandala.storages.kv_impl.sqlite_impl import SQLiteStorage
from mandala.storages.kv_impl.joblib_impl import JoblibStorage
from mandala.storages.kv_impl.dict_impl import DictStorage
from mandala.storages.rel_impl.sqlite_impl import SQLiteRelStorage
from mandala.storages.rel_impl.psql_utils import get_db_root
from mandala.ui.execution import *
from mandala.ui.storage import ValAdapter, RelAdapter, CallAdapter, CallBuffer
from mandala.ui.context import context, run, query, transient, delete, noop, define, GlobalContext, ContextError, Context, qdelete, retrace, capture
from mandala.ui.funcs import op, FuncOpUI, superop
from mandala.ui.vars import Var, BuiltinVars, Query
from mandala.queries.rel_weaver import ValQuery, GetItemQuery, MakeList, MakeDict
from mandala.queries.qfunc import qfunc

from mandala.util import shell_ut

IndexQuery = BuiltinVars.IndexQuery
KeyQuery = BuiltinVars.KeyQuery

set_logging_level(level=CoreConfig.logging_level)

################################################################################
### test configuration
################################################################################
if EnvConfig.has_dask:
    import dask
if EnvConfig.has_ray:
    import ray
    
    
def setup_test_config():
    if CoreConfig.test_output_root is None:
        test_output_root = Path(os.path.dirname(__file__)) / '../../test_output/'
    else:
        test_output_root = CoreConfig.test_output_root
    CoreConfig.set_fs_storage_root(test_output_root)
    if EnvConfig.has_psql:
        # TODO: export these settings somehow
        CoreConfig.psql.set(user='amakelov', host='localhost', port=5432, 
                root_db_name='postgres', password='postgres')

setup_test_config()
################################################################################
### generative helpers
################################################################################
class GenIndex(object):
    
    def __init__(self):
        self.gens = {}
    

gen_index = GenIndex()
    
class gen(object):
    
    def __init__(self):
        pass
    
    def __call__(self, func:TCallable) -> 'func':
        gen_index.gens[self] = func
        return func

        
class ValueGenerator(object):

    def __init__(self):
        self.pool = []
    
    def depth(self, x) -> int:
        if isinstance(x, list):
            if len(x) == 0:
                return 1
            return 1 + max(self.depth(elt) for elt in x)
        elif isinstance(x, dict):
            if len(x) == 0:
                return 1
            return 1 + max(self.depth(val) for val in x.values())
        else:
            return 0

    def choose(self) -> TAny:
        depths = [self.depth(x) for x in self.pool]
        return self.pool[np.argmax(depths)]
    
    def populate(self, iterations:int):
        for i in range(iterations):
            generator = np.random.choice(list(gen_index.gens.values()))
            generator(self)
    
    ############################################################################ 
    ### generators
    ############################################################################ 
    @gen()
    def gen_int(self):
        self.pool.append(int(np.random.choice(range(-10, 10))))
    
    @gen()
    def gen_str(self):
        self.pool.append(get_uid())
    
    # @gen() hashing potentially non-deterministic
    def gen_array(self):
        self.pool.append(np.random.uniform(size=(4, 5)))

    @gen()
    def gen_list(self):
        if not self.pool:
            self.pool.append([])
        else:
            idxs = np.random.choice(a=range(len(self.pool)), size=np.random.choice(range(len(self.pool))),
                                    replace=True)
            lst = [self.pool[idx] for idx in idxs]
            self.pool.append(lst)
    
    @gen()
    def gen_dict(self):
        if not self.pool:
            self.pool.append({})
        else:
            idxs = np.random.choice(a=range(len(self.pool)), size=np.random.choice(range(len(self.pool))),
                                    replace=True)
            keys = [get_uid() for _ in range(len(idxs))]
            dct = {k: self.pool[idx] for k, idx in zip(keys, idxs)}
            self.pool.append(dct)
    
    def gen_atom(self):
        pass
