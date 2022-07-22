from .kv import InMemoryStorage
from .rels import RelStorage, RelAdapter
from .calls import CallStorage

from ..common_imports import *
from ..core.sig import Signature
from ..core.config import Config

class Storage:
    """
    Groups together all the components of the storage system. 
    
    Responsible for things that require multiple components to work together,
    e.g. 
        - committing: moving calls from the "temporary" partition to the "main"
        partition. See also `CallStorage`.
        - synchronizing: connecting an operation with the storage and performing
        any necessary updates 
    """
    def __init__(self, root:Optional[Path]=None):
        self.root = root
        self.calls = CallStorage()
        # all objects (inputs and outputs to operations, defaults) are saved here
        self.objs = InMemoryStorage()
        # stores the memoization tables
        self.rel_storage = RelStorage()
        # manipulates the memoization tables
        self.rel_adapter = RelAdapter(rel_storage=self.rel_storage) 
        # stores the signatures of the operations connected to this storage
        # (external name, version) -> signature
        self.sigs = {}
    
    def commit(self):
        """
        Move calls from the temp partition to the main partition, putting them
        in relational storage.
        """
        keys = self.calls.temp.keys()
        temp_calls = self.calls.temp.mget(keys)
        self.rel_adapter.upsert_calls(calls=temp_calls)
        self.calls.main.mset(dict(zip(keys, temp_calls)))
        self.calls.temp.mdelete(keys)
    
    def synchronize(self, sig:Signature) -> Signature:
        """
        Synchronize an op's signature with this storage.

        - If this is a new operation, it's just added to the storage.
        - If this is an existing operation, 
            - if the new signature is compatible with the old one, it is updated
            and returned. TODO: if a new input is created, a new column is
            created in the relation for this op.
            - otherwise, an error is raised
        """
        if (sig.name, sig.version) not in self.sigs:
            res = sig.generate_internal()
            self.sigs[(res.name, res.version)] = res
            # create relation
            columns = [Config.uid_col] + list(res.internal_input_names) + [f'output_{i}' for i in range(res.n_outputs)]
            self.rel_storage.create_relation(name=res.internal_name, columns=columns)
            return res
        else:
            current = self.sigs[(sig.name, sig.version)]
            res = current.update(new=sig)
            # TODO: update relation if a new input was created
            return res