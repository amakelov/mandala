from ..common_imports import *
from ..core.sig import Signature

class SigStorage:
    """
    Keeps track of the operations connected to a given storage.
    
    Responsible for synchronizing a memoized function with the storage, making
    sure the interface of the function you pass is compatible with what's on
    record.
    """
    def __init__(self):
        # (external name, version) -> signature
        self.sigs:Dict[Tuple[str, int], Signature] = {}
    
    def synchronize(self, sig:Signature) -> Signature:
        """
        Synchronize a signature with this signature storage.

        - If this is a new operation, it's just added to the storage.
        - If this is an existing operation, 
            - if the new signature is compatible with the old one, it is updated
            and returned
            - otherwise, an error is raised
        """
        if (sig.name, sig.version) not in self.sigs:
            res = sig.generate_internal()
            self.sigs[(res.name, res.version)] = res
            return res
        else:
            current = self.sigs[(sig.name, sig.version)]
            res = current.update(new=sig)
            return res