from .common_imports import *
from .util.logging_ut import set_logging_level
from .util.common_ut import get_uid
from .core.config import CoreConfig

class Session(object):
    
    ### debugging
    def setup_logging(self, level:str='info'):
        set_logging_level(level=level)
    
sess = Session()

def get_scratch_dir(rel:TUnion[str, Path]=None, exist_ok:bool=False, 
                create:bool=True) -> Path:
    suffix = get_uid() if rel is None else rel
    path = CoreConfig.fs_storage_root / suffix
    if create:
        os.makedirs(path, exist_ok=exist_ok)
    return path