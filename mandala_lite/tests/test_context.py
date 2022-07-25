from mandala_lite.all import *
from mandala_lite.tests.utils import *

def test_nesting():
    storage = Storage()
    
    with run(storage) as c:
        assert c.mode == MODES.run
    
    with query(storage) as q:
        assert q.mode == MODES.query
    
    with run(storage) as c:
        assert c.mode == MODES.run
        assert c.storage is storage
        with c(mode=MODES.query) as q:
            assert q.storage is storage
            assert q.mode == MODES.query
        assert c.mode == MODES.run