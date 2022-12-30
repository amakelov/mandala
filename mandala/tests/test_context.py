from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_nesting_old_api():
    storage = Storage()

    with FreeContexts.run(storage) as c:
        assert c.mode == MODES.run

    with FreeContexts.query(storage) as q:
        assert q.mode == MODES.query

    with FreeContexts.run(storage) as c:
        assert c.mode == MODES.run
        assert c.storage is storage
        with c(mode=MODES.query) as q:
            assert q.storage is storage
            assert q.mode == MODES.query
        assert c.mode == MODES.run


def test_nesting_new_api():
    storage = Storage()

    with storage.run() as c:
        assert c.mode == MODES.run

    with storage.query() as q:
        assert q.mode == MODES.query

    with storage.run() as c:
        assert c.mode == MODES.run
        assert c.storage is storage
        with storage.query() as q:
            assert q.storage is storage
            assert q.mode == MODES.query
        assert c.mode == MODES.run


def test_noop():
    # check that ops are noops when not in a context

    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    assert inc(23) == 24


def test_failures():
    storage = Storage()

    try:
        with storage.run(bla=23):
            pass
        assert False
    except:
        assert True
