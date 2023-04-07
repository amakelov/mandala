from mandala.all import *
from mandala.tests.utils import *


def test_unit():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        x = inc(23)
    df = storage.get_table(inc, values="uids")
    assert df.shape[0] == 1
    with storage.run():
        y = inc(22)
        z = inc(y)
    df = storage.get_table(inc, values="uids")
    assert df.shape[0] == 3


def test_queries():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def dec(x: int) -> int:
        return x - 1

    with storage.run():
        x = inc(23)

    with storage.run():
        y = dec(24)

    with storage.query():
        x = inc(Q())
        y = dec(x)
        df = storage.df(x, y)
    assert df.empty
    num_content_calls_inc = storage.get_table(
        inc, drop_duplicates=True, values="uids"
    ).shape[0]
    num_content_calls_dec = storage.get_table(
        dec, drop_duplicates=True, values="uids"
    ).shape[0]
    assert num_content_calls_inc == 1
    assert num_content_calls_dec == 1

    with storage.run(allow_calls=False):
        x = inc(23)
        y = dec(x)
    num_content_calls_inc = storage.get_table(
        inc, drop_duplicates=True, values="uids"
    ).shape[0]
    num_content_calls_dec = storage.get_table(
        dec, drop_duplicates=True, values="uids"
    ).shape[0]
    assert num_content_calls_inc == 1
    assert num_content_calls_dec == 1
    with storage.query():
        x = inc(Q())
        y = dec(x)
        df = storage.df(x, y)
    assert df.shape[0] == 1


def test_bug():
    storage = Storage()

    @op
    def inc(x: int) -> int:
        return x + 1

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        for x in range(5):
            for y in range(5):
                z = inc(x)
                w = add(z, y)
    assert storage.get_table(add).shape[0] == 25
