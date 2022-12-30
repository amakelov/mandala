from mandala.all import *
from mandala.tests.utils import *

if Config.has_dask:
    import dask
    import pytest
    from dask.distributed import Client
    import time

    DB_PATH = Path(__file__).parent / "output/dask.db"

    def test_computation():
        if DB_PATH.exists():
            DB_PATH.unlink()
        try:
            client = Client(n_workers=4)
            storage = Storage(multiproc=True, db_path=DB_PATH)

            @op(executor="dask")
            def f(x: int) -> int:
                time.sleep(1)
                return x + 1

            @op(executor="dask")
            def g(x: int) -> int:
                time.sleep(1)
                return x - 1

            @op(executor="dask")
            def h(x: int, y: int) -> int:
                time.sleep(1)
                return x + y

            for func in [f, g, h]:
                storage.synchronize(f=func)

            with storage.run():
                futures = []
                for i in range(10):
                    futures.append(h(f(i), g(i)))
                results = dask.compute(*futures)
                time.sleep(10)
        except:
            pass
        finally:
            if DB_PATH.exists():
                DB_PATH.unlink()
