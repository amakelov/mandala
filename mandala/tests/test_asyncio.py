from mandala.all import *
from mandala.tests.utils import *
import pytest

DB_PATH = Path(__file__).parent / "output/asyncio.db"


@pytest.mark.asyncio
async def _test_computation():

    if DB_PATH.exists():
        DB_PATH.unlink()
    try:
        storage = Storage(db_path=DB_PATH)

        with storage.define():

            @op
            async def inc(x: int) -> int:
                return x + 1

            @op
            async def add(x: int, y: int) -> int:
                return x + y

        with storage.run():
            x = 23
            y = await inc(x)
            z = await add(x, y)

        with storage.run():
            futures = []
            for i in range(10):
                futures.append(inc(i))
            results = await asyncio.gather(*futures)

    except Exception as e:
        raise e
    finally:
        if DB_PATH.exists():
            DB_PATH.unlink()
