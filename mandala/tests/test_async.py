from mandala.all import *
from mandala.tests.utils import *


@pytest.mark.asyncio
async def test_unit():
    storage = Storage()

    @op
    async def inc(x: int) -> int:
        time.sleep(1)
        print("Hi there!")
        return x + 1

    with storage.run():
        z = await inc(1)
    assert storage.similar(z).shape[0] == 1

    # now run 10 calls in parallel
    with storage.run():
        tasks = [inc(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
    assert storage.similar(results[0]).shape[0] == 10

    # run again
    with storage.run():
        tasks = [inc(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
    assert storage.similar(results[0]).shape[0] == 10
