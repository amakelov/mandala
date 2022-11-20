from mandala_lite.all import *
from mandala_lite.tests.utils import *

if Config.has_torch:
    import torch

    def test_jit_script():
        storage = Storage()

        @op
        @torch.jit.script
        def f(x: torch.Tensor, y: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            # a function with multiple outputs
            if y:
                return x, x
            else:
                return 2 * x, x

        with storage.run():
            x = torch.ones(1)
            y = f(x)

        @op
        @torch.jit.script
        def g(x: torch.Tensor) -> torch.Tensor:
            # a function with a single output
            return x

        with storage.run():
            x = torch.ones(1)
            y = g(x)

        @op
        @torch.jit.script
        def h(x: torch.Tensor):
            # a function with no outputs
            return

        with storage.run():
            x = torch.ones(1)
            h(x)

        data = storage.rel_adapter.get_all_call_data()
        assert len(data) == 3
        for k, v in data.items():
            assert v.shape[0] == 1
