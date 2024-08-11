from .common_imports import *

def get_mandala_path() -> Path:
    import mandala

    return Path(os.path.dirname(mandala.__file__))

class Config:
    func_interface_cls_name = "Op"
    mandala_path = get_mandala_path()
    module_name = "mandala"
    tests_module_name = "mandala.tests"

    try:
        import PIL

        has_pil = True
    except ImportError:
        has_pil = False

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import rich

        has_rich = True
    except ImportError:
        has_rich = False
    
    try:
        import prettytable

        has_prettytable = True
    except ImportError:
        has_prettytable = False


if Config.has_torch:
    import torch

    def tensor_to_numpy(obj: Union[torch.Tensor, dict, list, tuple, Any]) -> Any:
        """
        Recursively convert PyTorch tensors in a data structure to numpy arrays.

        Parameters
        ----------
        obj : any
            The input data structure.

        Returns
        -------
        any
            The data structure with tensors converted to numpy arrays.
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            return {k: tensor_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(tensor_to_numpy(v) for v in obj)
        else:
            return obj
