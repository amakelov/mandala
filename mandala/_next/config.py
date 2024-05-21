from common_imports import *

class Config:

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

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
