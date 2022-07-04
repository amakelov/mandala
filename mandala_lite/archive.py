
class Signature:
    def __init__(self, input_names:List[str], n_outputs:int, defaults:Dict[str, Any]):
        self._input_names = input_names
        self._n_outputs = n_outputs
        self._defaults = defaults 
    
    @staticmethod
    def from_py(sig:inspect.Signature) -> 'Signature':
        input_names = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        n_outputs = len(sig.return_annotation) if sig.return_annotation is not inspect.Parameter.empty else 0
        defaults = {param.name:param.default for param in sig.parameters.values() if param.default is not inspect.Parameter.empty}
        return Signature(input_names, n_outputs, defaults)
