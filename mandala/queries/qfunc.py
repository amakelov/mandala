from ..common_imports import *


class Modes(object):
    func = 'func'
    query = 'query'


class QueryFunc(object):
    def __init__(self, func:TCallable):
        self.func = func
        self.mode = Modes.func
        self.sig = inspect.signature(self.func)
        self.kwargs_accumulator = []
    
    def __enter__(self):
        self.mode = Modes.query
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):   
        self.mode = Modes.func
        if exc_type:
            raise exc_type(exc_value).with_traceback(exc_traceback)
        return None
    
    def __call__(self, *args, **kwargs):
        if self.mode == Modes.func:
            return self.func(*args, **kwargs)
        else:
            self.kwargs_accumulator.append(
                dict(self.sig.bind(*args, **kwargs).arguments)
            )
    
    def query(self) -> pd.DataFrame:
        kwarg_df = pd.DataFrame(self.kwargs_accumulator)
        frames = [self.func(**kwargs) for kwargs in self.kwargs_accumulator]
        for frame, (idx, kwargs_series) in zip(frames, kwarg_df.iterrows()):
            for k, v in kwargs_series.items():
                frame[k] = v
        return pd.concat(frames)


class QFuncDecorator(object):
    def __init__(self):
        pass
    
    def __call__(self, func:TCallable) -> 'func':
        return QueryFunc(func=func)
    
    
qfunc = QFuncDecorator