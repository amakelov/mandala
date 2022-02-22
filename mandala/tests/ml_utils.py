from .utils import *
from numpy import ndarray

if EnvConfig.has_sklearn:
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score


    ### going all SQLite here
    storage = Storage(obj_kv=SQLiteStorage, call_kv=SQLiteStorage)
    with context(storage=storage):
        @op()
        def get_data(n_samples:int, n_features:int) -> TTuple[ndarray, ndarray]:
            X, y = make_classification(n_samples=n_samples, 
                                       n_features=n_features)
            return X, y

        @op()
        def split_data(X:ndarray, y:ndarray,
                       test_size:float=0.2) -> TTuple[ndarray, ndarray, ndarray, ndarray]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            return X_train, y_train, X_test, y_test

        @op()
        def train_lr(X:ndarray, y:ndarray, 
                     model_params:TDict[str, TAny]) -> LogisticRegression:
            model = LogisticRegression(**model_params)
            model.fit(X, y)
            return model
            
        @op()
        def eval_model(model:LogisticRegression, X:ndarray, y:ndarray) -> float:
            return accuracy_score(y_true=y, y_pred=model.predict(X))