from .utils import *
if EnvConfig.has_sklearn:
    from .ml_utils import storage
    from .ml_utils import *

    def test_lr():
        with run(storage=storage, buffered=True, lazy=True) as c:
            for n_samples in tqdm.tqdm((10, 100, 1000, 10_000)):
                for n_features in (10, 100):
                    X, y = get_data(n_samples=n_samples, n_features=n_features)
                    X_train, y_train, X_test, y_test = split_data(X=X, y=y)
                    for C in (0.01, 0.1, 1.0, 10.0):
                        model = train_lr(X=X_train, y=y_train, 
                                         model_params={'C': C})
                        acc = eval_model(model=model, X=X_test, y=y_test)
            c.commit()
       
        with query(storage=storage) as c:
            n_samples = Query(int)
            n_features = Query(int)
            X, y = get_data(n_samples=n_samples, n_features=n_features)
            X_train, y_train, X_test, y_test = split_data(X=X, y=y)
            model_params = Query(TDict[str, TAny])
            model = train_lr(X=X_train, y=y_train, model_params=model_params)
            acc = eval_model(model=model, X=X_test, y=y_test)
            df = c.qeval(n_samples, n_features, model_params, acc)