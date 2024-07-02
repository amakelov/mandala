# Changing `@op`s and managing versions

Passing a value to the `deps_path` parameter of the `Storage` class enables
dependency tracking and versioning. This means that any time a memoized function
*actually executes* (instead of reusing a past call's results), it keeps track
of the functions and global variables it accesses along the way. 

Usually, the functions we want to track are limited to user-defined ones (you
typically don't want to track changes in installed libraries!):
- Setting `deps_path` to `"__main__"` will only look for dependencies `f` defined in the current interactive session or process (as determined by `f.__module__`).
- Setting it to a folder will only look for dependencies defined in this folder. 

### Caveat: The `@track` decorator
The most efficient and reliable implementation of dependency tracking currently
requires you to explicitly put `@track` on non-memoized functions and classes
you want to track. This limitation may be lifted in the future, but at the cost
of more magic (i.e., automatically applying the decorator to functions in the
current local scope that originate in given paths).

The alternative (experimental) decorator implementation is based on
`sys.settrace`. Limitations are described in this [blog
post](https://amakelov.github.io/blog/deps/#syssettrace).

### What is a version of an `@op`?
A **version** for an `@op` is (to a first approximation) a collection of
- hashes of the source code of functions and methods;
- hashes of values of global variables

accessed when a call to this `@op` was executed. Even if you don't change
anything in the code, a single function can have multiple versions if it invokes
different dependencies for different calls. 

### Versioning in action
For example, consider this code:
```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mandala.imports import Storage, op, track
from typing import Tuple, Any

N_CLASS = 10

@track # to track a non-memoized function as a dependency
def scale_data(X):
    return StandardScaler(with_mean=True, with_std=False).fit_transform(X)

@op
def load_data():
    X, y = load_digits(n_class=N_CLASS, return_X_y=True)
    return X, y

@op
def train_model(X, y, scale=False):
    if scale:
        X = scale_data(X)
    return LogisticRegression().fit(X, y)

@op
def eval_model(model, X, y, scale=False):
    if scale:
        X = scale_data(X)
    return model.score(X, y)

storage = Storage(deps_path='__main__')

with storage:
    X, y = load_data()
    for scale in [False, True]:
        model = train_model(X, y, scale=scale)
        acc = eval_model(model, X, y, scale=scale)
```
When you run it, `train_model` and `eval_model` will each have two versions -
one that depends on `scale_data` and one that doesn't. You can confirm this by
calling `storage.versions(train_model)`. Now suppose we make some changes
and re-run:
```python
N_CLASS = 5

@track
def scale_data(X):
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)

@op
def eval_model(model, X, y, scale=False):
    if scale:
        X = scale_data(X)
    return round(model.score(X, y), 2)

with storage:
    X, y = load_data()
    for scale in [False, True]:
        model = train_model(X, y, scale=scale)
        acc = eval_model(model, X, y, scale=scale)
```
When entering the `storage` block, the storage will detect the changes in
the tracked components, and for each change will present you with the functions
affected:
- `N_CLASS` is a dependency for `load_data`;
- `scale_data` is a dependency for the calls to `train_model` and `eval_model`
  which had `scale=True`;
- `eval_model` is a dependency for itself.

### Semantic vs content changes and versions
For each change to the content of some dependency (the source code of a function
or the value of a global variable), you can choose whether this content change
is also a **semantic** change. A semantic change will cause all calls that
have accessed this dependency to not appear memoized **with respect to the new
state of the code**. The content versions of a single dependency are organized
in a `git`-like DAG (currently, tree) that can be inspected using
`storage.sources(f)` for functions. 

### Going back in time
Since the versioning system is content-based, simply restoring an old state of
the code makes the storage automatically recognize which "world" it's in, and
which calls are memoized in this world.

### A warning about non-semantic changes
The main motivation for allowing non-semantic changes is to maintain clarity in
the storage when doing routine code improvements (refactoring, comments,
logging). **However**, non-semantic changes should be applied with care. Apart from
being prone to errors (you wrongly conclude that a change has no effect on
semantics when it does), they can also introduce **invisible dependencies**:
suppose you factor a function out of some dependency and mark the change
non-semantic. Then the newly extracted function may in reality be a dependency
of the existing calls, but this goes unnoticed by the system.
