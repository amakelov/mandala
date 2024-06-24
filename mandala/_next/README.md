<div align="center">
  <br>
    <img src="assets/logo-no-background.png" height=128 alt="logo" align="center">
  <br>
<a href="#install">Install</a> |
<a href="#quickstart">Quickstart</a> |
<a href="#testimonials">Testimonials</a> |
<a href="#video-walkthroughs">Demos</a> |
<a href="#basic-usage">Usage</a> |
<a href="#other-gotchas">Gotchas</a> |
<a href="#tutorials">Tutorials</a>
</div>

# Computations that save, query and version themselves

<br>

**`mandala` eliminates the overhead of saving, loading,
querying and versioning results in computational projects**. 

To use it, you break your logic into functions whose outputs you want to save,
and decorate them with `@op`. Then you write plain Python composing these
functions, without any consideration for how results will be organized and
accesssed.

`mandala` automatically captures results, code and its dependencies as you
compute. Repeating an already saved call reuses the saved results. The results
form an interconnected web - a big computational graph that can be queried via
`ComputationFrame`s, a generalization of `pandas` `DataFrame`s.

While `mandala` is designed primarily with machine learning projects in mind,
it is much more generic.

## Main features

- **plain-Python**: decorate the functions whose calls you want to save, and
  just write ordinary Python code using them - including data structures
and control flow. The results are automatically accessible upon a
re-run, queriable, and versioned. No need to save, load, or name anything by yourself.
- **never compute the same thing twice**: `mandala` saves the result of each
  function call, and (hashes of) the inputs and the dependencies
  (functions/globals) accessed by the call. If later the inputs and dependencies
  are the same, it just loads the results from storage.
- **query easily using `ComputationFrame`s**: your code
  already knows the relationships between the variables in your project! To
  easily explore and query these relationships, the `ComputationFrame`
  generalizes `pandas` `DataFrame`s: the "columns" are a computational graph,
  and the "rows" are computations that (partially) follow this graph. Computation frames can be iteratively expanded backward/forward to add more computational context to the graph, and plain old `DataFrame`s can be extracted from any `ComputationFrame` for further analysis.
- **fine-grained versioning that's under your control**: each function's source
  code has its own "mini git repo" behind the scenes, and each call tracks the
  versions of all dependencies that went into it. You can decide when a change
  to a dependency (e.g. refactoring a function to improve readability) doesn't change its semantics (so calls dependent on it won't be recomputed). 


## Install
```
pip install git+https://github.com/amakelov/mandala
```

## Testimonials

> "`mandala` addresses a core challenge in my notebook workflow: being able to
> explore data with code, without having to worry about losing the results of
> expensive calculations." - *Adam Jermyn, Member of Technical Staff, Anthropic*


## Walkthroughs
### Rapidly iterate on a project with memoization
Decorate the functions you want to memoize with `@op`, and compose programs out
of them by chaining their inputs and outputs using ordinary **control flow** and
**collections**. Every such program is **end-to-end memoized**:
- it becomes an **imperative query interface** to its own results by
  (quickly) re-executing the same code, or parts of it
- it is **incrementally extensible** with new logic and parameters in-place,
  which makes it both easy and efficient to interact with experiments

### Query with `ComputationFrame`s
TODO

### Automatic per-call versioning and dependency tracking
![deps](https://user-images.githubusercontent.com/1467702/231246159-fc8996a1-0987-4cec-9f0d-f0408609886e.gif)

`mandala` comes with a very fine-grained versioning system:
- **per-call dependency tracking**: automatically track the functions and global
variables accessed by each memoized call, and alert you to changes in them, so
you can (carefully) choose whether a change to a dependency requires
recomputation of dependent calls (like bug fixes and changes to logic) or not
(like refactoring, comments, and logging)
- **the code determines all versions automatically**: use the current state of
each dependency in your codebase to automatically determine the currently
compatible versions of each memoized function to use in computation and queries.
In particular, this means that:
  - **you can go "back in time"** and access the storage relative to an earlier
  state of the code (or even branch in a new direction like in `git`) by
  just restoring this state
  - **the code is the truth**: when in doubt about the meaning of a result, you
  can just look at the current code.

## Quickstart
```python
from mandala.imports import *

# the storage saves calls and tracks dependencies, versions, etc.
storage = Storage( 
    deps_path='__main__' # track dependencies in current session
    ) 

@op # memoization (and more) decorator
def increment(x): 
  print('hi from increment!')
  return x + 1

increment(23) # function acts normally

with storage: # work against a given storage using a `with storage` block
  y = increment(23) # now the call is saved to `storage`

print(y) # result wrapped with metadata. 
print(storage.unwrap(y)) # `unwrap` gets the raw value

with storage:
  y = increment(23) # loads result from `storage`; doesn't execute `increment`

@op # type-annotate data structures w/ custom annotations to store elts separately
def average(nums: MList[int]): 
  print('hi from average!')
  return sum(nums) / len(nums)

# memoized functions are designed to be composed!
with storage: 
    # sliding averages of `increment`'s results over 3 elts
    nums = [increment(i) for i in range(5)]
    for i in range(3):
        result = average(nums[i:i+3])

# TODO: comp frames

# change implementation of `increment` and re-run
# you'll be asked if the change requires recomputing dependencies (say yes)
@op
def increment(x):
  print('hi from new increment!')
  return x + 2

with storage: 
    nums = [increment(i) for i in range(5)]
    for i in range(3):
        # only one call to `average` is executed!
        result = average(nums[i:i+3])

# query is ran against the *new* version of `increment`
# TODO
```

## Basic usage
This is a quick guide on how to get up to speed with the core features and avoid
common pitfalls.

- [defining storages and memoized functions](#storage-and-the-op-decorator)
- [memoization basics](#compute--memoize-with-storagerun)
- [query storage directly from computational code](#implicit-declarative-queries)
- [explicit query interface](#explicit-declarative-queries-with-storagequery)
- [versioning and dependency tracking](#versioning-and-dependency-tracking)

### `Storage` and the `@op` decorator
A `Storage` instance holds all the data (saved calls and metadata) for a
collection of memoized functions. In a given project, you should have just one
`Storage` and many memoized functions connected to it. This way, the calls to
memoized functions create a queriable web of interlinked objects. 

```python
from mandala.imports import Storage, op

storage = Storage(
  db_path='my_persistent_storage.db', # omit for an in-memory storage
  deps_path='path_to_code_folder/', # omit to disable automatic dependency tracking
)
```
The `@op` decorator marks a function `f` as memoizable.

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import numpy as np

@op # core mandala decorator
def load_data(n_class):
    return load_digits(n_class=n_class, return_X_y=True)

@op
def train_model(X, y, n_estimators=5):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=2).fit(X, y)
```

**Calling an `@op`-decorated function "normally" does not memoize**. To actually
put data in the storage, you must put calls inside a `with storage:` block.

### Compute & memoize inside `with storage:` blocks
**`@op`-decorated functions are designed to be composed** with one another
inside `with storage:` blocks. This composability lets you use the same piece of
ordinary Python code to compute, save, load, *and any combination of the three*:
```python
# generate the dataset. This saves `X, y` to storage.
with storage:
    X, y = load_data()

# later, train a model by directly adding on top of this code. `load_data` is
# not computed again
with storage:
    X, y = load_data()
    model = train_model(X, y)
  
# iterate on this with more parameters & logic
from sklearn.metrics import accuracy_score

@op
def get_acc(model, X, y):
    return round(accuracy_score(y_pred=model.predict(X), y_true=y), 2)

with storage:
    for n_class in (10, 5, 2):
        X, y = load_data(n_class)
        for n_estimators in (5, 10, 20):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            print(acc)
```
```python
AtomRef(0.66, cid='15a...')
AtomRef(0.73, cid='79e...')
AtomRef(0.81, cid='5a4...')
AtomRef(0.84, cid='6c4...')
AtomRef(0.89, cid='fb8...')
AtomRef(0.93, cid='c3d...')
AtomRef(1.0, cid='b67...')
AtomRef(1.0, cid='b67...')
AtomRef(1.0, cid='b67...')
```

Memoized functions return `Ref` instances (`AtomRef`, `ListRef`, ...), which
bundle the actual return value with storage metadata. To get the return value
itself, use `storage.unwrap(ref)` It works recursively on collections (lists,
dicts, sets, tuples) as well. 

You can **imperatively query** storage just by retracing some code that's been
entirely memoized:
```python
# use composable memoization as imperative query interface
with storage:
    X, y = load_data(5)
    for n_estimators in (5, 20):
        model = train_model(X, y, n_estimators=n_estimators)
        acc = get_acc(model, X, y)
        print(storage.unwrap(acc))
```
```python
0.84
0.93
```

### Versioning and dependency tracking
Passing a value to the `deps_path` parameter of the `Storage` class enables
dependency tracking and versioning. This means that any time a memoized function
*actually executes* (instead of loading an already saved call), it keeps track of
the functions and global variables it accesses along the way. 

The number of tracked functions should be limited for efficiency (you typically
don't want to track changes in installed libraries!). Setting `deps_path` to
`"__main__"` will only look for dependencies defined in the current interactive
session or process. Setting it to a folder will only look for dependencies
defined in this folder. 

#### NOTE: The `@track` decorator
The most efficient and reliable implementation of dependency tracking currently
requires you to explicitly put `@track` on non-memoized functions and classes
you want to track. This limitation may be lifted in the future, but at the cost
of more magic (i.e., automatically applying the decorator to functions in the
current local scope that originate in given paths).

#### What is a version?
A **version** for a memoized function is (to a first approximation) a set of
source codes for functions/methods/global variables accessed by some call to
this function. Even if you don't change anything in the code, a single function
can have multiple versions if it invokes different dependencies for different calls. For example, consider this code:
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

#### Semantic vs content changes and versions
For each change to the content of some dependency (the source code of a function
or the value of a global variable), you can choose whether this content change
is also a **semantic** change. A semantic change will cause all calls that
have accessed this dependency to not appear memoized **with respect to the new
state of the code**. The content versions of a single dependency are organized
in a `git`-like DAG (currently, tree) that can be inspected using
`storage.sources(f)` for functions. 

#### Going back in time
Since the versioning system is content-based, simply restoring an old state of
the code makes the storage automatically recognize which "world" it's in, and
which calls are memoized in this world.

#### A warning about non-semantic changes
The main motivation for allowing non-semantic changes is to maintain clarity in
the storage when doing routine code improvements (refactoring, comments,
logging). **However**, non-semantic changes should be applied with care. Apart from
being prone to errors (you wrongly conclude that a change has no effect on
semantics when it does), they can also introduce **invisible dependencies**:
suppose you factor a function out of some dependency and mark the change
non-semantic. Then the newly extracted function may in reality be a dependency
of the existing calls, but this goes unnoticed by the system.

## Other gotchas

- **under development**: the biggest gotcha is that this project is under active
development, which means things can change unpredictably.
- **slow at times**: it hasn't been optimized for performance, so you may run
into inefficiencies in large projects
- **pure functions**: you should probably only use it for functions with a
  deterministic input-output behavior if you're new to this project:
    - **changing a `Ref`'s object in-place will generally break things**. If you
    really need to update an object in-place, wrap the update in an `@op` so
    that you get instead a new `Ref` (with updated metadata) pointing to the
    same (changed) object, and discard the old `Ref`.
    - if a function does not have a **deterministic set of dependencies**
    it invokes for each given call, this may break the versioning system's
    invariants.
- **don't rename anything (yet)**: there isn't support yet for renaming
functions or their arguments.

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala/blob/master/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala/blob/master/tutorials/01_random_forest.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- TODO: dependency tracking

## Related work
`mandala` combines ideas from, and shares similarities with, many technologies.
Here are some useful points of comparison:
- **memoization**: 
  - standard Python memoization solutions are [`joblib.Memory`](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html)
  and
  [`functools.lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache).
  `mandala` uses `joblib` serialization and hashing under the hood.
  - [`incpy`](https://github.com/pajju/IncPy) is a project that integrates
    memoization with the python interpreter itself. 
  - [`funsies`](https://github.com/aspuru-guzik-group/funsies) is a
    memoization-based distributed workflow executor that uses an analogous notion
    of hashing to `mandala` to keep track of which computations have already been done. It
    works on the level of scripts (not functions), and lacks queriability and
    versioning.
  - [`koji`](https://arxiv.org/abs/1901.01908) is a design for an incremental
    computation data processing framework that unifies over different resource
    types (files or services). It also uses an analogous notion of hashing to
    keep track of computations. 
- **computation frames**:
  - computation frames are related to the idea of using certain functions   category theory, see e.g.
    [here](https://blog.algebraicjulia.org/post/2020/12/cset-conjunctive-queries/). 
- **versioning**:
  - the revision history of each function in the codebase is organized in a "mini-[`git`](https://git-scm.com/) repository" that shares only the most basic
    features with `git`: it is a
    [content-addressable](https://en.wikipedia.org/wiki/Content-addressable_storage)
    tree, where each edge tracks a diff from the content at one endpoint to that
    at the other. Additional metadata indicates equivalence classes of
    semantically equivalent contents.
  - [semantic versioning](https://semver.org/) is another popular code
    versioning system. `mandala` is similar to `semver` in that it allows you to
    make backward-compatible changes to the interface and logic of dependencies.
    It is different in that versions are still labeled by content, instead of by
    "non-canonical" numbers.
  - the [unison programming language](https://www.unison-lang.org/learn/the-big-idea/) represents
    functions by the hash of their content (syntax tree, to be exact).
