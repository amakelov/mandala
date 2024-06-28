<div align="center">
  <br>
    <img src="../../assets/logo-no-background.png" height=128 alt="logo" align="center">
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

`mandala` eliminates the effort and code overhead of ML experiment tracking
([and beyond!](#galaxybrained-explanation)) with two tools:
- a decorator, `@op`, that automatically captures inputs, outputs and code
(+dependencies) of Python function calls, and ensures **the same call is never
computed twice**. `@op`s are **designed to be composed** into complex end-to-end
memoized programs.
- a data structure, `ComputationFrame` (a generalization of dataframes), which
enables explorations, queries and high-level operations over the saved web of
`@op` calls. By automatically propagating relationships through `@op`
composition, tables relating any variables in a project are readily available.

# Install
```
pip install git+https://github.com/amakelov/mandala
```

# Quickstart

[Run in Colab](https://colab.research.google.com/github/amakelov/mandala/blob/master/mandala/_next/tutorials/hello.ipynb)

# FAQs

## How is this different from other experiment tracking frameworks?
Compared to popular tools like W&B, MLFlow or Comet, `mandala`:
- **is tightly integrated with the actual Python code execution**, as
opposed to being an external logging framework. This makes it much
easier to compose multi-step experiments with non-trivial control flow 
    - For instance, Python's collections can be (if so desired) made
    transparent to the storage system, so that individual elements are
    stored separately and can be reused across collections and calls.
- **emphasizes memoization, which allows direct and transparent reuse of results**
- **allows reuse, queries and versioning on a more granular and flexible
level** - the function call - as opposed to entire scripts and/or notebooks.

## How is the `@op` cache invalidated?
- given inputs for a call to an `@op`, e.g. `f`, it searches for a past call
to `f` on inputs with the same contents (as determined by a hash function) where the dependencies accessed by this call (including `f`
itself) have versions compatible with their current state.
- compatibility between versions of a function is decided by the user: you
have the freedom to mark certain changes as compatible with past results.
- internally, `mandala` uses slightly modified `joblib` hashing to compute a
content hash for Python objects. This is practical for many use cases, but
not perfect, as discussed in the "gotchas" notebook TODO.

## Can I change the code of `@op`s, and what happens if I do?
- a frequent use case: you have some `@op` you've been using, then want to
extend its functionality in a way that doesn't invalidate the past results.
The recommended way is to add a new argument `a`, and provide a default
value for it wrapped with `NewArgDefault(x)`. When a value equal to `x` is
passed for this argument, the storage falls back on calls before 

## How self-contained is it?
- `mandala`'s core is simple (only a few kLoCs) and only depends on `pandas`
and `joblib`. 
- for visualization of `ComputationFrame`s, you should have `dot` installed
on the system level, and/or the Python `graphviz` library installed.

# Basic Documentation
Most methods in `mandala` are provided by the `Storage` and `ComputationFrame`
classes, and their docstrings provide detailed explanations. In general, you
should find yourself only interacting with 5-10 methods on a regular basis.

To complement this, here are a few more comprehensive tutorials on how to
get up to speed with the core features and methods, and tips on avoiding some
pitfalls and limitations.

- [create a `Storage`, save calls to `@op`s](#creating-a-storage-and-saving-calls-to-ops)
- [memoization basics](#compute--memoize-with-storagerun)
- [versioning and dependency tracking](#versioning-and-dependency-tracking)

## Create a `Storage`, save calls to `@op`s
A `Storage` object holds all data (saved calls, code and dependencies) for a
collection of memoized functions. In a given project, you should have just one
`Storage` and many memoized functions connected to it. This way, the calls to
memoized functions create a queriable web of interlinked objects. 

```python
from mandala.imports import Storage, op

storage = Storage(
    # omit for an in-memory storage
    db_path='my_persistent_storage.db', 
    # omit to disable automatic dependency tracking
    # use "__main__" to only track functions defined in the current session
    deps_path='__main__', 
)
```
`@op`-decorated functions will interact with a `storage` when called inside a 
`with storage:` block. 

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@op 
def load_data(n_class):
    return load_digits(n_class=n_class, return_X_y=True)

with storage:
    X, y = load_data()
    print(X)
```
The objects (`X, y`) returned by `@op`s are `Ref` instances, i.e. **references
to values**, and they may not even be in memory, because a previously executed
call's results are loaded only if they're needed. Regardless of that, every
`Ref` has some hashes that identify it uniquely w.r.t. the storage. For example,
the above prints:
```
AtomRef(array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ..., 10.,  0.,  0.],
       [ 0.,  0.,  1., ...,  3.,  0.,  0.],
       ...,
       [ 0.,  0.,  5., ...,  8.,  1.,  0.],
       [ 0.,  0.,  6., ...,  4.,  0.,  0.],
       [ 0.,  0.,  6., ...,  6.,  0.,  0.]]), hid='16e...', cid='908...')
```
where `hid` and `cid` are hashes that identify the *history* and *content* of
the `Ref`. If we run the code again, `X` is no longer in memory:
```python
with storage:
    X, y = load_data()
    print(X)
```
```
AtomRef(hid='16e...', cid='908...', in_memory=False)
```
To get the object wrapped by a `Ref`, call `storage.unwrap`:
```python
storage.unwrap(X) # loads from storage only if necessary
```
Other useful methods of the `Storage` include:
- `Storage.attach(inplace: bool)`: like `unwrap`, but puts the objects in the `Ref`s if they are not in-memory.
- `Storage.load_ref(hid: str, in_memory: bool)`: load a `Ref` by its history ID, optionally also loading the underlying object.

## Iterate on computations with memoization
**`@op`-decorated functions are designed to be composed** with one another.
Given Python code that's a composition of `@op`s, it becomes very cheap to
extend the computation by directly dumping new logic and parameters on top of that code:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

### new ops to train an ML model and evaluate
@op
def train_model(X, y, n_estimators=5):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=2).fit(X, y)

@op
def get_acc(model, X, y):
    return round(accuracy_score(y_pred=model.predict(X), y_true=y), 2)

### iterate on saved results by just dumping more computations on top
with storage:
    for n_class in (2, 5,):
        X, y = load_data(n_class) 
        for n_estimators in (5, 10):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            print(acc)
```
```
AtomRef(1.0, hid='146...', cid='b67...')
AtomRef(1.0, hid='60f...', cid='b67...')
AtomRef(0.85, hid='178...', cid='302...')
AtomRef(0.86, hid='01e...', cid='70e...')
```

You can **imperatively query** storage just by retracing some code that's been
entirely memoized, and/or combine this with new computation:
```python
# use composable memoization as imperative computation+query interface
with storage:
    X, y = load_data(5) # select a specific value of n_class
    for n_estimators in (5, 20): # 5 is reused, 20 is new
        model = train_model(X, y, n_estimators=n_estimators)
        acc = get_acc(model, X, y)
        print(storage.unwrap(acc))
```
```python
0.85
0.94
```

## Use `ComputationFrame`s to explore storage
There are several ways `ComputationFrames` can be created, all dispatched
through the `Storage.cf()` method.

### Visualize the full history of any result
Computation frames can load and visualize the history of a given value. Below,
we create a CF from the last captured value of `acc`, and apply the
`.expand_back()` method to add to the CF all calls reachable from `acc` going
backwards:
```python
cf = storage.cf(acc).expand_back(); cf.draw(verbose=True)
```
![](tutorials/readme_cf.svg)

The `.draw()` method shows the computational graph of a CF and how many
calls/refs there are in each function/variable. When called with `verbose=True`,
this method shows a summary of how the variable values and function calls
connect across the CF:
- **source (i.e. initial) `Ref`s** in each variable: the `Ref`s in this variable
that aren't outputs of any call in a function node using the variable as output.
- **sink (i.e. final) `Ref`s** in each variable: dual to source refs, these are
the `Ref`s that are not inputs to calls in any function node using the variable
as input.
- **edge `Ref`s** for each edge: for edges that are inputs to functions, this is
the number of refs from the input variable that are used in calls to the
function. For edges that are outputs of functions, this is the number of refs in
the output variable created by calls in the function node.

### Get the joint history of groups of results
You can also "manually" create variables and populate them with `Ref`s, by
passing in a dictionary where keys will become variable names and values are
`Ref`s used to populate these variables.

```python
with storage:
    accs = []
    models = []
    for n_class in (2, 5,):
        X, y = load_data(n_class) 
        for n_estimators in (5, 10):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            models.append(model)
            accs.append(acc)


cf = storage.cf({'acc': accs, 'model': models}).expand()
cf.draw(verbose=True, orientation='LR', path='readme_cf2.svg')
print(cf.df().drop(columns=['X', 'y']).to_markdown())
```
![](tutorials/readme_cf2.svg)

```
Extracting tuples from the computation graph:
    (X)@output_0, (y)@output_1 = load_data(n_class=n_class)
    (model)@output_0 = train_model(X=X, y=y, n_estimators=n_estimators)
    (acc)@output_0 = get_acc(model=model, X=X, y=y)
```
|    |   n_estimators |   n_class | load_data                                   | train_model                                   | model                                                | get_acc                                   |   acc |
|---:|---------------:|----------:|:--------------------------------------------|:----------------------------------------------|:-----------------------------------------------------|:------------------------------------------|------:|
|  0 |              5 |         2 | Call(load_data, cid='7e4...', hid='9e5...') | Call(train_model, cid='b39...', hid='bab...') | RandomForestClassifier(max_depth=2, n_estimators=5)  | Call(get_acc, cid='d87...', hid='c4c...') |  1    |
|  1 |              5 |         5 | Call(load_data, cid='6ee...', hid='40d...') | Call(train_model, cid='fd5...', hid='8ba...') | RandomForestClassifier(max_depth=2, n_estimators=5)  | Call(get_acc, cid='ea8...', hid='9c1...') |  0.85 |
|  2 |             10 |         2 | Call(load_data, cid='7e4...', hid='9e5...') | Call(train_model, cid='976...', hid='95b...') | RandomForestClassifier(max_depth=2, n_estimators=10) | Call(get_acc, cid='183...', hid='f5f...') |  1    |
|  3 |             10 |         5 | Call(load_data, cid='6ee...', hid='40d...') | Call(train_model, cid='03e...', hid='177...') | RandomForestClassifier(max_depth=2, n_estimators=10) | Call(get_acc, cid='7d5...', hid='de5...') |  0.86 |

## Versioning and dependency tracking
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

# Other gotchas

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

# Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala/blob/master/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala/blob/master/tutorials/01_random_forest.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- TODO: dependency tracking

# Roadmap

# Testimonials

> "`mandala` addresses a core challenge in my notebook workflow: being able to
> explore data with code, without having to worry about losing the results of
> expensive calculations." - *Adam Jermyn, Member of Technical Staff, Anthropic*


# Related work
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
