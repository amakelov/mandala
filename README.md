<div align="center">
  <br>
    <img src="assets/logo-no-background.png" height=128 alt="logo" align="center">
  <br>
<a href="#install">Install</a> |
<a href="#features">Features</a> |
<a href="#video-walkthroughs">Videos</a> |
<a href="#usage">Usage</a> |
<a href="#tutorials">Tutorials</a> |
<a href="https://amakelov.github.io/blog/pl/">Blog post</a> |
<a href="#why-mandala">Why mandala?</a> 
</div>

# Computations that save, query and version themselves
https://user-images.githubusercontent.com/1467702/230719928-1981bd7b-3dbd-4a5c-891e-4b45c3e51aba.mp4

`mandala` is a memoization cache on steroids: shared and queriable across
function/data structure compositions, dependency-tracked and versioned, all
through a single decorator. Its primary use case is experiment data management.

It works by turning Python function calls and collections into interlinked pieces of data that
are automatically co-evolved with your codebase, and can be used to query
artifacts based on relationships. This unlocks extremely flexible yet simple
patterns of data management over computational artifacts.

## Features
- **simple and Python-native**: write computations in ordinary Python code, and
  they're automatically queriable and versioned. Data structures and control flow work just fine! 
- **fine-grained incremental computation**: with per-call dependency tracking,
changes in the code are automatically detected, and only the calls that
actually accessed the changed code are recomputed.
- **pattern-matching queries**: produce tables by directly pointing to
  variables in Python code; the rows will contain all values in storage that
  have the same functional relationships as the variables in the code.

## Install
```
pip install git+https://github.com/amakelov/mandala
```

## Video walkthroughs
### Incrementally grow a project with memoization
https://user-images.githubusercontent.com/1467702/230775562-d4bb4af6-f84c-45bd-b506-3818ba1b3423.mp4

Decorate the functions you want to memoize with `@op`, and compose programs
out of them by chaining their inputs and outputs using ordinary control flow and
data structures. Every such program is **end-to-end memoized**, which 
- turns it into an "imperative query interface" to its own results
- makes it simple to iterate on a project by growing a single piece of code
  without ever re-doing work

### Query by directly pattern-matching Python code
https://user-images.githubusercontent.com/1467702/230775616-773ffd7f-47f1-478d-92f2-8b2d45f9d4b1.mp4

Any computation in a `with storage.run():` block is also a
**declarative query interface** to analogous computations in the entire storage.
For example, `storage.similar(x, y, ...)` returns a table of all values in the
storage computed in the same way as `x, y, ...`, but possibly from different
initial parameters.

**Queries propagate relationships through collections (list, dict and set)**,
where only the qualitative composition of the elements matters (so, a list
variable can match e.g. a list where each element is of the form `f(i, g(j))`,
regardless of the length). The qualitative query is extracted from a concrete
computation in a principled way using a modified [color refinement
algorithm](https://en.wikipedia.org/wiki/Colour_refinement_algorithm). Note that
such queries have not been optimized for performance, and are likely to be slow
for large computational graphs (e.g. 1000+ calls).

A more expressive and explicit declarative interface is available via the `with
storage.query():` context manager. For more on how this works, see below #TODO.

### Automatic per-call versioning and dependency tracking
https://user-images.githubusercontent.com/1467702/230776548-a3bcb88f-8bc1-4c15-9658-1ea6c48badd6.mp4

Code changes to memoized functions or their dependencies (other memoized
functions, non-memoized functions and class methods decorated with `@track`, and
global variables in your codebase) are automatically tracked, and you can choose
whether they require recomputation of dependents (e.g., changes to core logic)
or not (e.g., refactoring, logging or backward-compatible extension of
functionality). 

**Every call** to a memoized function records *which* dependencies it accessed
and their versions. This fine-grained tracking means that a code change will
cause a recomputation only of the calls affected by it, instead of all calls to
a given memoized function. 

All code is **content-addressed**, meaning that restoring a previous state of
the codebase will reinterpret all memoized functions accordingly.

## Basic usage
This is a quick guide on how to get up to speed with the core features and avoid
common pitfalls.

### `Storage` and the `@op` decorator
A `Storage` instance holds all the data (saved calls and metadata) for a
collection of memoized functions. In a given project, you should have just one
`Storage` and many memoized functions connected to it. This way, the calls to
memoized functions create a queriable web of interlinked objects. 

```python
from mandala.all import Storage, op

storage = Storage(
  db_path='my_persistent_storage.db', # omit for an in-memory storage
  deps_path='path_to_code_folder/', # omit to disable automatic dependency tracking
  spillover_dir='spillover_dir/', # spillover storage for large objects
  # see docs for more options
)
```
The `@op` decorator marks a function `f` as memoizable. Some notes apply:
- `f` must have a **fixed number of arguments** (defaults are allowed, and
  arguments can always be added backward-compatibly)
- `f` must have a **fixed number of return values**, and this must be annotated in
  the function signature **or** declared as `@op(nout=...)`
- the `list`, `dict` and `set` collections, when used as argument/return
  annotations, cause elements of these collections to be stored separately. To
  avoid confusion, `tuple`s are reserved for specifying the number of outputs.

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import numpy as np

@op # core mandala decorator
def load_data(n_class: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    return load_digits(n_class=n_class, return_X_y=True)

@op
def train_model(X: np.ndarray, y: np.ndarray,
                n_estimators:int = 5) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=2).fit(X, y)
```

**Calling an `@op`-decorated function "normally" does not memoize**. To actually
put data in the storage, you must put calls inside a `with storage.run():`
block.

### Compute & memoize `with storage.run():`
**`@op`-decorated functions are designed to be composed** with one another
inside `storage.run()` blocks. This composability lets
you use the same piece of ordinary Python code to compute, save, load, *and any
combination of the three*:
```python
# generate the dataset. This saves `X, y` to storage.
with storage.run():
    X, y = load_data()

# later, train a model by directly adding on top of this code. `load_data` is
# not computed again
with storage.run():
    X, y = load_data()
    model = train_model(X, y)
  
# iterate on this with more parameters & logic
from sklearn.metrics import accuracy_score

@op
def get_acc(model:RandomForestClassifier, X:np.ndarray, y:np.ndarray) -> float:
    return round(accuracy_score(y_pred=model.predict(X), y_true=y), 2)

with storage.run():
    for n_class in (10, 5, 2):
        X, y = load_data(n_class)
        for n_estimators in (5, 10, 20):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            print(acc)
```
Memoized functions return `Ref` instances (`ValueRef`, `ListRef`,
...), which bundle the actual return value with storage metadata. To get the
return value itself, use `unwrap`. It works recursively on collections (lists,
dicts, sets, tuples) as well. 

You can **imperatively query** storage just by retracing some code that's been
entirely memoized:
```python
from mandala.all import unwrap
# use composable memoization as imperative query interface
with storage.run():
    X, y = load_data(5)
    for n_estimators in (5, 20):
        model = train_model(X, y, n_estimators=n_estimators)
        acc = get_acc(model, X, y)
        print(unwrap(acc))
```
### Implicit declarative queries
You can point to local variables in memoized code, and get a table of all values
in storage with the same functional dependencies as these variables have in the
code. For example, the `storage.similar()` method can be used to get values with
the same **joint computational history** as the given variables. To be able to
use a local variable in `storage.similar()`, it needs to be `wrap`ped as a `Ref`
(which has no effect on computation):

```python
from mandala.all import wrap
# use composable memoization as imperative query interface
with storage.run():
    n_class = wrap(5)
    X, y = load_data(n_class)
    for n_estimators in wrap((5, 20)): # `wrap` maps over list, set and tuple
        model = train_model(X, y, n_estimators=n_estimators)
        acc = get_acc(model, X, y)
    
df = storage.similar(n_class, n_estimators, acc)
```
When `verbose=True` in `storage.similar` (the default), you'll see the
computational graph that was inferred from the query:

```python
Pattern-matching to the following computational graph (all constraints apply):
    n_class = Q() # input to computation; can match anything
    X, y = load_data(n_class=n_class)
    n_estimators = Q() # input to computation; can match anything
    model = train_model(X=X, y=y, n_estimators=n_estimators)
    acc = get_acc(model=model, X=X, y=y)
    result = storage.df(n_class, n_estimators, acc)
```
This is also a good starting point for running an explicit query where you
directly provide the computational graph instead of extracting it from a
program.

### Explicit declarative queries `with storage.query():`
The kind of printout above can be directly copy-pasted into a `with
storage.query():` block. Here it is with some more explanation:
```python
with storage.query():
    n_class = Q() # creates a variable that can match any value in storage
    # @op calls impose constraints between the values variables can match
    X, y = load_data(n_class)
    n_estimators = Q() # another variable that can match anything
    model = train_model(X, y, n_estimators=n_estimators)
    acc = get_acc(model, X, y)
    # get a table where each row is a matching of the given variables
    # that satisfies the constraints
    result = storage.df(n_class, n_estimators, acc)
```
#### How the `.query()` context works
- a query variable, generated with `Q()` or as return value from an `@op`
  (like `X`, `y`, ... above), can in
principle match any value in the storage. 
- by chaining together calls to `@op`s, you impose constraints between the
  inputs and outputs to the op. For exampe, `X, y = load_data(n_class)` imposes
  the constraint that a matching of values `(a, b, c)` to `(n_class, X, y)` must
  satisfy `b, c = load_data(a)`. 
- You can omit even required function arguments. This leaves them unconstrained.
- the `df` method takes any sequence of variables, and returns a table
where each row is a matching of values to the respective variables that
satisfies **all** the constraints.

#### A warning about queries
**The query implementation has not been optimized for performance at this point**. Keep in mind that 
- if your query is not sufficiently constrained, there may be a combinatorial
explosion of results;
- if you query involves many variables and constraints, the default `SQL` query
solver may have a hard time, or flat out raise an error. Try using
`engine='naive'` in the `storage.similar()` or `storage.df()` methods instead.

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
of more magic.

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
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    X, y = load_digits(n_class=N_CLASS, return_X_y=True)
    return X, y

@op
def train_model(X, y, scale=False) -> LogisticRegression:
    if scale:
        X = scale_data(X)
    return LogisticRegression().fit(X, y)

@op
def eval_model(model, X, y, scale=False) -> Any:
    if scale:
        X = scale_data(X)
    return model.score(X, y)

storage = Storage(deps_path='__main__')

with storage.run():
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
def eval_model(model, X, y, scale=False) -> Any:
    if scale:
        X = scale_data(X)
    return round(model.score(X, y), 2)

with storage.run():
    X, y = load_data()
    for scale in [False, True]:
        model = train_model(X, y, scale=scale)
        acc = eval_model(model, X, y, scale=scale)
```
When entering the `storage.run()` block, the storage will detect the changes in
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
state of the code**. 

The content versions of a single dependency are organized in a `git`-like DAG
(currently, tree) that can be inspected using `storage.sources(f)` for
functions. 

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

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala/blob/master/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala/blob/master/tutorials/01_logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- [dependency tracking](https://github.com/amakelov/mandala/blob/master/tutorials/02_dependencies.ipynb) tutorial

## Related work
`mandala` brings together several ideas into a coherent whole:
- **memoization**: 
  - `functools.lru_cache` and `joblib.Memory` are standard Python
  solutions for a memoization cache, with the latter offering persistence.
  - memoization-related projects in the Python ecosystem and beyond are
    [funsies](https://github.com/aspuru-guzik-group/funsies) and [koji](https://arxiv.org/abs/1901.01908)
- **queries**: 
  - the concept of [conjunctive
  queries](https://en.wikipedia.org/wiki/Conjunctive_query) from relational
  databases is the workhorse of the query system. 
  - to extract a query from an arbitrary computational graph involving
    collections and function calls, a modified [color refinement
algorithm](https://en.wikipedia.org/wiki/Colour_refinement_algorithm) is used
- **versioning**: 
  - it is primarily a content-addressable, "truth-of-code" system (like `git`)
  - it shares some features with [semantic versioning](https://semver.org/), in
    particular the ability to mark changes as backward compatible. However,
    unlike semantic versioning it does not use human-annotated numbers, but
    content hashes.