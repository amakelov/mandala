<div align="center">
  <br>
    <img src="assets/logo-no-background.png" height=128 alt="logo" align="center">
  <br>
<a href="https://amakelov.github.io/blog/pl/">Blog post</a> |
<a href="#install">Install</a> |
<a href="#testimonials">Testimonials</a> |
<a href="#video-walkthroughs">Demos</a> |
<a href="#basic-usage">Usage</a> |
<a href="#other-gotchas">Gotchas</a> |
<a href="#tutorials">Tutorials</a>
</div>

[![Gitter](https://img.shields.io/gitter/room/amakelov/mandala)](https://app.gitter.im/#/room/#mandala:gitter.im)

# Computations that save, query and version themselves

![tldr](https://user-images.githubusercontent.com/1467702/231244639-f318af0e-3993-4ad1-8822-e8d889003dc1.gif)

<br>

`mandala` is a framework for experiment tracking and [incremental
computing](https://en.wikipedia.org/wiki/Incremental_computing) with a simple
plain-Python interface. It automates away the pain of experiment data management
(*did I run this already? what's in this file? where's the result of that
computation?* ...) with the following mix of features:
- **plain-Python**: decorate the functions whose calls you want to save, and
  just write ordinary Python code using them - including data structures
and control flow. The results are automatically accessible upon a
re-run, queriable, and versioned. No need to save, load, or name anything by yourself
- **never compute the same thing twice**: `mandala` saves the result of each
  function call, and (hashes of) the inputs and the dependencies
  (functions/globals) accessed by the call. If later the inputs and dependencies
  are the same, it just loads the results from storage.
- **query by pattern-matching directly to computational code**: your code
  already knows the relationships between variables in your project! `mandala`
  lets you produce tables relating any variables by [directly pointing to
  the code establishing the wanted relationship between them](#query-by-directly-pattern-matching-python-code).
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


## Video walkthroughs
### Rapidly iterate on a project with memoization
![mem](https://user-images.githubusercontent.com/1467702/231246050-21855bb2-6ce0-43d6-b7c0-3e0ed2a68f28.gif)

Decorate the functions you want to memoize with `@op`, and compose programs out
of them by chaining their inputs and outputs using ordinary **control flow** and
**collections**. Every such program is **end-to-end memoized**:
- it becomes an **imperative query interface** to its own results by
  (quickly) re-executing the same code, or parts of it
- it is **incrementally extensible** with new logic and parameters in-place,
  which makes it both easy and efficient to interact with experiments

### Query by directly pattern-matching Python code
![query](https://user-images.githubusercontent.com/1467702/231246102-276d7ae9-3a7f-46f8-9899-ae9dcf4f0484.gif)

Any computation in a `with storage.run():` block is also a
**declarative query interface** to analogous computations in the entire storage:
- **get a table of all values with the same computational history**:
`storage.similar(x, y, ...)` returns a table of all values in the storage that
were computed in the same ways as `x, y, ...`, but possibly starting from
different inputs
- **query through collections**: queries propagate provenance from a collection
  to its elements and vice-versa. This means you can query through operations
  that aggregate many objects into one, or produce many objects from a fixed
  number. 
  - **NOTE**: a collection in a computation can pattern-match any collection in
  the storage with the same *kinds* of elements (as given by their computational
  history), but not necessarily in the same order or quantity. This ensures that
  you don't only match to the specific computation you have, but all *analogous*
  computations too.
- **define queries explicitly**: for full control, use the `with storage.query():` context manager. For more on how this works, see [below](#explicit-declarative-queries-with-storagequery)

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
- `f` must have a **compatible interface** throughout its life. 
  - **The only way to change `f`'s interface once it already has memoized calls
  is to add new arguments with default values.**
  - if you want to change the interface in an incompatible way, you should
    either just make a new function (under a new name), or increment the
    `@op(version=...)` argument.
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
```python
ValueRef(0.66, uid=15a...)
ValueRef(0.73, uid=79e...)
ValueRef(0.81, uid=5a4...)
ValueRef(0.84, uid=6c4...)
ValueRef(0.89, uid=fb8...)
ValueRef(0.93, uid=c3d...)
ValueRef(1.0, uid=b67...)
ValueRef(1.0, uid=b67...)
ValueRef(1.0, uid=b67...)
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
```python
0.84
0.93
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
    
storage.similar(n_class, n_estimators, acc)
```
```python
Pattern-matching to the following computational graph (all constraints apply):
    n_estimators = Q() # input to computation; can match anything
    n_class = Q() # input to computation; can match anything
    X, y = load_data(n_class=n_class)
    model = train_model(X=X, y=y, n_estimators=n_estimators)
    acc = get_acc(model=model, X=X, y=y)
    result = storage.df(n_class, n_estimators, acc)
   n_class  n_estimators       acc
1       10            5       0.66
0       10            10      0.73
2       10            20      0.81
7       5             5       0.84
6       5             10      0.89
8       5             20      0.93
4       2             5       1.00
3       2             10      1.00
5       2             20      1.00
```

The computational graph printed out by the query (default `verbose=True`) is
also a good starting point for running an explicit query where you directly
provide the computational graph instead of extracting it from a program.

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
- **slow**: it hasn't been optimized for performance, so many things are quite
inefficient
- **pure functions**: you should probably only use it for functions with a
  deterministic input-output behavior if you're new to this project:
    - **changing a `Ref`'s object in-place will generally break things**. If you
    really need to update an object in-place, wrap the update in an `@op` so
    that you get instead a new `Ref` (with updated metadata) pointing to the
    same (changed) object, and discard the old `Ref`.
    - if a function does not have a **deterministic set of dependencies**
    it invokes for each given call, this may break the versioning system's
    invariants.
- **avoid long (e.g. > 50) chains of calls in queries**: you should keep your
  workflows relatively shallow for queries to be efficient. This means e.g. no
  long recursive chains of calling a function repeatedly on its output
- **don't rename anything (yet)**: there isn't good support yet for renaming
functions, or moving functions around files. It's possible to rename functions
and their arguments, but this is still undocumented.
- **deletion**: no interfaces are currently exposed for deleting results.
- **examine complex queries manually**: the color refinement algorithm used to
  extract a declarative query from a computational graph can in rare cases fail
  to realize that two vertices have different roles in the computational graph
  when projecting to the query. When in doubt, you should examine the printout
  of the query and tweak it if necessary. 

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala/blob/master/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala/blob/master/tutorials/01_logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- [dependency
  tracking](https://github.com/amakelov/mandala/blob/master/tutorials/02_dependencies.ipynb)
  tutorial

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
- **queries**:
  - all queries in `mandala` are [conjunctive queries](https://en.wikipedia.org/wiki/Conjunctive_query), a
    fundamental class of queries in relational algebra.
  - conjunctive queries are also related to category theory, see e.g.
    [here](https://blog.algebraicjulia.org/post/2020/12/cset-conjunctive-queries/). 
  - the [color refinement
    algorithm](https://en.wikipedia.org/wiki/Colour_refinement_algorithm) used
    to extract a query from an arbitrary computational graph is a standard tool
    for finding similar substructure in graphs and testing for graph
    isomorphism.
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
  - the [unison programming
    language](https://www.unison-lang.org/learn/the-big-idea/) represents
    functions by the hash of their content (syntax tree, to be exact).
