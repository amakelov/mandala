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

# Mandala

<div align="left">
<a href="https://colab.research.google.com/github/amakelov/mandala/blob/master/mandala/tutorials/00_hello.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</div>

Mandala is a
[`functools.lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache)
on steroids, applied to elegantly solve experiment data management.

It turns Python function calls into composable, interlinked, queriable data that
is automatically co-evolved with your codebase. By applying different semantics
to this data, the same piece of ordinary Python code can be used to not only
compute, but also save, load, query, delete, and batch-compute computational
artifacts. This unlocks extremely flexible yet simple patterns of data
management in complex projects.

## Features
- [simple interface and usage](#easily-add-composable-memoization-to-existing-code): decorate functions with `@op` to
  memoize them, and put ordinary Python code in `storage.run()`, `storage.query()`,
  ... blocks depending on what you want to do. The rest is just Python.
- [rapid iteration](#iterate-rapidly-without-redoing-work) without redoing work,
  great for notebooks and other interactive environments
- [query by pattern-matching Python code](#query-by-directly-pattern-matching-python-code): produce tables of results
  directly from Python code encoding their relationships.
- [modify memoized functions](#modify-memoized-functions-without-forgetting-old-calls) seamlessly without
  forgetting old calls
- [automatic function-level dependency tracking](#function-level-dependency-tracking): get
  (optional) alerts when a function's dependencies change, and decide whether to
  recompute memoized calls.
- [use Python collections in Python-native ways](#use-python-collections-in-python-native-ways): store and track elements of
  Python collections separately, enabling Pythonic code and incremental
  computation.

## Install
```
pip install git+https://github.com/amakelov/mandala
```

## Video walkthroughs

### Easily add *composable* memoization to existing code
Decorate functions with `@op` and annotate the number of return values (with a
`typing.Tuple` for functions returning multiple values). Memoization writes each
object to storage once, even if Below is a simple
example with a `scikit-learn` pipeline:

<details closed><summary><kbd>Show/hide gif</kbd></summary>
<p>

![01_memoization](https://user-images.githubusercontent.com/1467702/210118002-4d2418a3-5d34-42f4-bf49-8a0522b788b1.gif)
</p>
</details>

### Iterate rapidly without redoing work
`mandala`'s memoization is designed to be composed across functions (and [data
structures](#data-structures)). This makes it straightforward to interact with
and grow a project:
- add new parameters and functions directly on top of a piece of memoized code
  to do new work;
- retrace existing memoized code to imperatively query results

<details closed><summary><kbd>Show/hide gif</kbd></summary>
<p>

![02_iteration](https://user-images.githubusercontent.com/1467702/210118075-f48501ab-ba13-473f-a8fe-0fd2d555b9e1.gif)
</p>
</details>

### Query by directly pattern-matching Python code
Sometimes, retracing memoized code is not enough to query a project; you need a
global view of all experiments matching some condition. To this end, you can
define rich declarative queries by directly using the code of your experiments.
In a `.query()` context, function calls are interpeted as building a graph of
**computational dependencies between variables**. Calling `get_table(...)` on
the context manager gives you tables of all the tuples in the storage satisfying 
these dependencies:

<details closed><summary><kbd> Show/hide gif </kbd></summary>
<p>

![03_queries](https://user-images.githubusercontent.com/1467702/210118099-0fcbfb60-cc02-438b-b975-3e335558d8d1.gif)
</p>
</details>

### Modify memoized functions without forgetting old calls
You can modify memoized functions without losing past calls by adding extra
arguments with default values. All past calls are treated as if they used this
default value. This is very convenient in machine learning and data science
scenarios, for e.g. exposing hard-coded constants as parameters, or adding new
behaviors to an algorithm:

<details closed><summary><kbd>Show/hide gif</kbd></summary>
<p>

![04_add_input](https://user-images.githubusercontent.com/1467702/210118150-f8abd146-9b3e-4987-9ac2-782be8c4f856.gif)
</p>
</details>

### Use Python collections in Python-native ways
Often, collections like lists, dicts and sets naturally come up in experiments.
Think of a clustering algorithm returning an unknown number of clusters,
ensembling multiple ML models, or picking the best element out of some
collection. It's desirable to be able to save, load and query elements of these
collections separately, as well as keep track of the relationship between
a collection and its elements. 

`mandala` allows you to do this by **using Python’s own collections in
Python-native ways** in your computations and queries. Below you can see a tiny
demo of this: a do-it-yourself random forest classifier where you directly pass
a list of trees to the evaluation function. Each tree is saved separately, which
means you can freely vary the number of trees in the list while storing each
tree only once, and you get to reuse past tree trainings. The relationship
between individual trees and lists of trees propagates through the declarative
query interface:

<details closed><summary><kbd>Show/hide gif</kbd></summary>
<p>

![05_data_structures](https://user-images.githubusercontent.com/1467702/210394133-8533e0ec-6f30-43ab-bbad-3facc3f6f909.gif)
</p>
</details>

### Function-level dependency tracking
It can be annoying and error-prone to manually keep track of whether a new code
change makes a memoized result "stale". `mandala` streamlines this process
through a very simple interface! 

Behind the scenes, it tracks the dependencies of memoized functions **on the
level of individual functions and global variables**. It presents you with a
diff of changes, alongside the memoized functions affected by each change, and
lets you decide whether to ignore the change (keeping the old memoized results),
or create new versions of the functions (hence re-computing memoized calls):

<details closed><summary><kbd>Show/hide gif</kbd></summary>
<p>

![06_dependencies](https://user-images.githubusercontent.com/1467702/210449697-3bd8fa87-38f1-4755-9a48-84cc9a1d2ad7.gif)
</p>
</details>


## Usage
This is a quick guide on how to get up to speed with the core features and avoid
common pitfalls.

### `Storage` and the `@op` decorator
A `Storage` instance holds all the data (saved calls and metadata) for a
collection of memoized functions. In a given project, you should have just one
`Storage` and many memoized functions connected to it. This way, the calls to
memoized functions create a web of interlinked objects. 

```python
from mandala.all import Storage, op

storage = Storage(
  db_path='my_persistent_storage.db', # omit for an in-memory storage
  deps_root='codebase_root_folder/', # omit to disable automatic dependency tracking
)
```
The `@op` decorator marks a function `f` as memoizable. Some notes apply:
- `f` must have a **fixed number of arguments** (defaults are allowed)
- `f` must have a **fixed number of return values**, and this must be annotated in
  the function signature **or** declared as `@op(nout=...)`
- the `list`, `dict` and `set` collections, when used as argument/return
  annotations, cause elements of these collections to be stored separately; [see
  below for more details](#python-collections-list-dict-set). To avoid
  confusion, `tuple`s are reserved for specifying the number of outputs.

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

@op # memoization decorator
def load_data(n_class: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    return load_digits(n_class=n_class, return_X_y=True)

@op(nout=1) # declare the number of outputs manually
def train_model(X: np.ndarray, y: np.ndarray, n_estimators:int = 5):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=2).fit(X, y)
```

**Calling an `@op`-decorated function "normally" does not memoize**. To actually
put data in the storage, you must put calls inside a `with storage.run():`
block.

### Compute & memoize: `storage.run()` blocks
**`@op`-decorated functions are designed to be composed** with one another
inside `storage.run()` blocks. This composability lets
you use the same piece of ordinary Python code to compute, save, load, *and any
combination of the three*:
```python
# generate the dataset. This saves `X, y` to storage.
with storage.run():
    X, y = load_data()

# later, train a model by directly adding on top of this code. `load_data` is
# not computed again!
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
            model = train_model(X, y)
            acc = get_acc(model, X, y)
            print(acc)
  
```
Memoized functions return something-`Ref` instances (`ValueRef`, `ListRef`,
...), which bundle the actual return value with storage metadata. To get the
return value itself, use `unwrap`. It works recursively on collections (lists,
dicts, sets, tuples) as well. 

You can **imperatively query** storage just by retracing some code that's been
entirely memoized:
```python
# later: use composable memoization as imperative query interface!
with storage.run():
    X, y = load_data(5)
    for n_estimators in (5, 20):
        model = train_model(X, y)
        acc = get_acc(model, X, y)
        print(acc)
```
### Query: `storage.query()` blocks
**Put (mostly) the same code in a `storage.query()` block** to get a
**declarative** query interface to storage. 
```python
with storage.query() as q:
    n_class = Q() # creates a variable that can match values in storage
    # @op calls create constraintss between the values variables can match
    X, y = load_data(n_class)
    model = train_model(X, y, n_estimators=5)
    acc = get_acc(model, X, y)
    # get a table where each row is a matching of the given variables
    # that satisfies the constraints
    df = q.get_table(n_class, acc) 
```
**How the `.query()` context works**: 
- a query variable, generated with `Q()` or as return value from an `@op`
  (like `X`, `y`, ... above), can in
principle match any value in the storage. 
- A "raw" value (like `5` above) creates a variable that can only match this raw value
- You can **omit** even required function arguments. This leaves them unconstrained.
- by chaining together calls to `@op`s, you impose constraints between the
  inputs and outputs to the op. For exampe, `X, y = load_data(n_class)` imposes
  the constraint that a matching of values `(a, b, c)` to `(n_class, X, y)` must
  satisfy `b, c = load_data(a)`. 
- the `get_table` method takes any sequence of variables, and returns a table
where each row is a matching of values to the respective variables that
satisfies **all** the constraints.

**Warning**: if your query is not sufficiently constrained, there may be a
combinatorial explosion of results!

### Python collections: `list`, `dict`, `set`
You can separately store and query the elements of Python's `list`s,
string-keyed `dict`s, and `set`s if wanted. This lets you write simple and
Pythonic code, while reaping benefits like incremental computation/writes/reads,
and sharing common substructure without duplicating storage. To get this
behavior, simply annotate the arguments or returns of an `@op` with the
respective collection:
```python
@op(nout=1)
def get_ensemble_prediction(models: list, X):
    return majority_vote([model.predict(X) for model in models])

with storage.run():
    X, y = load_data()
    model_1 = train_alg_1(X, y)
    model_2 = train_alg_2(X, y)
    model_3 = train_alg_3(X, y)
    with_two = get_ensemble_prediction([model_1, model_2], X)
    with_three = get_ensemble_prediction([model_1, model_2, model_3], X)
```
You can also incorporate these collections in declarative queries in Pythonic
ways. For example, you can issue a query to find all the ensembles some model
was a part of:
```python
with storage.query() as q:
    X, y = load_data()
    model_1 = train_alg_1(X, y)
    models = Q([model_1, ...]) # `[x, ...]` matches a list containing `x`
    some_model = models[Q()] # this will match to any element of `models`
    predictions = get_ensemble_prediction(models)
```

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala/blob/master/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala/blob/master/tutorials/01_logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- [dependency tracking](https://github.com/amakelov/mandala/blob/master/tutorials/02_dependencies.ipynb) tutorial

## Why mandala?
In a world of ideal interfaces, there would be nothing between our thoughts and
their expression on a computer. Towards this vision, we believe that the best
tools are the ones that mostly stay invisible, and let you get closer to this
"speed of thought" ideal. This is the main design principle behind `mandala`:
that it is possible, and superior, for the management of computational artifacts
to be built into the native semantics of the computational medium itself. 
