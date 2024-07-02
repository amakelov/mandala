# Patterns for Incremental Computation & Development
**`@op`-decorated functions are designed to be composed** with one another. This
enables the same piece of imperative code to adapt to multiple goals depending
on the situation: 

- saving new `@op` calls and/or loading previous ones;
- cheaply resuming an `@op` program after a failure;
- incrementally adding more logic and computations to the same code without
re-doing work.

**This section of the documentation does not introduce new methods or classes**.
Instead, it demonstrates the programming patterns needed to make effective use
of `mandala`'s memoization capabilities.

## How `@op` encourages composition
There are several ways in which the `@op` decorator encourages (and even
enforces) composition of `@op`s:

- **`@op`s return special objects**, `Ref`s, which prevents accidentally calling 
a non-`@op` on the output of an `@op`
- If the inputs to an `@op` call are already `Ref`s, this **speeds up the cache
lookups**.
- If the call can be reused, the **input `Ref`s don't even need to be in memory**
(because the lookup is based only on `Ref` metadata).
- When `@op`s are composed, **computational history propagates** through this
composition. This is automatically leveraged by `ComputationFrame`s when
querying the storage.
- Though not documented here, **`@op`s can natively handle Python
collections** like lists and dicts. This 

When `@op`s are composed in this way, the entire computation becomes "end-to-end
[memoized](https://en.wikipedia.org/wiki/Memoization)". 

## Toy ML pipeline example
Here's a small example of a machine learning pipeline:


```python
from mandala.imports import *
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@op
def load_data(n_class=2):
    print("Loading data")
    return load_digits(n_class=n_class, return_X_y=True)

@op
def train_model(X, y, n_estimators=5):
    print("Training model")
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=2).fit(X, y)

@op
def get_acc(model, X, y):
    print("Getting accuracy")
    return round(accuracy_score(y_pred=model.predict(X), y_true=y), 2)

storage = Storage()

with storage:
    X, y = load_data() 
    model = train_model(X, y)
    acc = get_acc(model, X, y)
    print(acc)
```

    Loading data
    Training model
    Getting accuracy
    AtomRef(1.0, hid='d16...', cid='b67...')


## Retracing your steps with memoization
Running the computation again will not execute any calls, because it will
exactly **retrace** calls that happened in the past. Moreover, the retracing is
**lazy**: none of the values along the way are actually loaded from storage:


```python
with storage:
    X, y = load_data() 
    print(X, y)
    model = train_model(X, y)
    print(model)
    acc = get_acc(model, X, y)
    print(acc)
```

    AtomRef(hid='d0f...', cid='908...', in_memory=False) AtomRef(hid='f1a...', cid='69f...', in_memory=False)
    AtomRef(hid='caf...', cid='87a...', in_memory=False)
    AtomRef(hid='d16...', cid='b67...', in_memory=False)


This puts all the `Ref`s along the way in your local variables (as if you've
just ran the computation), which lets you easily inspect any intermediate
variables in this `@op` composition:


```python
storage.unwrap(acc)
```




    1.0



## Adding new calls "in-place" in `@op`-based programs
With `mandala`, you don't need to think about what's already been computed and
split up code based on that. All past results are automatically reused, so you can
directly build upon the existing composition of `@op`s when you want to add new
functions and/or run old ones with different parameters:


```python
# reuse the previous code to loop over more values of n_class and n_estimators 
with storage:
    for n_class in (2, 5,):
        X, y = load_data(n_class) 
        for n_estimators in (5, 10):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            print(acc)
```

    AtomRef(hid='d16...', cid='b67...', in_memory=False)
    Training model
    Getting accuracy
    AtomRef(0.99, hid='6fd...', cid='12a...')
    Loading data
    Training model
    Getting accuracy
    AtomRef(0.8, hid='158...', cid='f0a...')
    Training model
    Getting accuracy
    AtomRef(0.87, hid='214...', cid='fce...')


Note that the first value of `acc` from the nested loop is with
`in_memory=False`, because it was reused from the call we did before; the other
values are in memory, as they were freshly computed. 

This pattern lets you incrementally build towards the final computations you
want without worrying about how results will be reused.

## Using control flow efficiently with `@op`s
Because the unit of storage is the function call (as opposed to an entire script
or notebook), you can transparently use Pythonic control flow. If the control
flow depends on a `Ref`, you can explicitly load just this `Ref` in memory
using `storage.unwrap`:


```python
with storage:
    for n_class in (2, 5,):
        X, y = load_data(n_class) 
        for n_estimators in (5, 10):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            if storage.unwrap(acc) > 0.9: # load only the `Ref`s needed for control flow
                print(n_class, n_estimators, storage.unwrap(acc))
```

    2 5 1.0
    2 10 0.99


## Memoized code as storage interface
An end-to-end memoized composition of `@op`s is like an "imperative" storage
interface. You can modify the code to only focus on particular results of
interest:


```python
with storage:
    for n_class in (5,):
        X, y = load_data(n_class) 
        for n_estimators in (5,):
            model = train_model(X, y, n_estimators=n_estimators)
            acc = get_acc(model, X, y)
            print(storage.unwrap(acc), storage.unwrap(model))
```

    0.8 RandomForestClassifier(max_depth=2, n_estimators=5)

