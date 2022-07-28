# Contexts: for and against
## Motivation for contexts and overview
Currently, `mandala_lite` is very simple, and contexts don't seem to do much
besides setting commit boundaries. However, in the full `mandala`, they are used
for a bunch of things.

### What contexts are used for 
Three broad categories:
- apply configuration options influencing how function calls are interpreted to
  an entire block of code. For example,
    -  lazy loading (though this can be applied to all code by default anyway), 
    -  setting and changing the mode of execution (`run`, `query`, `delete`).
       Note that `delete` mode marks each call it encounters for deletion (with
       more details explained below).
    -  whether to force recomputation of transient objects (see below)
- perform actions automatically (e.g., commit, deletion) before/after a
  block of code
- encode some combinatorial structure (this is used for branching queries, and
  more advanced than the other use cases)

### Why contexts are a good choice
Contexts are a natural way to achieve all these things in a 
- **recursive** (i.e. nested), 
- **cascading** (i.e. only apply the diff when branching into a sub-context), and
- **readable** (indentation, concision)

way.

### Problems with parallelism
Ideally, you'd want to be able to parallelize some of the function calls in a
memoized workflow with minimum effort. Tools like `dask` and `ray` are an
example of how simple this can be made when there's no data management involved.
We should be able to get the same level of simplicity with mandala. The problem
is that there's no longer a single shared context object when there are multiple
processes. In the full `mandala`, this is fine because each process writes its
outputs and call object to the file system immediately. 

Here, we'd have to do something different! 

TODO - understand design constraints

### Alternatives to contexts: good and bad
In many of these cases, you can replace the use of contexts with something else.
That's good because it's (maybe) more understandable. But, this can lead to some
clumsiness:
- in general, you have to write more code and think more; contexts are good at
  naturally doing what you would do
- sometimes, you have to remember to do something before *and* after the block of
  code (e.g., revert the value of a setting)
- sometimes you have to collect all objects computed in a block of code (e.g. to
  mark them for deletion) as opposed to just put the entire block in a context

Below are discussions of various uses of contexts and how we can do without
them.

## Performing actions at context boundaries
It's convenient that certain things, like committing, pre-loading data and/or
syncing, are automatically performed when entering/leaving a context. 

### Alternatives
It's obvious what to do instead, but more verbose and error-prone (imagine you
forget to commit and/or sync at the end of your work...). Making each function
call commit is clearly not efficient.

## Making sure you don't make new calls
That's an example of applying a setting to all calls captured in a context.
Sometimes you want to verify that some workflow has been memoized end to end,
but don't want to trigger any new computation. You can do this easily using the
`allow_calls=False` context option:
```python
with run(storage):
    y = f(x)
    z = g(y)

with run(storage, allow_calls=False):
    y = f(x)
    z = g(y)
    w = h(z) # reaching this point will raise an error!
```
and that's it.

### Alternatives
You can just turn some global configuration option on/off when doing this - not
too bad, as long as you don't have to switch it on and off all the time (which
would be twice as much code, but many more times error-prone).

## "Imperative deletions"
### Notes on deletion
Deletion is a bit tricky, because you must take care to leave the storage in a
"good" state. What "good" means depends on how much missingness you are willing
to tolerate. There are two main dimensions to evaluate this on:
- **maintaining the history of each value**: if you want every value in storage
to be "explainable" via a chain of calls to "initial" values (explicitly
supplied by the user), then deleting a value must also delete all calls and
values that depend on it (hence the need for a quick way to figure what these
are, which is what the provenance table is used for).
- **maintaining memoization of entire calls**: if you want a call to either be
  fully memoized or not, it doesn't make sense to delete only one of its
  outputs. So if you adopt this, **the natural unit of deletion is a call**.

**Important note**: whether you assign output UIDs via content hashing or causal
hashing has implications for deletion. 

### How it works with contexts
Context nesting offers a particularly simple way to do "imperative" deletion by
pointing to the code whose results you want to delete (like an "undo" button for
code). For example, suppose you memoized a piece of code like this:
```python
with run(storage):
    preprocess = True
    X, y = load_data()
    if preprocess:
        X = preprocess_data(X)
    for alpha in (0.1, 0.2):
        model = train_v2(X, y, alpha=alpha)
        result = analyze_model(model)
```
and now want to delete the `model` and `result` variables for both values of
`alpha`. You can do this by nesting the relevant code in a `delete` context:
```python
with run(storage):
    preprocess = True
    X, y = load_data()
    if preprocess:
        X = preprocess_data(X)
    with delete():
        for alpha in (0.1, 0.2):
            model = train_v2(X, y, alpha=alpha)
            result = analyze_model(model)
```
which takes care of deleting all the calls captured in this block of code (and
consequently their outputs and any values that depend on them). This is
implemented by collecting all the call UIDs in a local variable while retracing,
and actually deleting these calls (and all dependents) when exiting the context.

### Alternatives
You could explicitly pick which values to delete, which would delete the calls
that generate them (or just the values, if there are no calls). This could mean
using a `delete` statement in several places instead of one (or grouping things
in some sort of collection)

## Transient results
Transient results are a feature in `mandala` that allows you to mark function
outputs in order to prevent saving them to storage. For a motivating example,
```python
@op
def inc_large_array(large_array:Array) -> Array:
    return AsTransient(large_array + 1)
```
is a function that increments a huge array and works like an ordinary `op`,
except that there is no value stored for the output in storage (only a UID). 

### The problem
The problem now is that, sometimes you want to re-execute the function in order
to re-create the object (because it is needed by some computation), and other
times you want to retrace through the call because the value is not really
needed. 

### How contexts deal with this
Consider for example this (artificial) scenario:
```python
with run(context):
    x_inc = inc_large_array(x)
    final = f(x, x_inc) 
```
The first time you run this, it'll go through fine. Now imagine you want to add
some more computation on `x_inc`:
```python
with run(context):
    x_inc = inc_large_array(x)
    final = f(x, x_inc) 
    more_final = g(x_inc)
```
At the time you hit the call to `inc_large_array`, you have no way of knowing
whether you'll need the actual output or just a pointer to it. So by default you
don't recompute and hope that everything will go fine. To enforce computation,
you turn on the `force` setting:
```python
with run(context) as c:
    with c(force=True):
        x_inc = inc_large_array(x)
    final = f(x, x_inc) 
    more_final = g(x_inc)
```

### Alternatives
You can just add a special function argument indicating you want to force
recomputation: 
```python
with run(context) as c:
    x_inc = inc_large_array(x, __force__=True)
    final = f(x, x_inc) 
    more_final = g(x_inc)
```
This seems to be a good solution, since there's no reason to expect there to be
many consecutive forced calls (which is what a context would be very natural
for). 

## Switching between `run` and `query` mode 
You can nest queries inside retraced code in order to pin down some values
precisely, without having to figure out how to get a handle on them via UIDs or
whatever. E.g.,
```python
with run(storage):
    best_model = train(X, y, alpha=0.23, beta=0.42)
    with query(storage) as q:
        thing = thing_we_did_with_model(model=best_model)
        ...
```
and have this be interpreted in the obvious way (where the `best_model` is
constrained to exactly equal the result of the call to `train`). 

### Alternatives
As you point out, we can have much more liberal rules for forming queries that
allow us to do the above without any need to nest contexts. Instead, each
function looks at its inputs, and if any of them is a query object, it will
return a query object too - I like that! 