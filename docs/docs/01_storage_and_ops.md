# `Storage` & the `@op` Decorator
<a href="https://colab.research.google.com/github/amakelov/mandala/blob/master/mandala/docs_notebooks/01_storage_and_ops.ipynb"> 
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

A `Storage` object holds all data (saved calls, code and dependencies) for a
collection of memoized functions. In a given project, you should have just one
`Storage` and many `@op`s connected to it. This way, the calls to memoized
functions create a queriable web of interlinked objects.  


```python
# for Google Colab
try:
    import google.colab
    !pip install git+https://github.com/amakelov/mandala
except:
    pass
```

## Creating a `Storage`

When creating a storage, you must decide if it will be in-memory or persisted on
disk, and whether the storage will automatically version the `@op`s used with
it:


```python
from mandala.imports import Storage
import os

DB_PATH = 'my_persistent_storage.db'
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

storage = Storage(
    # omit for an in-memory storage
    db_path=DB_PATH,
    # omit to disable automatic dependency tracking & versioning
    # use "__main__" to only track functions defined in the current session
    deps_path='__main__', 
)
```

## Creating `@op`s and saving calls to them
**Any Python function can be decorated with `@op`**:


```python
from mandala.imports import op

@op 
def sum_args(a, *args, b=1, **kwargs):
    return a + sum(args) + b + sum(kwargs.values())
```

In general, calling `sum_args` will behave as if the `@op` decorator is not
there. `@op`-decorated functions will interact with a `Storage` instance **only
when** called inside a `with storage:` block:


```python
with storage: # all `@op` calls inside this block use `storage`
    s = sum_args(6, 7, 8, 9, c=11,)
    print(s)
```

    AtomRef(42, hid='168...', cid='d92...')


This code runs the call to `sum_args`, and saves the inputs and outputs in the
`storage` object, so that doing the same call later will directly load the saved
outputs.

### When should something be an `@op`?
As a general guide, you should make something an `@op` if you want to save its
outputs, e.g. if they take a long time to compute but you need them for later
analysis. Since `@op` [encourages
composition](https://amakelov.github.io/mandala/02_retracing/#how-op-encourages-composition),
you should aim to have `@op`s work on the outputs of other `@op`s, or on the
[collections and/or items](https://amakelov.github.io/mandala/05_collections/)
of outputs of other `@op`s.

## Working with `@op` outputs (`Ref`s)
The objects (e.g. `s`) returned by `@op`s are always instances of a subclass of
`Ref` (e.g., `AtomRef`), i.e.  **references to objects in the storage**. Every
`Ref` contains two metadata fields:

- `cid`: a hash of the **content** of the object
- `hid`: a hash of the **computational history** of the object, which is the precise
composition of `@op`s that created this ref.  

Two `Ref`s with the same `cid` may have different `hid`s, and `hid` is the
unique identifier of `Ref`s in the storage. However, only 1 copy per unique
`cid` is stored to avoid duplication in the storage.

### `Ref`s can be in memory or not
Additionally, `Ref`s have the `in_memory` property, which indicates if the
underlying object is present in the `Ref` or if this is a "lazy" `Ref` which
only contains metadata. **`Ref`s are only loaded in memory when needed for a new
call to an `@op`**. For example, re-running the last code block:


```python
with storage: 
    s = sum_args(6, 7, 8, 9, c=11,)
    print(s)
```

    AtomRef(hid='168...', cid='d92...', in_memory=False)


To get the object wrapped by a `Ref`, call `storage.unwrap`:


```python
storage.unwrap(s) # loads from storage only if necessary
```




    42



### Other useful `Storage` methods

- `Storage.attach(inplace: bool)`: like `unwrap`, but puts the objects in the
`Ref`s if they are not in-memory.
- `Storage.load_ref(hid: str, in_memory: bool)`: load a `Ref` by its history ID,
optionally also loading the underlying object.


```python
print(storage.attach(obj=s, inplace=False))
print(storage.load_ref(s.hid))
```

    AtomRef(42, hid='168...', cid='d92...')
    AtomRef(42, hid='168...', cid='d92...')


## Working with `Call` objects
Besides `Ref`s, the other kind of object in the storage is the `Call`, which
stores references to the inputs and outputs of a call to an `@op`, together with
metadata that mirrors the `Ref` metadata:

- `Call.cid`: a content ID for the call, based on the `@op`'s identity, its
version at the time of the call, and the `cid`s of the inputs
- `Call.hid`: a history ID for the call, the same as `Call.cid`, but using the 
`hid`s of the inputs.

**For every `Ref` history ID, there's at most one `Call` that has an output with
this history ID**, and if it exists, this call can be found by calling
`storage.get_ref_creator()`: 


```python
call = storage.get_ref_creator(ref=s)
print(call)
display(call.inputs)
display(call.outputs)
```

    Call(sum_args, cid='e91...', hid='f99...')



    {'a': AtomRef(hid='c6a...', cid='489...', in_memory=False),
     'args_0': AtomRef(hid='e0f...', cid='2da...', in_memory=False),
     'args_1': AtomRef(hid='479...', cid='f2c...', in_memory=False),
     'args_2': AtomRef(hid='c37...', cid='81a...', in_memory=False),
     'b': AtomRef(hid='610...', cid='366...', in_memory=False),
     'c': AtomRef(hid='a33...', cid='c84...', in_memory=False)}



    {'output_0': AtomRef(hid='168...', cid='d92...', in_memory=False)}

