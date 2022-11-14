# Automate away the pain of experiment data management
Mandala eliminates the code and discipline required to manage data in
computational projects like machine learning / data science experiments. 

Mandala lets you write **only** the code expressing your computations (like you
would in a quick interactive session), but get the benefits of a storage that is
easy to add to, query, and evolve directly from the code of your experiments.

Mandala lite is a feature-limited but much more heavily tested version of the
[Mandala](https://github.com/amakelov/mandala) library.

## Installation
```
pip install git+https://github.com/amakelov/mandala_lite
```

## Features
- write the computational code only, without worrying about data management, and
  have the results automatically saved & queriable.
- use computational code as a flexible declarative query interface to its
  results, tapping into the power of SQL directly from plain Python
- modify and version computational primitives without worrying about updating
  past results: it just works
- delay computation to enable optimized batch processing without changing the
  code and having to bundle/unbundle data
- remote storage to enable collaboration between multiple users and machines

## Minimal example
```python
from mandala_lite.imports import *

# create a storage for results
storage = Storage()

@op # memoization decorator
def inc(x) -> int:
    return x + 1 

@op
def add(x: int, y: int) -> int:
    return x + y

with storage.run(): # calls inside `run` block are memoized
    for i in range(3):
        j = inc(i)

# add logic/parameters directly on top of memoized code without re-doing past work
with storage.run():
    for i in range(5):
        j = inc(i)
        k = add(j, i)

# pattern-match to memoized call graphs using code itself
with storage.query() as q: 
    i = Q() # placeholder for any value
    j = inc(i) # function calls create constraints
    k = add(j, i)
    # get a table where each row satisfies the constraints
    df = q.get_table(i.named('i'), j.named('j'), k.named('k')) 
df

>>>    i  j  k
>>> 0  0  1  1
>>> 1  1  2  3
>>> 2  2  3  5
>>> 3  3  4  7
>>> 4  4  5  9

# modify the code without worrying about updating past results
@op
def inc(x, how_many_times=1) -> int: # add a new argument
    return x + how_many_times

# still memoized
with storage.run():
    for i in range(5):
        j = inc(i)
        k = add(j, i)

# do work with the new code
with storage.run():
    for i in range(5):
        j = inc(i, how_many_times=2)
        k = add(j, i)

# ... 
```

## Tutorials 
- See [this logistic regression
notebook](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/demos/logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala!