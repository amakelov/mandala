# Mandala: self-managing experiments

[install](#install) | [quickstart](#quick-start) | [docs](https://amakelov.github.io/mandala/index.html)

## What is Mandala?
Mandala enables new, simpler patterns for working with complex and evolving
computational experiments. 

It eliminates low-level code and decisions for how to save, load, query,
delete and otherwise organize results. To achieve this, it lets computational
code "manage itself" by organizing and addressing its own data storage.

## Features at a glance
- **concise**: code computations in pure Python (w/ control flow, collections,
  ...) -- results are automatically tracked and queriable
- **iterate rapidly**: add/edit parameters/logic and rerun code -- past results
  are loaded on demand, and only new computations are executed
- **pattern-match against Python code**: query across complex, branching
  projects by reusing computational code itself

## Install
```console
pip install git+https://github.com/amakelov/mandala
```

## Quick start
### Recommended introductions
To build some understanding, check these out:
- 2-minute introduction: [intro to self-managing code](https://amakelov.github.io/mandala/intros/two_mins.html)
- 10-minute introduction: [manage a toy ML project](https://amakelov.github.io/mandala/intros/ten_mins.html)

### Minimal working examples
If you want to jump right into code, below are a few minimal, somewhat
interesting examples to play with and extend:
```python
from typing import List
from mandala.all import *
set_logging_level('warning')

# create a storage for results
storage = Storage(in_memory=True) # can also be persistent (on disk)

@op(storage) # memoization decorator
def inc(x) -> int:
    return x + 1 

@op(storage) 
def mean(x:List[int]) -> float: 
    # you can operate on / return collections of memoized results
    return sum(x) / len(x) 

with run(storage): # calls inside `run` block are memoized
    nums = [inc(i) for i in range(5)]
    result = mean(nums) # memoization composes through lists without copying data
    print(f'Mean of 5 nums: {result}')

# add logic/parameters directly on top of memoized code without re-doing past work
with run(storage, lazy=True):
    nums = [inc(i) for i in range(10)]
    result = mean(nums) 

# walk over chains of calls without loading intermediate data 
# to traverse storage and collect results flexibly
with run(storage, lazy=True):
    nums = [inc(i) for i in range(10)]
    result = mean(nums) 
    print(f'Reference to mean of 10 nums: {result}')
    storage.attach(result) # load the value in-place 
    print(f'Loaded mean of 10 nums: {result}')

# pattern-match to memoized compositions of calls
with query(storage) as q: 
    # this may not make sense unless you read the tutorials
    i = Query()
    inc_i = inc(i).named('inc_i')
    nums = MakeList(containing=inc_i, at_index=0).named('nums')
    result = mean(nums).named('result')
    df = q.get_table(inc_i, nums, result)
df
```

### Interested in seeing Mandala used in computational projects?
If you, too, find the programming patterns enabled by Mandala interesting and
useful, here are some things you can do to help:
- open an issue/discussion about use cases from your projects where Mandala can
  be a good (or bad!) fit
- share this repository with other people who might find it useful

### Next steps
See the [docs](https://amakelov.github.io/mandala/index.html) for more
information and examples on how to use Mandala. 