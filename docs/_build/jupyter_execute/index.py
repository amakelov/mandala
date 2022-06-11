#!/usr/bin/env python
# coding: utf-8

# # Mandala: concise experiment management
# ## What is Mandala?
# Mandala enables new, simpler, and more concise patterns for working with complex
# and evolving computational experiments. 
# 
# It eliminates low-level code and decisions for how to save, load, query,
# delete and otherwise organize results. To achieve this, it lets computational
# code "manage itself" by organizing and addressing its own data storage.
# 
# ```{admonition} Under construction
# :class: warning
# This project is under active development
# ```
# 
# ### Features at a glance
# - **concise**: code computations in pure Python (w/ control flow, collections,
#   ...) -- results are automatically tracked and queriable
# - **iterate rapidly**: add/edit parameters/logic and rerun code -- past results
#   are loaded on demand, and only new computations are executed
# - **pattern-match against Python code**: query across complex, branching
#   projects by reusing computational code itself

# ### Quick start
# #### Installation
# ```console
# pip install git+https://github.com/amakelov/mandala
# ```
# #### Recommended introductions
# To build some understanding, check these out:
# - 2-minute introduction: [intro to self-managing code](2mins)
# - 10-minute introduction: [manage a small ML project](10mins)
# #### Minimal working examples
# If you want to jump right into code, below are a few minimal, somewhat
# interesting examples to play with and extend:

# In[1]:


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


# ## Why Mandala?
# ### Advantages
# Compared to other tools for tracking and managing computations, the features that
# most set Mandala apart are the direct and concise patterns in which complex
# Python code can interact with its own storage. This manifests in several ways: 
# - **Python code as interface to its own storage**: you just write the code to compute
#   what you want to compute (freely using Python's control flow and collections),
#   and directly add more parameters and logic to it over time. Mandala takes
#   care of the rest:
#   - **the organization of storage mirrors the structure of code**, and Mandala
#   provides you with the tools to make maximum use of this --
#   retracing memoized code with on-demand data loading, and declarative
#   code-based pattern-matching.
#   - this leads to **simple, intuitive and flexible ways to query and iterate on 
#   experiments**, even when their logic gets quite complex -- without any data
#   organization efforts on your part.
#   - it also allows you to **query relationships between any variables in your
#   projects**, even when they are separated by many computational steps -- **without
#   explicitly annotating these relationships**.
# - **refactor code and data will follow**: Mandala makes it easy to apply
#   familiar software refactorings to code *without* losing the relationship to
#   this code's existing results. This gives you high-level tools to manage the
#   complexity of both the code and its data as the project grows.  
# - **organize all results and their relationships**: Mandala manages all the
#   artifacts produced by computations, not just a set of human-readable
#   metrics. It lets you use pure Python idioms to
#     - compute with **data structures with shared substructure**
#     - **index and view data in multiple ways** and on multiple levels of analysis
# 
#   without storage duplication. This gives you much flexibility in manipulating
#   the contents of storage to express your intent.
# 
# ### Comparisons
# Mandala takes inspiration from many other programming tools and concepts. Below
# is an (incomplete but growing) list of comparisons with relevant tools:
# - [algebraicjulia](https://www.algebraicjulia.org/):
#   [conjunctive](https://www.algebraicjulia.org/blog/post/2020/12/cset-conjunctive-queries/) [queries](https://www.algebraicjulia.org/blog/post/2020/11/sql-as-hypergraph/)
#   are integral to Mandala's declarative interface, and are generalized in
#   several ways to make them practical for complex experiments:
#     - a single table of values is used to enable polymorphism
#     - operations on lists/dicts are integrated with query construction
#     - queries can use the hierarchical structure of computations
#     - constraints can be partitioned (to avoid interaction) while using some
#       shared base (to enable code reuse)
#     - dynamic query generation can use conditionals to enable disjunctive
#       queries, and even loops (though this quickly becomes inefficient)
# - [koji](https://arxiv.org/abs/1901.01908) and [content-addressable computation](https://research.protocol.ai/publications/ipfs-fan-a-function-addressable-computation-network/delarocha2021a.pdf):
#   Mandala uses causal hashing to 
#   - ensure correct, deterministic and idempotent behavior;
#   - avoid hashing large (or unhashable) Python objects;
#   - avoid discrepancies between object hashes across library versions 
# 
#   Mandala can be thought of as a single-node, Python-only implementation of
#   general-purpose content-addressable computation with two extra features:
#     - hierarchical organization of computation,
#     - declarative queries
# - [funsies](https://github.com/aspuru-guzik-group/funsies) is a workflow engine
#   for Python scripts that also uses causal hashing. Mandala differs by
#   integrating more closely with Python (by using functions instead of scripts as
#   the units of work), and thus enabling more fine-grained control and
#   expressiveness over what gets computed and how. 
# - [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html#memory)
#   implements persistent memoization for Python functions that overcomes some of
#   the issues naive implementations have with large and complex Python objects.
#   Mandala augments `joblib.Memory` in some key ways:
#   - memoized calls can be queried/deleted declaratively
#   - collections and memoized functions calling other memoized functions can
#     reuse storage
#   - you can modify and refactor memoized functions while retaining connection to
#     memoized calls
#   - you can avoid the latency of hashing large/complex objects 
# - [incpy](https://dl.acm.org/doi/abs/10.1145/2001420.2001455?casa_token=ahM2UC4Uk-4AAAAA:9lZXVDS7nYEHzHPJk-UCTOAICGb2astAh2hrL00VB125nF6IGG90OwA-ujbe-cIg2hT4T1MOpbE2)
#   augments the Python interpreter with automatic persistent memoization. Mandala
#   also enables automatic persistent memoization, but it is different from
#   `incpy` in some key ways:
#   - uses decorators to explicitly designate memoized functions (which can be
#     good or bad depending on your goals)
#   - allows for lazy retracing of memoized calls
#   - provides additional features like the ones mentioned in the comparison with
#     `joblib.Memory`
# 
# ### Philosophy
# When can we declare data management for computational experiments a solved
# problem? It's unclear how to turn this question into a measurable goal, but
# there is a somewhat objective *lower bound* on how simple data management can
# get:
# 
# > At the end of the day, we have to *at least* write down the (Python) code to express
# > the computations we want to run, *regardless* of data management concerns. 
# > Can this be *all* the code we have to write, and *still* be able to achieve
# > the goals of data management?
# 
# Mandala aims to bring us to this idealized lower bound. It adopts the view that
# Python itself is flexible and expressive enough to capture our intentions about
# experiments. There shouldn't be a ton of extra interfaces, concepts and syntax
# between your thoughts, their expression in code, and its results.
# 
# By mirroring the structure of computational code in the organization of data,
# and harmoniously extending Python's tools for capturing intention and managing
# complexity, we can achieve a more flexible, natural and immediate way to
# interact with computations.
# 
# This echoes the design goals of some other tools. For example,
# [dask](https://dask.org) and [ray](https://ray.io) (both of which Mandala
# integrates with) aim to let you write Python code the way you are used to, and
# take care of parallelization for you.

# ## Limitations 
# This project is under active development, and not ready for production. Its goal
# so far has been to demonstrate that certain high-level programming patterns are
# viable by building a sufficiently useful working prototype. Limitations can be
# summarized as follows:
# - it is easy to get started, but effective use in complex projects requires some
#   getting used to;
# - much of the code does what it does in very simple and often inefficient ways;
# - interfaces and (more importantly) storage formats may change in backward
#   incompatible ways.
# - bugs likely still exist;
# 
# That being said, Mandala is already quite usable in many practical situations.
# Below is a detailed outline of current limitations you should be aware of if you
# consider using this library in your work. 
# 
# ### "Missing" features
# There are some things you may be used to seeing in projects like this that
# currently don't exist:
# - **functions over scripts**: Mandala focuses on functions as the basic
#   building blocks of experiments as opposed to Python scripts. There is no
#   fundamental conceptual distinction between the two, but:
#   - functions provide a better-behaved interface, especially when it comes to
#     typing, refactoring, and hierarchical organization
#   - using functions makes it much easier to use 
#     projects such as [ray](https://www.ray.io/) and [dask](https://dask.org/)
#     alongside Mandala
#   - if you don't need to do something extra complicated involving different
#   Python processes or virtual environments, it is easy to wrap a script as a
#   function that takes in some settings and resource descriptions (e.g., paths to
#   input files) and returns other resource descriptions (e.g., paths to output
#   files).  However, the burden of refactoring the script's interface manually
#   and organizing its input/output resources would still be on you. So, always
#   use a function where you can.
# - **no integration with git**: version control data is not automatically
#   included in Mandala's records at this point, thought this would be an easy
#   addition. There are other programming patterns available for working with
#   multiple versions of code.
# - **no GUI**: for now, the library leans heavily towards using computational
#   code itself as a highly programmable interface to results, and visualization
#   is left to other tools. 
# 
# ### Acquiring best practices
# Using some features effectively requires deeper understanding:
# - **declarative queries**: It's possible to create underconstrained
#   pattern-matching queries which return a number of rows that grows
#   multiplicatively with the numbers of rows of memoization tables of functions
#   in the query. Such queries may take a very long time or run out of RAM even
#   for moderately-sized projects (`sqlite` will usually complain about this at
#   the start of the query).
# 
#   Certain ways to define and compose memoized functions promote such queries, so
#   a good understanding of this issue may be needed depending on the project.
# - **deletions**: deleting anything from storage is subject to invariants that
#   prevent the existence of "mysterious" objects (ones without a computational
#   history tracing back to user inputs) from existing. This means that you must
#   understand well how deletion works to avoid deleting more things than you
#   really intend.
# 
# ### Performance
# The library has not been optimized much for performance. A few things to keep in
# mind for now:
# - When using disk-based persistence, Mandala introduces an overhead of a few 10s
#   of ms for each call to a memoized function, on top of any work to serialize
#   inputs/outputs and run the function.
# - Storing and loading large collections can be slow (a list of 1000 integers
#   already leads to a visible ~1s delay)
