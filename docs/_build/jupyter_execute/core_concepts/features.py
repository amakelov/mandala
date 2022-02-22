#!/usr/bin/env python
# coding: utf-8

# # Features
# 
# ## Problem description
# Mandala makes it easy to compose, run, evolve and query a broad range of
# computations using a minimal generating set of composable Python idioms familiar
# to a broad range of users. 
# 
# It focuses on precisely, yet unobtrusively, managing computations that
# - are as expressive as imperative programs (as opposed to a DAG of functions)
#   (*mutations still a work in progress*)
# - can operate on data structures with shared substructures without duplicating
#   storage
# - have hierarchical structure (think a function that calls other computational
#   primitives inside itself)
# 
# It is specifically geared towards applications in fields where computations are
# heavy enough to require some persistent storage (data science, machine learning,
# ...) -- and all the complexities that arise from this, such as the need to
# - name and organize data
# - avoid recomputing the same quantity 
# - ensure data provenance and reproducibility
# - adapt storage to changing logic
# - query data relationships across projects with many components, and make sense
#   of what exists in the storage
# 
# Mandala's goal is to remove these complexities while maintaining the flexibility
# and evolvability of projects.
# 
# ## Core capabilities
# There are four main components that work together to make Mandala practical:
# - **composable [memoization](https://en.wikipedia.org/wiki/Memoization)** lets you 
#   - express computations using pure Python (collections, loops, conditionals,
#   recursion, ...);
#   - save results as they are computed, and mirror the way code composes in
#     the organization of their storage;
#   - use code itself to flexibly traverse the structure of storage
#   (**retracing**), without loading unnecessary intermediate results. 
#   
#   The nice thing about this is that you can simply run your computations,
#   without worrying about how to keep track of the results and represent them.
#   However complex the process by which you compute results, you have a guarantee
#   that you can go back and query these results by simply (and quickly) stepping
#   through the code that created them. 
# - **pattern-matching queries** let you query computational dependencies between
#   variables in a project declaratively by more or less reusing the code inducing
#   these dependencies. This is a complementary query mechanism to retracing, and
#   can be composed with it. 
# - **abstraction** turns retracing and pattern-matching into practical tools for
#   working with large projects by using the natural hierarchical organization of
#   code to take shortcuts in the computational graph. This works for a simple
#   reason: parts of the computational graph involving a large number of tasks
#   overwhelmingly tend to have an abstractable shared structure. 
# - **refactoring** lets you evolve computational logic the way you refactor a
#   software project, while giving you easy control over how new versions of the
#   primitives relate to their past results.
# 
# ## Additional features and design goals
# - **disciplined data model**: the meaning of every object in storage arises
#   completely from the **computational** operations it participates in, and how
#   they trace back to inputs provided explicitly by the user.
#   - In particular, this discourages the storage of any metadata with no
#   objectively verifiable (i.e., computational) basis, since it would be flagged
#   as an unused argument in your functions.
#   - Since manual annotation is sometimes desirable, it is available in a
#   transparent and traceable (should doubts arise) way. A memoized function
#   without outputs and computational content is equivalent to a manually
#   populated table; more generally, any argument not used in a function's body
#   is equivalent to manual human annotation of data.
# - **immutability**: Mandala's storage is immutable by design, which makes it
#   nearly impossible to corrupt the data record. The exception is forgetting an
#   object, which leaves only a reference to its place in the computational
#   graph. Higher-level interfaces can create the illusion of mutable state by
#   using such forgetting.
# - **fine-grained control**: there are many ways to modify the default behavior of
#   the library: 
#   - **memoization can be disabled** on a per-output basis (which
#   means that you'll need to re-compute). This behavior
#   composes, meaning you can have entire chains of values that are reconstructed
#   dynamically when requested. 
#   - **custom storage formats** can be used on a per-type annotation basis, including
#     optimizations for reading/writing values of this type in bulk
#   - **low-level storage interfaces** can be exposed inside memoized functions to
#     allow direct control over which objects get loaded when (e.g., to iterate
#     over larger-than-memory collections)
#   - **cascading context managers** give you additional control of library
#     behaviors over blocks of code, and let you compose different ways of
#     interpreting Python.
# - **uniform model of computational resources**: fundamentally unserializable
#   objects (for example, a database connection or a web service) can participate
#   in computations in exactly the same way as serializable ones (like datasets,
#   arrays, machine learning models, ...), and are recreated in-memory when
#   demanded by control flow.
# - **integrations**: Mandala straightforwardly integrates with
#   [ray](https://ray.io) and [dask](https://dask.org), which are also based
#   around using functions as the units of work.
