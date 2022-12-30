<div align="center">
	<br>
		<img src="assets/logo-no-background.png" height=128 alt="logo" align="center">
	<br>
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
- [simple interface and usage](#add-composable-memoization-to-existing-code): decorate functions with `@op` to
  memoize them, and put ordinary Python code in `storage.run()`, `storage.query()`,
  ... blocks depending on what you want to do. The rest is just Python.
- [rapid iteration](#iterate-rapidly-without-redoing-work) without redoing work,
  great for notebooks and other interactive environments
- [query by pattern-matching Python
  code](#query-by-pattern-matching-python-code): produce tables of results
  directly from Python code encoding their relationships.
- [modify memoized
  functions](#modify-memoized-functions-without-forgetting-old-calls) seamlessly without
  forgetting old calls
- [automatic function-level dependency tracking](#dependency-tracking): get
  (optional) alerts when a function's dependencies change, and decide whether to
  recompute.
- [native data structure support](#data-structures): store and track elements of
  Python collections separately, enabling Pythonic code and incremental
  computation.

## Installation
```
pip install git+https://github.com/amakelov/mandala
```

## Video walkthrough

### Add *composable* memoization to existing code
Decorate functions with `@op` and annotate the number of return values (with a
`typing.Tuple` for functions returning multiple values). Below is a simple
example with a `scikit-learn` pipeline:

<details open><summary>Show/hide gif</summary>
<p>

![01_memoization](https://user-images.githubusercontent.com/1467702/210118002-4d2418a3-5d34-42f4-bf49-8a0522b788b1.gif)

</p>
</details>


### Iterate rapidly without redoing work
`mandala`'s memoization is designed to be composed across functions, and makes
it straightforward to grow a project: just add new parameters and functions on
top of a piece of memoized code.

<details open><summary>Show/hide gif</summary>
<p>

![02_iteration](https://user-images.githubusercontent.com/1467702/210118075-f48501ab-ba13-473f-a8fe-0fd2d555b9e1.gif)

</p>
</details>


### Query by pattern-matching Python code
Define rich declarative queries by directly using the code of your experiments.
In a `.query()` context, function calls are interpeted as building a graph of
**computational dependencies between variables**. Calling `get_table(...)` on the
context manager gives you tables of all the tuples in the storage satisfying all
these dependencies:

<details open><summary>Show/hide gif</summary>
<p>

![03_queries](https://user-images.githubusercontent.com/1467702/210118099-0fcbfb60-cc02-438b-b975-3e335558d8d1.gif)

</p>
</details>

### Modify memoized functions without forgetting old calls
You can modify memoized functions without losing past calls by adding extra
arguments with default values. All past calls are treated as if they used this
default value. This is very convenient in machine learning and data science
scenarios, for e.g. exposing hard-coded constants as parameters, or adding new
behaviors to a processing step.

<details open><summary>Show/hide gif</summary>
<p>

![04_add_input](https://user-images.githubusercontent.com/1467702/210118150-f8abd146-9b3e-4987-9ac2-782be8c4f856.gif)

</p>
</details>

### Dependency tracking
TODO

### Data structures
TODO

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/01_logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- [dependency tracking](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/02_dependencies.ipynb) tutorial