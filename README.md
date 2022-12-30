<div align="center">
	<br>
		<img src="assets/logo-512x512.png" height=55 alt="logo" align="right">
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
on steroids, applied to elegantly solve all your experiment data management
needs. 

It turns Python function calls into composable, interlinked, queriable data that
is automatically co-evolved with your codebase. By applying different semantics
to this data, the same piece of ordinary Python code can be used to not only
compute, but also save, load, query, delete, and batch-compute computational
artifacts. This unlocks extremely flexible yet simple patterns of data
management in complex projects.

## Features
- [simple interface and usage](#basic-usage): decorate functions with `@op` to
  memoize them, and put ordinary Python code in `storage.run()`, `storage.query()`,
  ... blocks depending on what you want to do. The rest is just Python.
- [query by pattern-matching Python
  code](#query-by-pattern-matching-python-code): produce tables of results
  directly from Python code encoding their relationships.
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

<!--
<video src="assets/Kazam_screencast_00000.webm" controls="controls" style="max-width: 730px;">
</video>
-->

## Video walkthrough

### Add *composable* memoization to existing code
Decorate functions with `@op` and annotate the number of return values (with a
`typing.Tuple` for functions returning multiple values):

<video src="assets/memoization.mp4" controls="controls" style="max-width: 730px;">
</video>

### Iterate rapidly w/ new parameters and logic

### Query by pattern-matching to Python code

### Dependency tracking

### Data structures

## Tutorials 
- see the ["Hello world!"
  tutorial](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/00_hello.ipynb)
  for a 2-minute introduction to the library's main features
- See [this notebook](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/01_logistic.ipynb)
for a more realistic example of a machine learning project managed by Mandala.
- [dependency tracking](https://github.com/amakelov/mandala_lite/blob/master/mandala_lite/tutorials/02_dependencies.ipynb) tutorial