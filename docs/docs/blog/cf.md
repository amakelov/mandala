# Tidy computations
In data-driven fields like machine learning, a lot of effort is spent organizing
*computational data* &mdash; results of running programs &mdash; so that it can be analyzed
and manipulated. This blog post introduces the `ComputationFrame` (CF) data
structure, which provides a natural and simple grammar of operations to automate
this. It is implemented as [part of](https://amakelov.github.io/mandala/03_cf/)
[mandala](https://github.com/amakelov/mandala), a Python library for experiment
tracking and incremental computation.

A `ComputationFrame` is a "generalized dataframe", where the set of columns is
replaced by a computation graph of variables and operations, and rows are
(possibly partial) executions of the graph. Any computation frame can be turned
into a dataframe by "forgetting" the graph structure:

![ComputationFrame](output.svg)

Computation frames share properties with both computation graphs and relational 
databases, enabling operations combining the structure of both, such as:

- **expansion**: explore computations by adding to the CF the calls that
produced/used the values in some variables. These calls and their inputs/outputs
are automatically organized into operations and variables, creating new ones if
necessary.
- **filtering** the CF along graph topology and/or properties of individual
values or calls.
- **flattening** the CF into a dataframe, where each row corresponds to an
execution of the computation graph. In database terms, this is a materialized 
view of the CF computed by joining the implicit tables of the operations and
variables.

Much of the exposition here is a re-imagining of the ideas in the [Tidy
Data](https://www.jstatsoft.org/article/view/v059i10) paper in the context of
computational data. The focus is on handling computations involving repeated
calls to the same set of functions, which is a common pattern in machine
learning and scientific computing.

## So what's a `ComputationFrame`?

### Expansion: adding more computations to the CF

### Filtering: restricting to a subset of computations
There are no nulls in a `ComputationFrame`, which makes it easy to filter
operations and variables based on their properties. 

### Flattening: turning the CF into a dataframe for downstream analysis

## Case studies

### Heterogeneous computations

### Pipelines that branch and/or merge

### Aggregation and/or indexing into data

