# A design for "mandala lite"
## What we're cutting out for now
The following things:
- **list/dict logic**: This makes a lot of things throughout the entire project
  more complicated, including
  - the wrapping and unwrapping code
  - the query code
  - the saving/loading code for objects
  - the list/dict-specific implementations of `ValueRef` and `Operation`
- **the type system**: We'll rely on Python's type system for now. This is OK for
  typechecking if we decide to enforce it. It means that by default all values
  of the same type will be stored in the same way. There's a workaround for
  custom per-input/output storages without a type system
- **bulk storage methods**: they introduce some non-essential overhead with
  unclear benefit
- **superops**: for now, let's see what things are like without them. 
  - They are easy to introduce later. Will also give us time to
  think about other ways to handle them.
  - whether we need them really depends on what the workflows are going to be
  like!
  - removing the distinction between `@op`s and `@superop`s would be problematic
- **provenance logic**: maybe we'll need to add it later for better deletion
- **mutation stuff**: not a priority
- **base classes with one implementation**: 
- **private attributes / properties**: let's just expose everything for now.
- **magic methods** for value references that let you manipulate them as if they
  are the underlying values.

## What we're simplifying
- **signatures**: upon reflection it will be much simpler to have a `Signature`
  object that keeps track of and updates all metadata for a function (version,
  name, external and internal input names)
- **op storage**: just basing it on `Signature` objects should make it easier
- **how we represent functions**: let's just roll with more Pythonic stuff. A
  function will have some named inputs, some defaults (we won't keep track of
  types), and a number of nameless outputs. A function returns a list of things,
  possibly empty.
- **storages**: maybe just use a single KVStore for calls, objects for now 
- **configuration**: we'll be more opinionated for now
- **generic stuff**: I think there's opportunity to refactor some interfaces and
  reduce indirection

## Some seemingly non-essential things I'd like to keep
- **changing functions**: this includes renaming functions and their arguments,
  and adding inputs to existing functions backward-compatibly. I feel it comes
  up often enough and wouldn't be much extra code to do. 
- 

## Overview of components
Let's just do it...

# A list of questions to answer for the design
Things:
    - do we want to support renaming ops/args? This is a bit about the "duration
    of life" of these projects. If they will live on a long time, it's kinda a
    priority. 
    - what are the inputs to functions we want to handle? Are we going to be
    passing super weird Python objects, or can we assume built-in scalars +
    np.ndarray + dataframes?
    - what will our functions return? Will it be crazy Python objects, or...?
    - what scale are we expecting? How many function calls? How many functions? 
    - **important question**: how "deep" are the workflows? What I'm thinking:
    there are maybe 10s of Python functions, you compose deep workflows out of
    them, how many times are you calling functions in these workflows though?
    - the trade-offs for superops: if workflows don't get too big, we don't need
    them. However, this also depends on latency constraints. What is too big?
    How fast can we expect a lookup to be?
    - in terms of designing object storage - there's an easy way to convert a
    Python object to a file. The question is: do we want to keep all these
    things as files, or have a dedicated solution for "small" things (e.g., put
    floats in a sqlite table?)
    - how do you parallelize jobs? I use things like dask and ray, but I guess
    you have something else. How does it work and what constraints does it place?