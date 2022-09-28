# Design of contexts
## Motivation
A few "first principles" observations motivate contexts as a feature:
- there are different "modes" (run, query, batch, delete, delete-by-query are
some, but there may appear more; right now, we only have run and query
implemented) in which you can execute the same piece of code, and it's in fact
natural and convenient to do so;
- there are multiple situations in which it is most natural to recursively nest
one mode of execution within another, in rough order of obscurity:
  - run -> batch: e.g., to batch-execute only the work that it makes sense to
  - run -> delete: e.g., to delete only some downstream results
  - run -> query: e.g., to point to some value and then use it in a query
  - batch -> delete: e.g., after the computation has finished and you want to
  delete part of the work
  - batch -> query: e.g., same as run -> query b/c you're too lazy to remove
  the batch context.
  - query -> delete-by-query: e.g., to declaratively pick out and delete only
  some results
  - delete -> delete-by-query: you want to delete lots of things. But to get
  to some of them, you need some imperative work + some declarative work
  - x -> x: if you copy-paste some code and are too lazy to remove the
  redundant context managers

These observations motivate something context-like, i.e. some way to say: for
this piece of code, use this mode of execution. Contexts as implemented now are
a reasonable way to do this that also give you:
- **readability** via indent. It's very unlikely that you'll need more than 3
levels of nesting, so things won't go out of control.
- **cascading configuration** for e.g. which storage to use and other settings,
so that you don't repeat yourself.
 
## How to demagic
It seems that the nesting requirement is quite real: even though there are a lot
of restrictions on how you can nest contexts, there are a lot of cases that are
reasonable, and you can even imagine (somewhat contrived, but) depth-4 scenarios
like run -> batch -> query -> delete-by-query. So I don't see how to replace
contexts with something more familiar that would work equally well. 

If you ignore all modes but run, query and batch, then all modes but run are
"final", meaning you can't nest any further from them. So everything is depth-1
or 2. But it's still unclear how that would help demagic things?
### Conventional demagic-ing tricks
So it seems the best we can do is
- minimize surprises in the use of contexts and make them very transparent via
logging and APIs 
- explain clearly to people why they are needed and how they work.