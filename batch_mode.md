# Designing and implementing call batching
## What we want
We want to be able to trace a piece of code, collect all the calls in it, and
hand them off to a "batched" executor that is more efficient than sequential
execution. The below is an imagined piece of code using such a functionality, in
a way that feels like it would be natural and pleasant to use:
```python
with run():
	# produce some results normally
	...
	a, b, c = f(...)
	with batch():
		# do some work using a, b, c and others produced so far
		xs = []
		things = [0.1, 0.2, ...]
		for thing in things: # here be performance gains
			xs.append(g(a, thing))
		y = h(xs, c, ...)
		...
		# hand off execution to batched executor 
	# continue normally in the `run` context
	z = ...
```
### Design constraints/thoughts that emerge
Some things stick out based on the above snippet:
- **`batch` must be nestable under other contexts like `run`, `query` and
`delete`**: because batched stages may appear in the middle of a workflow
- **must be retraceable**: for example, imagine your executor died halfway
through, but wrote some work. You want next time to register the fact that only
some of the work needs to be done. It also means that the executor
implementation should have some method like `flush_results()` that gets called
periodically in the background. 
- **you don't have to say something like `compute(...)` in `dask`**: the context
exit implicitly hands execution over to the executor and blocks, unlike with
frameworks like `dask`/`ray`, where you must say "compute these variables" to
start the actual computation.
- **control flow limitations**: the code inside the `batch` block may not branch
on conditions that depend on the values to be computed (or maybe it can, but
things get more difficult/hacky/impossible).
- **executor API**: in the simplest/most general case, it will be handed a
computational graph of calls; it should give you back values for all the
outputs. 
- **invisible to `query` context**: a `batch` context within a `query` context
should have no effect (i.e., as if the code inside is executed in the enclosing
`query` context directly). 

### Design questions
- **how will RAM be managed by the executor?**: when running a declarative
computation (like the computational graph here) as opposed to an imperative
computation, if not all the objects fit in memory, things get more complicated:
	- you need a scheduler to decide in what order to load things in memory and
	run computations. This is what `dask` and `ray` have to do.
	- you *may* also need the user code to have hints for optimizing memory
	usage, basically telling the executor to load some big thing in memory, run
	a lot of computations with it, then proceed with the next big thing. This is
	what you can use `ray.put` for.
### Implementation proposal
#### Overview
- The `batch` context collects "empty" `Call` and `ValueRef` instances for the
calls that are not found in memory. These represent the call graph. Since we are
using content hashing, the UIDs of the value references and calls are not known
in advance, and need to be filled in by the executor later. 
- At the `__exit__` of the context, the executor is called to 
	- compute the missing `ValueRef` objects
	- assign UIDs to them based on content
- After the context exits, everything should look as if this was just another
`run` context. This ensures the smooth nesting of `batch` contexts within `run`
contexts. 

#### Interfaces
```python
class BatchedExecutorClient:
	def execute_calls(self, calls:List[Call]) -> List[Call]:
		# given a web of calls interlinked with vrefs, assigns values and UIDs
		# as above. This may or may not periodically write partial results to
		# the storage.
		...

class BatchedExecutor:
	# depends on what backend we're targeting and how RAM will be managed.
```
