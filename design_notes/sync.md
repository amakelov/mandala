# Overview of sync architecture, protocols and invariants
This note gives a high-level overview of what state the clients and server
maintain, and how clients synchronize state with the server. 

## Goals
Maintain in a "somewhat distributed" way a collection of Python functions and calls to them, so that each client can
- change functions in one of a few useful ways, 
- compute and store calls to these functions,

while having these nice properties:
- **no data loss for calls**: each client's memoized calls are always accepted
by the server and recorded in the storage (even if the schema changes in the
meantime)
- **no merge conflicts for schemas**: each client's schema changes are always
accepted or rejected immediately due to an incompatible concurrent update.
Changes that conflict with each other cannot be made.
- **"eventual consistency"**: all clients that sync with the server after
changes to the calls/schema have stopped arrive at the same state of the schema
and memoization tables

These properties essentially enable a "collaborative computational environment"
for the clients that should hopefully be as understandable and ergonomic as
computing locally in native Python.

## How it works
- **changes clients can make to the function interfaces (schema)**: 
	- create a new function,
	- add an input with a default value to an existing function,
	- rename a function or an input.
	- create a new version of a function. This is just a new function that goes
	by the same name as an existing function.
- each function and input has a **unique, immutable identifier** throughout its
life that can be used to refer to it even if the human-readable name has
changed. This is similar to "field tags" in protocol buffers
- **each change to the schema is immediately synced with the server and checked
for consistency against the current state**. Only one client at a time may
update the server schema. This ensures that 
	- **schema merge conflicts don't arise** (e.g., two clients creating a
	function by the same name with a different interface),
	- **the schema on the server is a "possibly renamed superset" of the schema
	on each client at any point in time**: it may have more functions and inputs
	for functions, and they may have different human-readable names, but once an
	internal name is issued, it remains there forever in the same role (function
	name or input name).
- **stale calls**: calls to old function signatures can always be applied to the
storage (no matter how old). This is because the schema can only "grow" over
time, and renamings can be resolved via immutable function/input names. When
inserting calls in the database, default values for the missing columns fill in
any missing inputs.
- **any client who starts a computation will be able to commit its results**:
since stale calls are always correctly incorporated into the storage, changing
the schema during a computation won't lead to problems when the computation
finishes and sends results to the server.
## Caveats
### Non-deterministic computations
If a function `f` is non-deterministic, problems may arise when users
concurrently compute `f` on the same values:
- users may arrive at different values for the outputs. 
- if you use **content hashing** for outputs, this means two calls with the same
UID but different output UIDs, which ruins lots of things. The worst problem is
how to pick which outputs to return when re-tracing the call.
- if you use **causal hashing** for the outputs, the UIDs for the outputs will
be the same, but the values will still be different. You have to choose which
output values to mark as "true". If one user has work that builds upon the
"losing" branch of this choice, the conflict resolution will then lead to "fake
calls": an arrangement of input and output values in a call that don't actually
correspond to a real computation.
### Re-synchronizing a function that has been renamed
A problem with this protocol is that, if clients are not careful, they may end
up with multiple variants of the same function. This can happen when
- clients agree on some function `f`
- one client renames it to `g`
- another still has the old code that calls it `f`. Since nothing really
connects the code itself to the persistent identity of this function,
synchronizing this `f` again will observe that there is no function that goes by
UI name `f` in the storage, and will create a new memoization table etc. for
this `f`.