# Design of signatures and internal/human-readable names
- [Design of signatures and internal/human-readable names](#design-of-signatures-and-internalhuman-readable-names)
  - [Motivation](#motivation)
  - [On schema changes](#on-schema-changes)
    - [Classification](#classification)
    - [Things to consider when designing schema changes](#things-to-consider-when-designing-schema-changes)
  - [Schema changes that we'll implement](#schema-changes-that-well-implement)
  - [Backward-compatible schema changes that may or may not make sense](#backward-compatible-schema-changes-that-may-or-may-not-make-sense)
  - [Design discussion](#design-discussion)
    - [Renaming, and motivating immutable internal names](#renaming-and-motivating-immutable-internal-names)
    - [Designing immutable internal names](#designing-immutable-internal-names)
    - [Designing the remote signature storage](#designing-the-remote-signature-storage)
## Motivation
In `mandala`, functions are given many extra powers and responsibilities beyond
computation. This over-reliance on functions can limit your ability to manage
complexity in a project if the way a function is modeled by the system is not
flexible enough. 

## On schema changes
Plans for computational projects are often incomplete or initially misguided. We
want to enable you to make changes while seamlessly retaining the relationship
between code and storage.

**Backward-compatible changes**: The most difficult questions are about changes
that preserve the existing calls, in the sense that retracing these calls should
not trigger recomputation, and we are able to keep using the function in
computations *and* queries without introducing a version change.

### Classification
Atomic schema changes can be exhaustively categorized according to the part of a
function's interface that is being changed. The parts of an interface are:
- inputs;
- input names;
- input types;
- optional default values for inputs;
- outputs;
- output types. 

For each of these we can ask whether we can add, remove, or change them.

### Things to consider when designing schema changes
The high-level thing to worry about is introducing some nasty surprises with
such changes. As usual, the worst kind of surprise is the one you find out about
very very late: your code works, and you think it did one thing, but it was
actually doing another. 

There are a few specific parts to think about:
- how existing `run` context code will be affected. For example, whether
existing calls to the function will remain retraceable or will trigger
recomputation. 
- how existing `query` context code will be affected. Will the query results be
  the same?
- what the implications are for the internal architecture. For example, renaming
  raises some questions that, if not answered well, can make things very
  complicated. 

## Schema changes that we'll implement
In the design of `mandala`, the following features have proven to be
particularly helpful and easy to reason about:
- add a new input to a function. This is very useful when you want to introduce
new behaviors (e.g., add an option that switches between algorithms), or expose
a hard-coded value without losing the memory of past calls. It's a very natural
change in ML/DS things in particular.
- create a new version of a function under the same name, that starts with a
  fresh memoization table, and need not have any compatibility with the
  interface of the previous version (though likely it will). This is useful when
  you e.g. notice a bug and want to (correctly) recompute all results that
  depend on it. The benefit compared to creating a brand-new function is that
  you keep the namespace clean.
- rename a function or its arguments while keeping memory of past calls. In my
  view this is not a very frequent need, but still good to have

## Backward-compatible schema changes that may or may not make sense
- creating a default value: this could make sense. All past calls will have
explicitly provided a value for this argument, so old code will remain
retraceable. 
- removing a default value: this could make sense. But some calls would not be
retraceable any more unless you go and provide this default value. The good
thing is that potentially worse confusion won't arise because you'll get errors
thrown at you.
- changing a default value: this seems too confusing. Calls that used the old
  default implicitly will now be recomputed during retracing.
- adding an output in a backward-compatible way: could potentially be useful.
You'd have to generate a value for this output for past calls (say, `None` seems
a reasonable default). 

## Design discussion
### Renaming, and motivating immutable internal names 
If you can just rename things all the time, it can get confusing. Some
examples of the consequences of renaming:
- call UIDs: call UIDs are a concept that simplifies a lot of the internal
  architecture of `mandala`. They're what you put in the columns of memoization
  tables, and the fact that they're simple objects (strings) makes it easy to
  express queries as joins. They're how you reduce checking if a call has been
  computed to a key lookup. So they seem like a pretty nice thing to have.

  The way call UIDs are generated is by hashing a description of the inputs and
  the function's identity. If these change, the hash will end up different, and
  the system will think you haven't made that call yet. As a matter of fact,
  using any deterministic function of the inputs+function id to represent a
  call's identity will change its value after renaming.
  
  You might think one way to deal with this is to use the original names of
  everything. But then you may end up with two functions whose original names
  (of the functions and arguments) are the same, leading to hash collisions.

  So, to ensure things continue to work after renaming, you need some
  "internal", immutable names that point to the current human-readable names.
- synchronization between users: suppose you use the mutable human-readable
  names in the "ground truth" storage. Imagine also you keep a centralized,
  sequentially transactional store of just the function signatures. Imagine the
  following scenario, optimized for sadness:
    - user1 renames everything in the most terrible way, permuting names of
      functions and arguments, and relaying the change to the remote
      signature storage. 
    - meanwhile, user2 has been doing work using the old names and packs
    together the calls (containing the old names) to send off to the storage. 
   
  This illustrates the need for some sort of synchronized event log ("this got renamed to
  that") of all the atomic name changes that you must maintain and apply. When
  the data of user2 arrives, you must 
    - find all the renamings since the last time user2 synced
    - perform all these renames before putting it in storage.

  Similarly, when you pull from the remote storage, you must find all the name
  changes since you last pulled, figure out where all the new calls fit in
  between them, and apply things in the correct order. 

  Compare this to a design where the remote storage contains:
    - blobs of memoization tables named according to immutable internal names
    - a signature storage that contains the relationship between internal and
      human-readable names

  The benefit of this design is that **you only need to know the current state
  of the signatures**, and not the path they took to get there. Anyone can compute
  with old human-readable names and send their calls to the remote storage by
  "anonymizing" them. The main functions would look like this:
    - send calls to remote: switch the calls to internal names and put them in a
      blob - that's it
    - pull calls from remote: 
      - pull the current signatures to make sure you have the correct ones
      - pull the call blobs and "humanize" them according to the signatures you
        got. If you encounter a function name not in the signatures, this means
        the function was deleted.
    - even if you use old signatures, you won't do anything catastrophic,
    because the internal names are immutable. You will see human-readable
    names different from a more up to date user. But that won't affect the
    calls you push to remote. 

### Designing immutable internal names
There is tension between the usefulness of immutable names (as described above),
and the usefulness of human-readable names for debugging, transparency, and just
looking at things and having the reassurance that all is working correctly. 

To this end, immutable names should be maximally suppressed from the internal
workings of all local copies of the system. Here's the proposal for this:
- the `Signature` object contains these names and the mapping to human-readable
  names 
- at runtime, most all the names that flow through the system are the current
  human-readable names, with a few exceptions:
    - when forming the call UID, you internally pass the internal names. This
      won't be visible to a user!
    - when packing/unpacking calls to/from the remote, you will
      dehumanize/humanize them
    - the provenance table (when we implement it) will also contain the
      immutable identifiers. If users want to look at the provenance table,
      there would be a method to humanize it
- this is actually very little work! You only need extra logic in the situations
  above. 
- **note**: the event log table can use the current human-readable names,
this is fine! It is a purely "local" construct

### Designing the remote signature storage
With the adoption of immutable names, things are very simple. Here's what needs
to happen:
- each user performing a schema change must 
  - first sync with the signature storage
  - add the change
  - push the new version
- on startup, you load the current signatures from the remote, and use them to
  name things locally. As explained above, even if you have old signatures,
  nothing bad will happen (as long as they match your functions).