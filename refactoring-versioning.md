# Target features
After `mandala`, :
- add an argument to a function with a default value
- create new version
- rename a function/argument
# What needs to happen
## Renaming things
- rename database tables and columns [DONE]
- switch to qualified name as the name of a memoization table [DONE]
- create a database table for schema data [DONE]
- create event log for schema changes [TODO]
- high-level `Storage` methods to rename function/argument [DONE]
    - these must only be applicable when the event log and caches are empty
- sync must check that everything's alright
- print out informative messages
## Adding an argument
- `Signature.update` must return description of the changes
- must happen automatically during sync (with a message)
- must put value in objects, in a single transaction
- must take care to ensure that calls for which the new argument is with the
  default value don't mess up the call UID calculation
## Versioning
- version attribute in decorator, signature
- switch to qualified name for tables
- the superop problem