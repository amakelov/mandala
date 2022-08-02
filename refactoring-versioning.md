# Target features
After `mandala`, :
- add an argument to a function with a default value
- create new version
- rename a function/argument
# What needs to happen
## Adding an argument
- must happen automatically during sync (with a message)
- must put value in objects, in a single transaction
- must take care to ensure that calls for which the new argument is with the
  default value don't mess up the call UID calculation
## Renaming things
- rename database tables and columns [DONE]
- high-level `Storage` methods to rename function/argument
    - these must only be applicable when the event log and caches are empty
- sync must check that everything's alright
## Versioning
- version attribute in decorator, signature
- must be somehow reflected in database table names...
- the superop problem