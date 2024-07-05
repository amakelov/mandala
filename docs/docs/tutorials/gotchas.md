# Caveats
<a href="https://colab.research.google.com/github/amakelov/mandala/blob/master/docs_notebooks/tutorials/gotchas.ipynb"> 
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

This notebook lists some things that can go wrong when using `mandala`.


```python
# for Google Colab
try:
    import google.colab
    !pip install git+https://github.com/amakelov/mandala
except:
    pass
```

## If `x == y`, this doesn't guarantee that `x` and `y` will have the same content hash


```python
from mandala.utils import get_content_hash

print(f'Is 1 == True? {1 == True}')
print(f'Is the hash of 1 == the hash of True? {get_content_hash(1) == get_content_hash(True)}')
print(f'Is 23 == 23.0? {23 == 23.0}')
print(f'Is the hash of 23 == the hash of 23.0? {get_content_hash(23) == get_content_hash(23.0)}')
```

    Is 1 == True? True
    Is the hash of 1 == the hash of True? False
    Is 23 == 23.0? True
    Is the hash of 23 == the hash of 23.0? False


## Hashing numerical values is sensitive to precision and type
All three of the values `42, 42.0, 42.000000001` have different content hashes:


```python
from mandala.utils import get_content_hash

print(get_content_hash(42))
print(get_content_hash(42.0))
print(get_content_hash(42.00000000001))
```

    d922f805b5eead8c40ee21f14329d6c7
    ca276c58eef17e13c4f274c9280abc1e
    b61bb24b62bf6b1ab95506a62843be08


It's possible to define custom types that will be insensitive to types and
rounding errors when hashed, but this is currently not implemented.

## Non-deterministic hashes for complex objects
Below we illustrate several potentially confusing behaviors that are hard to
eradicate in general:
- even if we set all random seeds properly, certain computations (e.g., training
a `scikit-learn` model) result in objects with non-deterministic content IDs
- certain objects can change their content ID after making a roundtrip through
the serialization-deserialization pipeline


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import random
import numpy as np

from mandala.utils import get_content_hash, serialize, deserialize

X, y = load_digits(n_class=10, return_X_y=True)

def train_model():
    ### set both the numpy and python random seed
    np.random.seed(42)
    random.seed(42)
    ### train a model, passing the random_state explicitly
    model = RandomForestClassifier(max_depth=2, 
                                n_estimators=100, random_state=42).fit(X, y)
    return model

### training in the exact same way will produce different content hashes
model_1 = train_model()
model_2 = train_model()
print(f'Content IDs of the two models: {get_content_hash(model_1)} and {get_content_hash(model_2)}')

### a roundtrip serialization will produce a different content hash
roundtrip_model_1 = deserialize(serialize(model_1))
print(f'Content IDs of the original and restored model: {get_content_hash(model_1)} and {get_content_hash(roundtrip_model_1)}')
```

    Content IDs of the two models: c8d1485ebe003581fb2019b73a2de97a and c8d1485ebe003581fb2019b73a2de97a
    Content IDs of the original and restored model: c8d1485ebe003581fb2019b73a2de97a and c8d1485ebe003581fb2019b73a2de97a


**Why is this hard to get rid of in general?** One pervasive issue is that some
custom Python objects, e.g. many kinds of ML models and even `pytorch` tensors,
create internal state related to system resources, such as memory layout. These 
can be different between objects that otherwise have semantically equivalent
state, leading to different content hashes. It is impossible to write down a
hash function that always ignores these aspects for arbitrary classes, because 
we don't know how to interpret which attributes of the object are semantically
meaningful and which are contingent.

**What should you do about it?** This issue does come up that often in practice.
Note that this is not an issue for many kinds of objects, such as primitive
Python types and nested python collections thereof, as well as some other types
like numpy arrays. If you always pass as inputs to `@op`s objects like this, or
`Ref`s obtained from other `@op`s, this issue will not come up. Indeed, if
"unwieldy" objects are always results of `@op`s, a single copy of each such
object will be saved and deserialized every time.

This problem does, however, make it very difficult to detect when your `@op`s
have side effects.
