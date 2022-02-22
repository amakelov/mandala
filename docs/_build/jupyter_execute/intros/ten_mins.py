#!/usr/bin/env python
# coding: utf-8

# (10mins)=
# # Manage a toy ML project
# In this quick overview, you will use Mandala to manage a small ML project. You
# will
# - set up a pipeline to build and evaluate your own random forest classifier;
# - evolve the project by extending parameter spaces and functionality;
# - query the results in various ways, and delete the ones you don't need any
#   more. 
# 
# To start, let's import a few things:

# In[1]:


from typing import List, Tuple
import numpy as np
from numpy import ndarray
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mandala.all import *
set_logging_level(level='warning')


# ## Manage experiments with composable memoization
# Let's create a storage for the results, and functions to generate a synthetic
# dataset and train a decision tree:

# In[2]:


storage = Storage(in_memory=True)

@op(storage)
def get_data() -> Tuple[ndarray, ndarray]:
    return make_classification(random_state=0)

log_calls = True
@op(storage)
def train_tree(X, y, max_features=2, random_state=0) -> DecisionTreeClassifier:
    if log_calls: print('Computing train_tree...')
    return DecisionTreeClassifier(random_state=random_state, max_depth=2, 
                                  max_features=max_features).fit(X, y)


# The `@op` decorator turns functions into **operations**. Inputs/outputs for
# calls to operations can be
# [memoized](https://en.wikipedia.org/wiki/Memoization),
#  either in-memory or on
# disk. In addition to saving the inputs/outputs of function calls, Mandala's
# memoization 
# - smoothly *composes* through Python's control flow, collections, and functional
#   abstraction;
# - can load results on demand, getting only the data necessary for control flow
#   to proceed.
# 
# Below, you'll use this composable behavior to evolve and query the project. 
# 
# ### Retrace memoized code to flexibly interact with storage
# To start, let's train a few decision trees on the data:

# In[3]:


with run(storage): 
    X, y = get_data()
    for i in range(5): 
        train_tree(X, y, random_state=i)


# As expected, the `train_tree` operation was computed 5 times. All calls to
# operations inside a `run` block are saved in storage, and re-running a
# memoized call simply loads its results instead of computing the function. 
# 
# This makes it easy to add more computations directly to memoized code. For
# example, here's how you can add logic to evaluate the tree ensemble via a
# majority vote, and also increase the number of trees trained:

# In[4]:


@op(storage)
def eval_forest(trees:List[DecisionTreeClassifier], X, y) -> float:
    if log_calls: print('Computing eval_forest...')
    majority_vote = np.array([tree.predict(X) for tree in trees]).mean(axis=0) >= 0.5
    return round(accuracy_score(y_true=y, y_pred=majority_vote), 2)

with run(storage, lazy=True):
    X, y = get_data()
    trees = [train_tree(X, y, random_state=i) for i in range(10)]
    forest_acc = eval_forest(trees, X, y)
    print(f'Random forest accuracy: {forest_acc}')


# Note that
# - `train_tree` was computed only 5 times, since the first 5 calls were *retraced*
# - operations return **value references** (like `forest_acc` above), which contain
#   metadata needed for storage and computation.
# - you can use **collections** of individual value references as inputs/outputs
#   of operations -- like how `eval_forest` takes a list of decision trees -- without
#   duplicating the storage of the collection's elements.
# 
# By using `lazy=True` (as above) in the `run` context manager, you only load what
# is necessary for code execution not to be disrupted -- for example, to evaluate a
# branching condition, or to run new computations. 
# 
# #### Why should you care?
# The retracing pattern simplifies many aspects of experiment management. It
# - makes code **open to extension** of parameter spaces and functionality.
#   Directly add new logic to existing code, and re-run it without re-doing
#   expensive work;
# - enables **expressive queries**. Walk over chains of memoized calls to reach
#   the results you need by directly retracing the steps that generated them. By
#   sprinkling extra control flow when needed, you can query very complex computations;
# - makes it easy to **resume work after a crash**: just run the code again, and
#   [break jobs with many calls into stages](10mins-hierarchical) to speed up recovery.

# (10mins-hierarchical)=
# ### Reduce complexity with hierarchical memoization
# Once a composition of operations proves to be a useful building block in 
# experiments, you can *extract* it as a higher-level operation using the
# `@superop` decorator. Below, you'll extract the training of 
# decision as a higher-level operation, and expose the dataset and `max_features`
# as arguments:

# In[5]:


@superop(storage)
def train_trees(X, y, max_features=2, n_trees=10) -> List[DecisionTreeClassifier]:
    return [train_tree(X, y, max_features=max_features, random_state=i) for i in range(n_trees)]


# You can now use this superop to **refactor both the code and the results
# connected to it**: 

# In[6]:


with run(storage, lazy=True):
    X, y = get_data()
    trees = train_trees(X=X, y=y)
    forest_acc = eval_forest(trees=trees, X=X, y=y)


# Note that this did not execute any heavy operations -- it only added
# higher-level indexing to the memoized data. The first call to `train_trees`
# retraces its internal structure, but all subsequent re-executions will jump
# straight to its final results.
# 
# #### Why should you care?
# Hierarchical memoization extends the computer programming concept of extracting a 
# [subroutine](https://en.wikipedia.org/wiki/Subroutine#Advantages) to the 
# management of *both* code and its results, bringing additional benefits like
# - **simplicity**: make code *and* queries shorter;
# - **convenience**: index experimental outcomes in a more meaningful way;
# - **efficiency**: skip over uninteresting calls when retracing, and improve 
#   the time complexity of queries by matching to higher-level patterns. 
#   - This turns retracing into a practical tool for large projects, since in any
#   human-designed scenario where you must retrace a large number of calls, these
#   calls will have some abstractable structure that you can extract to a
#   `@superop` and skip over.
# 
# Together, these properties allow projects to *scale up practically
# indefinitely*, as long as the designers themselves have a scalable and
# hierarchical mental decomposition of the logic.

# ## Turn code into a declarative query interface
# Retracing is very effective at interacting with storage when the space of input
# parameters (from which retracing must begin) is known to you in advance. This is
# not always the case. 
# 
# To make sense of all that exists in storage, you can use a complementary
# interface for *pattern-matching* against the structure of experiments. First,
# let's create a bit more data to query by running more experiments:

# In[7]:


log_calls = False
with run(storage, lazy=True):
    X, y = get_data()
    for n_trees in (10, 20):
        trees = train_trees(X=X, y=y, n_trees=n_trees)
        forest_acc = eval_forest(trees=trees, X=X, y=y)


# Composing a query works a lot like composing the experiment being queried (so
# much that a good first step is to just copy-paste the computational code). For
# example, here is how you can query the relationship between the accuracy of the
# random forest and the number of trees in it:

# In[8]:


with query(storage) as q:
    X, y = get_data()
    n_trees = Query().named('n_trees')
    trees = train_trees(X, y, n_trees=n_trees)
    forest_acc = eval_forest(trees, X, y).named('forest_acc')
    df = q.get_table(n_trees, forest_acc)
df


# ### What just happened?
# The `query` context *significantly* changes the meaning of Python code:
# - calls to `@op`/`@superop`-decorated functions build a pattern-matching
#   graph encoding the query;
# - This graph represents computational dependencies between variables (such as
#   `n_trees` and `forest_acc`) imposed by operations (such as `train_trees`);
# - to query the graph, you provide a sequence of variables. The result is a table,
#   where each row is a matching of values to these variables that satisfies *all*
#   the dependencies.

# ### Advanced use: matching more interesting patterns
# Above, you queried using the higher-level operation `train_trees` to directly
# get the list of decision trees from the data. However, you can also do a
# lower-level query through the internals of `train_trees`! 
# 
# #### A bit of refactoring
# To make this a little more interesting, suppose you also want to look at the
# accuracy of each individual tree in the forest. The trees are computed inside
# the `train_trees` operation, so this is also the natural place to compute their
# accuracies. 
# 
# However, execution skips over past calls to `train_trees`, so you can't really
# go inside. To overcome this, you can create a **new version** of `train_trees`,
# which does not remember the calls to the previous version, and will thus execute
# them again with the updated logic to evaluate the trees. Then, simply re-run the
# project to re-compute calls to the new version:

# In[9]:


@op(storage)
def eval_tree(tree, X, y) -> float:
    return round(accuracy_score(y_true=y, y_pred=tree.predict(X)), 2)

@superop(storage, version='1') # note the version update!
def train_trees(X, y, max_features=2, n_trees=10) -> List[DecisionTreeClassifier]:
    trees = [train_tree(X, y, max_features=max_features, random_state=i) for i in range(n_trees)]
    for tree in trees: eval_tree(tree, X, y) # new logic
    return trees

with run(storage, lazy=True):
    X, y = get_data()
    for n_trees in (10, 20):
        trees = train_trees(X=X, y=y, n_trees=n_trees)
        forest_acc = eval_forest(trees=trees, X=X, y=y)


# Note that the new version of `train_trees` still returns the same list of
# trees, so downstream calls using this list are unaffected. The only new work
# done was running `eval_tree` inside `train_trees`.
# 
# #### Advanced query patterns
# Below is code to query the internals of `train_tree`:

# In[10]:


with query(storage) as q:
    X, y = get_data()
    tree = train_tree(X, y).named('tree')
    tree_acc = eval_tree(tree, X, y).named('tree_acc')
    df_trees = q.get_table(tree, tree_acc)
    print(df_trees.head())
    with q.branch():
        trees = MakeList(containing=tree, at_index=0).named('trees')
        forest_acc = eval_forest(trees, X, y).named('forest_acc')
        df_forest = q.get_table(trees, forest_acc)
        print(df_forest)


# The above example introduces two new concepts:
# - the `MakeList` built-in function matches a list given a variable representing
#   an element of the list. The additional constraint `at_index=0` makes it so
#   that you don't match `trees` multiple times for each of its elements.
# - the `.branch()` context can be used to give scope to dependencies, so
#   that different code blocks don't interact with one another unless one
#   explicitly uses another's variables. This allows you to match to any
#   `tree` outside the `branch()` context, yet match only the first tree in the
#   list `trees` inside it. 
# 
# The upshot is that query structure can be re-used to see both the forest and
# the trees :)

# ## A few other things
# ### Update an operation to preserve past results
# Above, you created a new version of `train_trees` -- which behaves like an
# entirely new operation, sharing only its name with the previous version. In
# particular, all connection to past calls of `train_trees` is lost in the new
# version, which forces recomputation of all calls to `train_trees`.
# 
# Often, you don't want to lose the connection to past results. You can do this by
# **updating** an operation instead. For example, let's expose a parameter of
# `get_data`: 

# In[11]:


@op(storage)
def get_data(n_samples=CompatArg(default=100)) -> Tuple[ndarray, ndarray]:
    return make_classification(n_samples=n_samples, random_state=0)


# Calls to `get_data` can now use the new argument; all existing calls will behave
# as if they were called with `n_samples=100`. 

# #### Why should you care?
# Updating an operation makes it easy to evolve logic in many practical cases:
# - expose parameters in the operation's interface that were hard-coded in its body until now;
# - extend the operation's behavior, for example by adding a new method for processing the
#   input data;
# - create a closely related variant of the operation that can coexist
#   with the current one, of fix a bug affecting only some calls to the operation. 

# ### Deletion: run code in undo mode
# If you want to un-memoize the results of some calls, wrap them in a `delete`
# block:

# In[12]:


with run(storage, lazy=True, autocommit=True):
    X, y = get_data()
    with delete():
        trees = train_trees(X=X, y=y, n_trees=20)


# It's important to understand just what you deleted by doing this. Deletion works by
# - collecting all calls captured in a `delete` block, recursively including those inside
#   higher-level operations;
# - deleting these calls and their outputs;
# - and all results and calls that were computed using these outputs;
# - and all calls that involve (as inputs or outputs) any of the deleted values
#   described above.
# 
# This means that all decision trees trained in the call to `train_trees` got
# deleted. These are *all* decision trees in the project, since the first 10 are
# shared between the calls with `n_trees=10` and `n_trees=20`. As a consequence,
# the call to `train_trees` with `n_trees=10` is deleted too, because it involves
# a list of deleted trees.
# 
# There’s also a declarative interface to delete results, which is analogous to the
# declarative query interface.

# ## Next steps
# There are some important capabilities of the library not covered in this
# tutorial. These will be linked to if tutorials on them become available:
# - you can disable memoization selectively on a per-output basis. This can be
#   used to force quantities to be recomputed on demand when this is more
#   practical than saving and loading (think `lambda x: x + 1` for a large array
#   `x`);
# - you can use custom object storage logic on a per-type basis (the default is
#   `joblib`, and there's builtin support for `sqlite`+`pickle` too);
# - you can nest different kinds of contexts, for example a `query` context inside
#   a `run` context, to compose their functionality
