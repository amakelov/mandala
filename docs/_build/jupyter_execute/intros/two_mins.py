#!/usr/bin/env python
# coding: utf-8

# (2mins)=
# # A 2-minute primer
# 
# ## Self-managing experiments
# With Mandala, the way you compose experiments out of function calls and
# collection creation/indexing automatically induces the organization of storage.
# For example, below are line-by-line analogous code blocks to
# - save results of an imagined clustering experiment (**left**);
# - query results from this and all analogous experiments in the storage (**right**):
# 
# ```python
# with run(storage, lazy=True):                        with query(storage) as q:                    
#     X, y = get_data(preprocess=True)                     X, y = get_data(preprocess=True)         
#     for n_clusters in (2, 4, 8):                         n_clusters = Query(int)                  
#         clusters = get_clusters(X, y, n_clusters)        clusters = get_clusters(X, y, n_clusters)
#         score = eval_clusters(clusters, y)               score = eval_clusters(clusters, y)       
#         if score > 0.95:                                 score > 0.95                             
#             for cluster in clusters:                     cluster = clusters[IndexQuery()] 
#                 centroid = get_centroid(X, cluster)      centroid = get_centroid(X, cluster)
#                                                          df = q.get_table(n_clusters,             
#                                                                       score, centroid)          
# ```
# These blocks look a lot like code to just run computations, without data
# management concerns -- but in fact support various natural ways to store and
# interact with results.
# 
# ### The `run` context: use code to traverse storage directly 
# If you re-run `run`-wrapped code -- *or any sub-program of it* -- function calls
# you've already computed will load their results as needed to allow control flow
# to proceed. Consequences include
# - **flexible queries**: you can get to the results you want by directly
#   retracing the steps that created them, however complicated they may be (an
#   **imperative** query interface)
# - **easy iteration**: you can organically grow a piece of code with new
#   parameters and functionality, without re-doing expensive work
# - **resilience**: recovering from failure is as simple as running the same
#   code again -- which will retrace the steps that completed successfully, up to
#   the failed computations.
# 
# ### The `query` context: search in storage by pattern-matching to code
# With the `query` block, you
# - interpret function calls and collection operations as building a structure of
#   dependencies between variables (such as `n_clusters` and `score`);
# - point to a sequence of variables to get a table where each row is a
#   matching of values to these variables satisfying the dependencies (a
#   **declarative** query interface)
# 
# ## Next steps: abstraction and refactoring
# Read on to see how
# - these patterns can be composed with **abstraction** and **refactoring** to
#   scale them up to evolving projects with many components
# - you can use imperative/declarative patterns similar to the above for deletions
