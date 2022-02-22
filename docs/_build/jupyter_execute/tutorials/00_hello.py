#!/usr/bin/env python
# coding: utf-8

# (tutorial-hello)=
# # Hello world!
# In this tutorial, you will create a storage and use it to track calls to a
# Python function. You will save time when calling the function on the same inputs
# a second time, by having the result loaded for you instead of being computed
# again.

# ## Define an operation
# First, import the necessary objects and create a storage:

# In[1]:


from mandala.all import *
storage = Storage(in_memory=True)


# The `storage` object is used to store the inputs and outputs for calls to Python
# functions. To make this automatic, you must tell the storage about the function
# by decorating the function with the `@op()` decorator, passing the storage as
# argument:

# In[2]:


@op(storage)
def add(x, y) -> int:
    print('Hello world!')
    # ...some long computation...
    return x + y 


# This process of connecting a function to the storage turns it into an
# **operation**, which is the fundamental unit of tracking and computation in
# Mandala. 

# ## Call the operation in a `run` context to save results
# Next, call the operation `add` you just defined:

# In[3]:


add(23, 42)


# This is just the normal behavior of the function. Importantly, **no data was
# saved to storage by the above call**, and the function returns what you would
# expect. To track the results of computations, you must instead wrap them in the
# `run` context manager:

# In[4]:


with run(storage):
    result = add(23, 42)
result


# The above code recorded in storage that the function `add` was called on the
# inputs `23` and `42`, and that the result's value was `65`. The returned object
# `result` is a **value reference**. Value references represent an arbitrary
# Python object wrapped together with metadata used for storage and computation. 
# 
# ## Call the operation again to *load* results
# Importantly, re-running this code will not recompute the function, since the results
# already exist in storage. Instead, the storage will figure out for you that you
# have already called the function on these inputs, and will retreive the result;
# this is called **retracing** the computation:

# In[5]:


with run(storage):
    result = add(23, 42)
result


# Observe that this time nothing was printed out, because `add` was not actually
# computed.

# ## Exercise: extend the computation
# As an exercise, run the following code next:

# In[6]:


with run(storage):
    for n in range(20, 25):
        result = add(n, 42)


# What happened? How many times was `"Hello, world!"` printed out? Why? 

# ## Exercise: compose operations
# Operations are intended to be used as building blocks of larger computations. 
# Below, define an increment operation `inc` that on input `x` returns `x+1` and prints
# out a message. 

# In[7]:


### your code here


# In[8]:


@op(storage)
def inc(x) -> int:
    print('Hello from inc!')
    return x + 1 


# Now run the following code that combines `add` and `inc` into a toy pipeline and
# saves the results:

# In[9]:


with run(storage):
    for n in range(3):
        n_inc = inc(n)
        result = add(n, n_inc)


# Run the computation again; is anything printed out? Why?

# In[10]:


with run(storage):
    for n in range(3):
        n_inc = inc(n)
        result = add(n, n_inc)


# ## Next steps
# You have tracked and resumed your first computation with Mandala. Dive into the
# next tutorials (coming soon) to build a toolkit for extending these concepts to
# increasingly realistic applications.
