#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from itertools import combinations_with_replacement


# In[2]:


def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    
    return np.array([X_1, X_2])


# In[3]:


def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = X.shape[0]
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []
    subsample_size = n_samples // 2
    if replacements:
        subsample_size = n_samples
    for _ in range(n_subsets):
        idx = np.random.choice(range(n_samples), size=np.shape(range(subsample_size)), replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets


# In[4]:


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


# In[5]:


def polynomial_feature(X, degree):
    n_samples, n_features = np.shape(X)
    
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)
    return X_new


# In[6]:


def make_diagonal(x):
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
        
    return m


# In[23]:


def poly_feature(X, degree):
    poly_frame = pd.DataFrame()
    poly_frame['power_1'] = X
    for power in range(2, degree+1):
        name = 'power_' + str(power)
        poly_frame[name] = X ** power
    return poly_frame


# In[24]:


get_ipython().system('jupyter nbconvert --to script manipulations.ipynb')


# In[10]:


poly_frame = pd.DataFrame()


# In[11]:


X = np.array([1,2,3,4,5])


# In[12]:


poly_frame['power_1'] = X


# In[17]:


X ** 3


# In[ ]:




