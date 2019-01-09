#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[11]:


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return self.__call__(x) * (1 -  self.__call__(x))


# In[10]:


get_ipython().system('jupyter nbconvert --to script activations.ipynb')

