#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))


# In[5]:


s = Sigmoid()


# In[8]:


s(np.array([3, 5, 2]))


# In[ ]:




