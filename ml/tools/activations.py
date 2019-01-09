#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[11]:


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return self.__call__(x) * (1 -  self.__call__(x))


# In[13]:


class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


# In[14]:


class Tanh:
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1
    
    def gradient(self, x):
        return  1 - np.power(self.__call__(x), 2)


# In[15]:


class Relu:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)
    
    def gradient(self, x):
        return np.wher(x >=0, 1, 0)


# In[16]:


class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)
    
    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


# In[20]:


class ELU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
    
    def gradient(self, x):
        return np.where(x >= 0, 1, self.__call__(x) + self.alpha)


# In[18]:


class SELU:
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 
    
    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
    
    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha * np.exp(x))


# In[19]:


class SoftPlus:
    def __call__(self, x):
        return np.log(1 + np.exp(x))
    
    def gradient(self, x):
        return 1 / (1 + np.exp(-x))


# In[17]:


get_ipython().system('jupyter nbconvert --to script activations.ipynb')


# In[ ]:





# In[ ]:




