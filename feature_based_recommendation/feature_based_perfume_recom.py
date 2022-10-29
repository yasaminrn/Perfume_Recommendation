#!/usr/bin/env python
# coding: utf-8

# Creating feature based perfume recommendation system

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np


# In[2]:


#importing libraries for recommendation modelling, to be moved to modelling
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel


# In[3]:


#getting the data
perfume= pd.read_csv('perfume_cl.csv')


# In[4]:


#removing rows with NA values
perfume.dropna(inplace = True)
perfume.reset_index(inplace = True)


# In[5]:


#feathures used for recommendation
perfume_features = perfume.copy().drop(columns = ['index','brand', 'title', 'rating_score'])


# In[6]:


#titles of perfumes
perfume_title = perfume['title']


# In[7]:


#user input for which we are providing in recommendation
perfume_example = 'Mon Paris Yves Saint Laurent for women'


# In[8]:


#index of the perfume for which we are providing in recommendation 
ex_index = (perfume_title[perfume_title == perfume_example]).index.values


# In[9]:


#taking numerical value from index object
index_val = ex_index[0]


# Both cosine similaity and corrolation function give similarity matrix which are very close to each other, and the recommended perfume based on the two approach are mostly identical. Both formulations are kept here for futute use.

# In[10]:


#cosine similarity matrix
cosine_sim = cosine_similarity(perfume_features)


# In[11]:


#row of similarity matrix for the target perfume
cosine_sim[ex_index]


# In[12]:


#recreating the dataframe with titles included
cosine_concat = pd.DataFrame(cosine_sim[ex_index], columns = perfume_title, index = ex_index).T


# In[13]:


#adding index to data frame
cosine_concat.reset_index()


# In[14]:


#testing that maximum index is the same as original index (item is the most similar to itself)
maxindex = cosine_concat.max(axis=0).index


# In[15]:


#sorting the similarity data frame from most to least similar
cosine_concat.sort_values(by = [index_val], inplace = True, ascending=False)
cosine_concat.reset_index(inplace = True)


# In[16]:


#starting from second value (first value is the target perfume itself) listing 10 most similar perfumes
cosine_concat[1:11]['title']


# In[17]:


#corrolation matrix
corrolation_sim = np.corrcoef(perfume_features)


# In[18]:


#row of similarity matrix for the target perfume
corrolation_sim[ex_index]


# In[19]:


#recreating the dataframe with titles included
corrolation_concat = pd.DataFrame(corrolation_sim[ex_index], columns = perfume_title, index = ex_index).T


# In[20]:


#adding index to data frame
corrolation_concat.reset_index()


# In[21]:


#sorting the similarity data frame from most to least similar
corrolation_concat.sort_values(by = [index_val], inplace = True, ascending=False)
corrolation_concat.reset_index(inplace = True)


# In[22]:


#starting from second value (first value is the target perfume itself) listing 10 most similar perfumes
corrolation_concat[1:11]['title']

