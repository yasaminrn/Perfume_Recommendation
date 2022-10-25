#!/usr/bin/env python
# coding: utf-8
Pre-Processing of fragrance data to make it ready for content-based recoomendation
# In[1]:


#importing the libraries
import pandas as pd
import numpy as np


# In[2]:


#importing sklearn libraries for onehotencoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[3]:


#getting the data
fragrance= pd.read_csv('perfume.csv')


# In[4]:


#initial exploratory assessment of fragrance data


# In[5]:


print('head of the data:')
fragrance.head()


# In[6]:


fragrance.info()


# In[7]:


fragrance.columns


# In[8]:


#adding new parameter (gender) extracted from title
for i in range(len(fragrance['title'])):
    if ('for women and men') in fragrance['title'][i]:
        fragrance['gender'][i] = 0.5
    elif 'for women' in fragrance['title'][i]:
        fragrance['gender'][i] = 0
    elif 'for men' in fragrance['title'][i]:
        fragrance['gender'][i] = 1


# For the recommendation purpose, we will focus on perfumes which are rated by a substantial number of users.
# When the perfumes are reviewed by few users, the review results could be highly biased.
# Usuually first reviewers could be celebrities in the industry, getting it as gift from brands and it could cause unintended review bias.

# In[9]:


fragrance=fragrance[(fragrance['votes']>200)]


# In[10]:


#resetting the index back to start from 0 after dropping the ones with less than 200 votes
fragrance = fragrance.reset_index()


# The accord list of a perfume is one of the main component showing its category.
# People's test is perfumes usually shows itself in the perfume accords they prefer.

# In[11]:


#showing fragrance accords 
print('sample accords:')
fragrance['accords']


# The accords column consist of a long string with numerous accords listed for each perfume, sorted by the number of people who found the perfume to be in this category.
# We are focusing on the top three accord for each perfume and extracting them to 3 new columns

# In[12]:


fragrance['accords'] = fragrance['accords'].apply(lambda x: x.split(','))


# In[13]:


l=len(fragrance['accords'])
print('length of accords is '+ str(l))


# In[14]:


accord1 = (fragrance['accords'].iloc[0:l]).str.get(0)


# In[15]:


accord2 = (fragrance['accords'].iloc[0:l]).str.get(1)


# In[16]:


accord3 = (fragrance['accords'].iloc[0:l]).str.get(2)


# In[17]:


fragrance['accord1']=accord1


# In[18]:


fragrance['accord2']=accord2


# In[19]:


fragrance['accord3']=accord3


# Next step, only columns that would be used for content recommendation are kept in the cleaned data frame.
# Columns related to like/love/dislike are removed, as not independant from the rating column itself.
# Notes are removed, as they are captured in accords. Unique accords are in order of 60 distinct labels.
# Given the huge variety of notes, and considering they are categorial as well, these columns are removed as well.

# In[20]:


frag_cl = fragrance[['brand', 'title', 'rating_score', 'gender', 
       'longevity_poor', 'longevity_weak', 'longevity_moderate',
       'longevity_long', 'longevity_very_long', 'sillage_soft',
       'sillage_moderate', 'sillage_heavy', 'sillage_enormous','clswinter', 'clsspring', 'clssummer',
       'clsautumn','clsday', 'clsnight', 'accord1', 'accord2', 'accord3']]


# Removing N/A values. As the numver of rows did not change, there was no N/A values.

# In[21]:


fragcl = frag_cl.dropna()


# In[22]:


frag_cl.info()


# Comparing the number of votes per perfume with different ratings for longevity, sillage, season and daytime suitability, it is understood that a reviewer does not need to provide input for all the fields. 
# Therefore for each category of voting, the proportional ratio of votes are calculated

# In[23]:


longevity_list = frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']]


# In[24]:


frag_cl['longevity_rating'] = longevity_list.sum(axis=1)


# In[25]:


frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']] = np.divide((frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']]),frag_cl[['longevity_rating']])


# In[26]:


sillage_list = frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']]


# In[27]:


frag_cl['sillage_rating'] = sillage_list.sum(axis=1)


# In[28]:


frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']] = np.divide((frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']]), frag_cl[['sillage_rating']])


# In[29]:


season_list = frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']]


# In[30]:


frag_cl['season_rating'] = season_list.sum(axis=1)


# In[31]:


frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']] = np.divide((frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']]),frag_cl[['season_rating']])


# In[32]:


daytime_list = frag_cl[['clsday', 'clsnight']]


# In[33]:


frag_cl['daytime_rating'] = frag_cl[['clsday', 'clsnight']].sum(axis=1)


# In[34]:


frag_cl[['clsday', 'clsnight']] = np.divide((frag_cl[['clsday', 'clsnight']]),frag_cl[['daytime_rating']])


# In[35]:


frag_cl=frag_cl.drop(columns=['longevity_rating', 'sillage_rating', 'season_rating',
       'daytime_rating'])


# All accord columns are categorical data, which needs to be converted to a form of numeric values through onehotencoding

# In[36]:


# instantiate labelencoder object
le = LabelEncoder()


# In[37]:


frag_cl['accord1'] = frag_cl['accord1'].astype(str) 


# In[38]:


frag_cl['accord2'] = frag_cl['accord2'].astype(str) 


# In[39]:


frag_cl['accord3'] = frag_cl['accord3'].astype(str) 


# In[40]:


accord1_new = frag_cl[['accord1']].apply(lambda col: le.fit_transform(col))


# In[41]:


accord2_new = frag_cl[['accord2']].apply(lambda col: le.fit_transform(col))


# In[42]:


accord3_new = frag_cl[['accord3']].apply(lambda col: le.fit_transform(col))


# In[43]:


onehot_encoder = OneHotEncoder(sparse=False)


# In[44]:


accord1_encoded = onehot_encoder.fit_transform(accord1_new)


# The columns for encoding accord1, accord2, and accord3 need to have unique values.
# Therefore continuing the numbering of the columns sequentially.

# In[45]:


accord1column=range(fragrance['accord1'].nunique())


# In[46]:


accord2column=range(fragrance['accord1'].nunique(), fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+1)


# In[47]:


accord3column=range(fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+1,fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+fragrance['accord3'].nunique()+2)


# In[48]:


accord1_hot_encoded = pd.DataFrame(accord1_encoded, columns = accord1column)


# In[49]:


accord2_encoded = onehot_encoder.fit_transform(accord2_new)


# In[50]:


accord2_hot_encoded = pd.DataFrame(accord2_encoded, columns = accord2column)


# In[51]:


accord3_encoded = onehot_encoder.fit_transform(accord3_new)


# In[52]:


accord3_hot_encoded = pd.DataFrame(accord3_encoded, columns = accord3column)


# In[53]:


#concatinating the three sets of hit_encoded accords 
accords_encoded = pd.concat([accord1_hot_encoded, accord2_hot_encoded,accord3_hot_encoded ], axis=1)


# In[54]:


#shoowing the encoded results of accord columns
print('encoded accord columns:')
accords_encoded


# In[55]:


#dropping the columns which will be included by oneHotEncoding
todrop = ['accord1', 'accord2', 'accord3']
frag_cl = frag_cl.drop(columns = todrop)


# In[56]:


#adding the encoded accords back
frag_cl_encoded = pd.concat([frag_cl, accords_encoded], axis=1)


# In[57]:


#showing how new dataframe looks like
print('the new data frame:')
frag_cl_encoded


# In[58]:


#saving the ready updated data frame as csv.file ready to use for content based recommendation modelling
frag_cl_encoded.to_csv('perfume_cl.csv', sep=',', index=False, header=True)
