#Pre-Processing of fragrance data to make it ready for content-based recoomendation

#importing the libraries
import pandas as pd
import numpy as np

#importing sklearn libraries for onehotencoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#getting the data
fragrance= pd.read_csv('perfume.csv')

#initial exploratory assessment of fragrance data

print('head of the data:')
fragrance.head()

fragrance.info()

fragrance.columns

# For the recommendation purpose, we will focus on perfumes which are rated by a substantial number of users.
# When the perfumes are reviewed by few users, the review results could be highly biased.
# Usuually first reviewers could be celebrities in the industry, getting it as gift from brands and it could cause unintended review bias.

fragrance=fragrance[(fragrance['votes']>200)]

#resetting the index back to start from 0 after dropping the ones with less than 200 votes
fragrance = fragrance.reset_index()

# The accord list of a perfume is one of the main component showing its category.
# People's test is perfumes usually shows itself in the perfume accords they prefer.

#showing fragrance accords 
print('sample accords:')
fragrance['accords']

# The accords column consist of a long string with numerous accords listed for each perfume, sorted by the number of people who found the perfume to be in this category.
# We are focusing on the top three accord for each perfume and extracting them to 3 new columns

fragrance['accords'] = fragrance['accords'].apply(lambda x: x.split(','))

l=len(fragrance['accords'])
print('length of accords is '+ str(l))

accord1 = (fragrance['accords'].iloc[0:l]).str.get(0)

accord2 = (fragrance['accords'].iloc[0:l]).str.get(1)

accord3 = (fragrance['accords'].iloc[0:l]).str.get(2)

fragrance['accord1']=accord1

fragrance['accord2']=accord2

fragrance['accord3']=accord3


# Next step, only columns that would be used for content recommendation are kept in the cleaned data frame.
# Columns related to like/love/dislike are removed, as not independant from the rating column itself.
# Notes are removed, as they are captured in accords. Unique accords are in order of 60 distinct labels.
# Given the huge variety of notes, and considering they are categorial as well, these columns are removed as well.

frag_cl = fragrance[['brand', 'title', 'rating_score', 
       'longevity_poor', 'longevity_weak', 'longevity_moderate',
       'longevity_long', 'longevity_very_long', 'sillage_soft',
       'sillage_moderate', 'sillage_heavy', 'sillage_enormous','clswinter', 'clsspring', 'clssummer',
       'clsautumn','clsday', 'clsnight', 'accord1', 'accord2', 'accord3']]


# Removing N/A values. As the numver of rows did not change, there was no N/A values.

fragcl = frag_cl.dropna()

frag_cl.info()


# Comparing the number of votes per perfume with different ratings for longevity, sillage, season and daytime suitability, it is understood that a reviewer does not need to provide input for all the fields. 
# Therefore for each category of voting, the proportional ratio of votes are calculated

longevity_list = frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']]

frag_cl['longevity_rating'] = longevity_list.sum(axis=1)

frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']] = np.divide((frag_cl[['longevity_poor','longevity_weak','longevity_moderate','longevity_long','longevity_very_long']]),frag_cl[['longevity_rating']])

sillage_list = frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']]

frag_cl['sillage_rating'] = sillage_list.sum(axis=1)

frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']] = np.divide((frag_cl[['sillage_soft', 'sillage_moderate', 'sillage_heavy', 'sillage_enormous']]), frag_cl[['sillage_rating']])

season_list = frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']]

frag_cl['season_rating'] = season_list.sum(axis=1)

frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']] = np.divide((frag_cl[['clswinter', 'clsspring', 'clssummer', 'clsautumn']]),frag_cl[['season_rating']])

daytime_list = frag_cl[['clsday', 'clsnight']]

frag_cl['daytime_rating'] = frag_cl[['clsday', 'clsnight']].sum(axis=1)

frag_cl[['clsday', 'clsnight']] = np.divide((frag_cl[['clsday', 'clsnight']]),frag_cl[['daytime_rating']])

frag_cl=frag_cl.drop(columns=['longevity_rating', 'sillage_rating', 'season_rating',
       'daytime_rating'])


# All accord columns are categorical data, which needs to be converted to a form of numeric values through onehotencoding

# instantiate labelencoder object
le = LabelEncoder()

frag_cl['accord1'] = frag_cl['accord1'].astype(str) 

frag_cl['accord2'] = frag_cl['accord2'].astype(str) 

frag_cl['accord3'] = frag_cl['accord3'].astype(str) 

accord1_new = frag_cl[['accord1']].apply(lambda col: le.fit_transform(col))

accord2_new = frag_cl[['accord2']].apply(lambda col: le.fit_transform(col))

accord3_new = frag_cl[['accord3']].apply(lambda col: le.fit_transform(col))

onehot_encoder = OneHotEncoder(sparse=False)

accord1_encoded = onehot_encoder.fit_transform(accord1_new)

# The columns for encoding accord1, accord2, and accord3 need to have unique values.
# Therefore continuing the numbering of the columns sequentially.

accord1column=range(fragrance['accord1'].nunique())

accord2column=range(fragrance['accord1'].nunique(), fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+1)

accord3column=range(fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+1,fragrance['accord1'].nunique()+fragrance['accord2'].nunique()+fragrance['accord3'].nunique()+2)

accord1_hot_encoded = pd.DataFrame(accord1_encoded, columns = accord1column)

accord2_encoded = onehot_encoder.fit_transform(accord2_new)

accord2_hot_encoded = pd.DataFrame(accord2_encoded, columns = accord2column)

accord3_encoded = onehot_encoder.fit_transform(accord3_new)

accord3_hot_encoded = pd.DataFrame(accord3_encoded, columns = accord3column)

#concatinating the three sets of hit_encoded accords 
accords_encoded = pd.concat([accord1_hot_encoded, accord2_hot_encoded,accord3_hot_encoded ], axis=1)

#shoowing the encoded results of accord columns
print('encoded accord columns:')
accords_encoded

#dropping the columns which will be included by oneHotEncoding
todrop = ['accord1', 'accord2', 'accord3']
frag_cl = frag_cl.drop(columns = todrop)

#adding the encoded accords back
frag_cl_encoded = pd.concat([frag_cl, accords_encoded], axis=1)

#showing how new dataframe looks like
print('the new data frame:')
frag_cl_encoded

#saving the ready updated data frame as csv.file ready to use for content based recommendation modelling
frag_cl_encoded.to_csv('perfume_cl.csv', sep=',', index=False, header=True)

