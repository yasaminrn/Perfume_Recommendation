# Creating feature based perfume recommendation system

#importing the libraries
import pandas as pd
import numpy as np

#importing libraries for recommendation modelling, to be moved to modelling
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel

#getting the data
perfume= pd.read_csv('perfume_cl.csv')

#removing rows with NA values
perfume.dropna(inplace = True)
perfume.reset_index(inplace = True)

#feathures used for recommendation
perfume_features = perfume.copy().drop(columns = ['index','brand', 'title', 'rating_score'])

#titles of perfumes
perfume_title = perfume['title']

#user input for which we are providing in recommendation
perfume_example = 'Mon Paris Yves Saint Laurent for women'

#index of the perfume for which we are providing in recommendation 
ex_index = (perfume_title[perfume_title == perfume_example]).index.values

#taking numerical value from index object
index_val = ex_index[0]

# Both cosine similaity and corrolation function give similarity matrix which are very close to each other, and the recommended perfume based on the two approach are mostly identical. Both formulations are kept here for futute use.

#cosine similarity matrix
cosine_sim = cosine_similarity(perfume_features)

#row of similarity matrix for the target perfume
cosine_sim[ex_index]

#recreating the dataframe with titles included
cosine_concat = pd.DataFrame(cosine_sim[ex_index], columns = perfume_title, index = ex_index).T

#adding index to data frame
cosine_concat.reset_index()

#testing that maximum index is the same as original index (item is the most similar to itself)
maxindex = cosine_concat.max(axis=0).index

#sorting the similarity data frame from most to least similar
cosine_concat.sort_values(by = [index_val], inplace = True, ascending=False)
cosine_concat.reset_index(inplace = True)

#starting from second value (first value is the target perfume itself) listing 10 most similar perfumes
cosine_concat[1:11]['title']

#corrolation matrix
corrolation_sim = np.corrcoef(perfume_features)

#row of similarity matrix for the target perfume
corrolation_sim[ex_index]

#recreating the dataframe with titles included
corrolation_concat = pd.DataFrame(corrolation_sim[ex_index], columns = perfume_title, index = ex_index).T

#adding index to data frame
corrolation_concat.reset_index()

#sorting the similarity data frame from most to least similar
corrolation_concat.sort_values(by = [index_val], inplace = True, ascending=False)
corrolation_concat.reset_index(inplace = True)

#starting from second value (first value is the target perfume itself) listing 10 most similar perfumes
corrolation_concat[1:11]['title']

