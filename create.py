# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:43:12 2019

@author: Dell
"""

import pandas as pd
import numpy as np
import pickle
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data.csv')
                    
data['comb'] = data['title'] + ' ' + data['author'] + ' '+ data['generes'] + ' '+ data['publisher'] 

# creating a count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])

# creating a similarity score matrix
sim = cosine_similarity(count_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('data.csv',index=False)

#Saving model to disk
pickle.dump(sim, open('main.pkl','wb'))

# Loading model to compare the results
main = pickle.load(open('main.pkl','rb'))