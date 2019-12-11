# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:46:59 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:01:33 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:01:13 2019

@author: Dell
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


#model = pickle.load(open('model.pkl', 'rb'))

def create_sim():
    data = pd.read_csv('data.csv')
    
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    
    return data,sim
  
# defining a function that recommends 10 most similar movies
def rcmd(m):
    data = pd.read_csv('data.csv')
    
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['title'].unique():
       
        return('This Book is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
      
        i = data.loc[data['title']==m].index[0]
        
        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))
        
        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        
        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        
        lst = lst[1:5]
       
        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
           
            l.append(data['title'][a])
                      
        return l
    
app = Flask(__name__ )
main = pickle.load(open('main.pkl', 'rb'))
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    title = request.args.get('title')
 
    r = rcmd(title)
    title = title.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',title=title,r=r,t='s')
    else:
        return render_template('recommend.html',title=title,r=r,t='l')



if __name__ == '__main__':
    app.run()