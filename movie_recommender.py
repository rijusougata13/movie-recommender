#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:31:05 2020

@author: sougata
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('movie_dataset.csv')
#print(df.head())

features=['keywords','cast','genres','director']

def combinedfeatures(row):
    return row['keywords']+' '+row['cast']+ ' '+row['genres']+ ' '+row['director']

for feature in features:
    df[feature]=df[feature].fillna('')
df['combinedfeature']=df.apply(combinedfeatures,axis=1)

#print(df['combinedfeature'])
cv=CountVectorizer()
count_matrix=cv.fit_transform(df['combinedfeature'])

cosine_sim=cosine_similarity(count_matrix)

def get_index_from_title(title):
    return df[df.title==title]['index'].values[0]
def get_title_from_index(index):
    return df[df.index==index]['title'].values[0]

movie_user_liked=input('enter the movie name : ')
movie_index=get_index_from_title(movie_user_liked)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print('recommend movies are->')
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i+=1
    if(i>6):
        break
    
    
    
    
