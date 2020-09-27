from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template
from flask import request,redirect
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

app = Flask(__name__)


df=pd.read_csv('movie_dataset.csv')
features=['keywords','cast','genres','director']
df['title']=df['title'].str.lower()
df['original_title']=df['original_title'].str.lower()
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
    try:
        return df[df.title==title]['index'].values[0]
    except :
        pass        
def get_title_from_index(index):
    try:
        return df[df.index==index]['title'].values[0]
    except:
        pass


@app.route('/',methods=['POST','GET'])
def hello():
    if request.method=='POST':
        movie_user_liked= request.form.get("fname").lower() 
        movie_index=get_index_from_title(movie_user_liked)
        similar_movies =  list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        i=0
        movie_list=[]
        
        for element in sorted_similar_movies:
            name=(get_title_from_index(element[0]))
            movie_list.append(name)
            i+=1
            if(i>6):
                break
        try:
            return render_template('submit1.html',value1=movie_list[0],value2=movie_list[1],value3=movie_list[2],value4=movie_list[3],value5=movie_list[4])
        except:
            return render_template('error1.html')
    else:
        return render_template("hello2.html")
@app.route('/about.html',methods=['POST','GET'])
def about():
    return render_template("about.html")

@app.route('/contact.html',methods=['POST','GET'])
def contact():
    return render_template("contact.html")
if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=process.env.PORT )