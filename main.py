from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests

app = Flask(__name__)

# Google Custom Search API configuration
API_KEY = 'AIzaSyC4rmRH67HRyQTcpmrt8NTT9gpDNi2HJ9Q'
CX = 'b0c5c6c01ce3a4959'


# INITIALIZING SOME VARIABLES
datasets=pickle.load(open("movies.pkl","rb"))
similarity=pickle.load(open("similarity.pkl",'rb'))

cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(datasets['tags']).toarray()


similarity=cosine_similarity(vectors)


# FUNCTIONS FOR THE RECOMMENDATIONS
def get_poster_url(movie_title):
    # Make a request to Google Custom Search API
    url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={movie_title}&searchType=image'
    response = requests.get(url)
    data = response.json()
    
    # Extract the first image URL from the response
    if 'items' in data:
        return data['items'][0]['link']
    else:
        return None



def recommend(movie):
    movie_index=datasets[datasets['title']==movie].index[0]
    
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True , key=lambda x:x[1])[1:7]
    
    movie_indices = [i[0] for i in movie_list]

    return datasets['title'].iloc[movie_indices]


# HOME PAGE OF THE WEBSITE
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def get_recommendations():  # Rename the function here
    movie_title = request.form['movie']
    recommendations=recommend(movie_title)
    poster_urls = [get_poster_url(movie) for movie in recommendations]
    
    # Zip the recommendations and poster_urls
    recommendation_data = zip(recommendations, poster_urls)
    
    return render_template('recommendation.html', movie_title=movie_title, recommendation_data=recommendation_data)


if __name__=='__main__':
    app.run(debug=True)