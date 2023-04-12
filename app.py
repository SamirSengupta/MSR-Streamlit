import streamlit as st
import pickle
import pandas as pd
import requests
from fuzzywuzzy import fuzz


def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=1a219482c51e5dc798ae1ac0a309d459&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def fetch_movie_details(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=1a219482c51e5dc798ae1ac0a309d459&language=en-US'.format(movie_id))
    data = response.json()
    return data


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index
    if len(movie_index) == 0:
        return None, None, None, None
    movie_index = movie_index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    recommended_movies=[]
    recommended_movies_posters=[]
    recommended_movies_overview=[]
    recommended_movies_ratings=[]
    for i in movies_list:
        if len(recommended_movies) == 6:
            break
        if i[0] == movie_index:
            continue
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster from api
        recommended_movies_posters.append(fetch_poster(movie_id))
        #fetch movie details from api
        movie_details = fetch_movie_details(movie_id)
        recommended_movies_overview.append(movie_details['overview'])
        recommended_movies_ratings.append(round(movie_details['vote_average'], 1))
    return recommended_movies, recommended_movies_posters, recommended_movies_overview, recommended_movies_ratings


def find_closest_match(movie_name, movie_titles):
    movie_scores = [(title, fuzz.ratio(movie_name.lower(), title.lower())) for title in movie_titles]
    closest_match = max(movie_scores, key=lambda x: x[1])
    return closest_match[0]


movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

movie_titles_str = ','.join(movies['title'].values)

st.set_page_config(page_title="Movie Recommender", page_icon=":movie_camera:", layout="wide")

st.title('Movie Recommendation System')
st.write('Enter the name of a movie below to get a list of similar movies:')

selected_movie_name = st.text_input('Movie Name', value='', key=1, autocomplete=movie_titles_str)

if selected_movie_name:
    closest_match = find_closest_match(selected_movie_name, movies['title'])
    movie_matches = movies[movies['title'] == closest_match]
    
    if len(movie_matches) == 0:
        st.write('Sorry, the movie you are searching for is either incorrect or does not exist or may not be present in our database.')
    else:
        selected_movie_index = movie_matches.index[0]
        selected_movie_poster = fetch_poster(movies.iloc[selected_movie_index].movie_id)
        selected_movie_overview = fetch_movie_details(movies.iloc[selected_movie_index].movie_id)['overview']
        selected_movie_rating = round(fetch_movie_details(movies.iloc[selected_movie_index].movie_id)['vote_average'], 1)
        st.write('Selected Movie:', closest_match)
        st.image(selected_movie_poster, width=200)
        st.write('Rating:', selected_movie_rating)
        st.write('Overview:', selected_movie_overview)
        

        recommended_movies, recommended_movies_posters, recommended_movies_overview, recommended_movies_ratings = recommend(closest_match)

    if recommended_movies:
        st.write('Recommended Movies:')
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        cols = [col1, col2, col3, col4, col5, col6]
        for i in range(len(recommended_movies)):
            with cols[i]:
                st.write(recommended_movies[i])
                st.image(recommended_movies_posters[i], width=150)
                st.write('Rating:', recommended_movies_ratings[i])
                st.write('Overview:', recommended_movies_overview[i])

                
    else:
        st.write('Sorry, we could not find any recommendations for this movie.')
