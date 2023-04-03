import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import plotly.graph_objects as go


# Set page config
st.set_page_config(page_title="Movie Recommendation System", page_icon=":clapper:")


# Set up the first tab
def first_tab():
    
    st.title('Cosine Similarity')
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
        if closest_match[1] >= 70:
            return closest_match[0]
        else:
            return None



    movies_dict = pickle.load(open('movie_dict.pkl','rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))

    movie_titles_str = ','.join(movies['title'].values)


    st.title('Movie Recommendation System')
    st.write('Enter the name of a movie below to get a list of similar movies:')

    selected_movie_name = st.text_input('Movie Name', value='', key=1, autocomplete=movie_titles_str)

    if selected_movie_name:
        try:
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
        except:
            st.write('Sorry, please search for any other movie.')

# Set up the second tab
def second_tab():
    st.title('TMDB')
    import requests


    # Define the API key
    api_key = "1a219482c51e5dc798ae1ac0a309d459"

    # Define the base URL for TMDb API requests
    BASE_URL = "https://api.themoviedb.org/3/{}?api_key=1a219482c51e5dc798ae1ac0a309d459&language=en-US"


    # Define the function to fetch movie information
    def fetch_movie_info(query):
        # Make API request to search for movie
        search_url = BASE_URL.format("search/movie") + "&query=" + query
        search_response = requests.get(search_url)
        search_data = search_response.json()

        # Check if there are any search results
        if "results" not in search_data or not search_data["results"]:
            st.write("TMDB Does not have the data of the movie you are searching for")
            return None

        # Get ID of first search result
        movie_id = search_data["results"][0]["id"]

        # Make API request to get movie info
        movie_url = BASE_URL.format("movie/" + str(movie_id))
        movie_response = requests.get(movie_url)
        movie_data = movie_response.json()

        # Extract relevant movie info
        movie_info = {
            "id": movie_data["id"],
            "name": movie_data["original_title"],
            "poster": "https://image.tmdb.org/t/p/w500" + movie_data["poster_path"],
            "title": movie_data["title"],
            "rating": round(movie_data["vote_average"], 1),
            "overview": movie_data["overview"]
        }

        return movie_info



    # Define the function to fetch related movies
    def fetch_related_movies(movie_id, num_recommendations=10):
        url = f"{BASE_URL.format('movie/' + str(movie_id) + '/recommendations')}?api_key={api_key}&language=en-US"
        response = requests.get(url)
        movies = response.json()["results"][:num_recommendations]
        return [
            {
                "name": movie["original_title"],
                "poster": f"https://image.tmdb.org/t/p/w185{movie['poster_path']}",
                "title": movie["original_title"],
                "rating": round(movie["vote_average"], 1),
                "overview": movie["overview"]
            }
            for movie in movies
        ]


    # Define the UI layout
    st.title("Movie Recommendation System")

    # Define the search box
    search_query = st.text_input("Search for a movie")

    # If the search box is not empty, fetch and display the movie information
    if search_query:
        movie_info = fetch_movie_info(search_query)

        if movie_info is None:
            st.write("No results found. Please enter a valid movie name.")
        elif "status_message" in movie_info:
            st.write("The movie does not exist on TMDb server. Please enter a valid movie name.")
        else:
            # Create two columns for the image and movie information
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(movie_info["poster"])
            with col2:
                st.write("## " + movie_info["name"])
                st.write("**Title:** " + movie_info["title"])
                st.write("**Rating:** " + str(movie_info["rating"]))
                st.write("**Overview:** " + movie_info["overview"])
            # Fetch and display related movies
            st.header("Related Movies")

            # Fetch  related movies
            related_movies = fetch_related_movies(movie_info["id"])

            # Display the related movies
            for i, movie in enumerate(related_movies[:5]):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(movie["poster"], width=200)
                with col2:
                    st.write(f"## {i+1}. {movie['name']}")
                    st.write("**Title:** " + movie["title"])
                    st.write("**Rating:** " + str(movie["rating"]))
                    st.write("**Overview:** " + movie["overview"])

# Set up the third tab
def third_tab():
    import requests
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # TMDB API URL and key
    tmdb_url = "https://api.themoviedb.org/3"
    tmdb_key = "1a219482c51e5dc798ae1ac0a309d459"

    # Function to search for a movie by name
    def search_movie(query):
        url = f"{tmdb_url}/search/movie?api_key={tmdb_key}&query={query}"
        response = requests.get(url)
        data = response.json()
        return data

    # Function to get movie details from TMDB API
    def get_movie_details(movie_id):
        url = f"{tmdb_url}/movie/{movie_id}?api_key={tmdb_key}&language=en-US"
        response = requests.get(url)
        data = response.json()
        return data

    # Function to get movie reviews from TMDB API
    def get_movie_reviews(movie_id):
        url = f"{tmdb_url}/movie/{movie_id}/reviews?api_key={tmdb_key}&language=en-US&page=1"
        response = requests.get(url)
        data = response.json()
        return data

    # Function to perform sentimental analysis using TextBlob
    def analyze_textblob(review):
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

    # Function to perform sentimental analysis using Vader
    def analyze_vader(review):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(review)
        compound = sentiment['compound']
        if compound > 0:
            return "Positive"
        elif compound < 0:
            return "Negative"
        else:
            return "Neutral"

    # Streamlit app
    def app():
        st.title("Sentimental Analysis of Movie Reviews")
        
        movie_name = st.text_input("Enter the movie name:", key="movie_name_input")
        if movie_name:
            # Search for movie by name
            search_results = search_movie(movie_name)
            if len(search_results["results"]) == 0:
                st.error("Movie not found. Please enter a valid movie name.")
                return
            # Get the first search result and extract movie ID
            movie_id = search_results["results"][0]["id"]
            # Get movie details from TMDB API
            movie_details = get_movie_details(movie_id)
            title = movie_details["title"]
            st.header(title)
            release_date = movie_details["release_date"]
            st.write(f"Release date: {release_date}")
            overview = movie_details["overview"]
            st.write(f"Overview: {overview}")
            # Get movie poster from TMDB API and display it
            poster_path = movie_details["poster_path"]
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                st.image(poster_url)
            else:
                st.warning("No poster found for this movie.")
            # Get movie reviews from TMDB API
            movie_reviews = get_movie_reviews(movie_id)
            if len(movie_reviews["results"]) == 0:
                st.warning("No reviews found for this movie.")
                return
            # Perform sentimental analysis using TextBlob and Vader
            textblob_sentiments = []
            vader_sentiments = []
            for review in movie_reviews["results"]:
                content = review["content"]
                # Perform sentimental analysis using TextBlob
                textblob_sentiments.append(analyze_textblob(content))
                # Perform sentimental analysis using Vader
                vader_sentiments.append(analyze_vader(content))

            # Generate visualization of sentiment analysis using Plotly
            import plotly.express as px
            import pandas as pd

            # Create dataframe with sentiment data
            data = {'Review': [review['content'] for review in movie_reviews["results"]],
                    'TextBlob': textblob_sentiments,
                    'Vader': vader_sentiments}
            df = pd.DataFrame(data)

            # Create bar chart
            fig = px.bar(df, x='Review', y=['TextBlob', 'Vader'], barmode='group', 
                        title=f"Sentiment Analysis of Reviews for '{title}'")
            st.plotly_chart(fig)

            # Create pie chart for sentiment distribution
            textblob_counts = df['TextBlob'].value_counts()
            vader_counts = df['Vader'].value_counts()

            fig = go.Figure(data=[go.Pie(labels=textblob_counts.index, values=textblob_counts.values, name="TextBlob"),
                                go.Pie(labels=vader_counts.index, values=vader_counts.values, name="Vader")])
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(
                title={
                    'text': f"Sentiment Distribution for '{title}'",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

            # Create histogram for sentiment polarity distribution
            textblob_polarity = df['TextBlob'].apply(lambda x: TextBlob(x).sentiment.polarity)
            vader_polarity = df['Vader'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

            fig = go.Figure(data=[go.Histogram(x=textblob_polarity, name="TextBlob", nbinsx=50),
                                go.Histogram(x=vader_polarity, name="Vader", nbinsx=50)])
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.75)
            fig.update_layout(
                title={
                    'text': f"Sentiment Polarity Distribution for '{title}'",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title="Polarity",
                yaxis_title="Count")
            st.plotly_chart(fig)
    app()

# Create a dictionary of tabs
tabs = {
    "Cosine Similarity": first_tab,
    "TMDB": second_tab,
    "Sentimental Analysis" : third_tab
}

# Create a function to run the selected tab
def run_tab():
    tab = st.sidebar.selectbox('Select a tab', options=list(tabs.keys()))
    tabs[tab]()

# Run the app
if __name__ == '__main__':
    run_tab()

