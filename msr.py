import requests
import streamlit as st


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
st.set_page_config(page_title="Movie Recommendation System", page_icon=":clapper:")
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

