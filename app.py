import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import requests
import time
import pickle

load_dotenv()


# --- 1. HELPER FUNCTIONS (GLOBAL) ---

# Get TMDb API from .env
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize the NLTK stemmer
ps = PorterStemmer()

def stem_word_list(word_list):
    return [ps.stem(word) for word in word_list]


def remove_spaces(list_of_strings):
    return [item.replace(" ", "") for item in list_of_strings]


def weighted_rating(movie_data, m, C):
    v = movie_data['vote_count']
    R = movie_data['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)


def predict_rating(user_id, movie_id, predictions_matrix, user_map, movie_map):
    try:
        user_index = user_map[user_id]
        movie_index = movie_map[movie_id]
        rating = predictions_matrix[user_index, movie_index]
        return rating
    except KeyError:
        return 3.0


# --- Function to get poster from the API ---
@st.cache_data
def fetch_poster(tmdb_id):
    """
    Fetches the poster URL for a movie from the TMDb API.
    Uses a reliable placeholder for missing images.
    """
    # Define a reliable placeholder
    placeholder_url = 'https://dummyimage.com/500x750/363945/ffffff.png&text=No+Poster'

    if pd.isna(tmdb_id):
        return placeholder_url + '+Available'

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses (404, 500, etc.)

        data = response.json()
        poster_path = data.get('poster_path')

        if poster_path:
            # Build the full, high-quality URL
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            # Movie exists, but has no poster
            return placeholder_url

    except (requests.RequestException, ValueError, TypeError):
        # Handle API errors, bad JSON, or bad tmdb_id
        return placeholder_url

# --- 2. MAIN DATA LOADING & MODEL TRAINING (CACHED) ---

@st.cache_data
def load_models():
    """
    Loads all data, performs all cleaning and merging,
    and trains both the TF-IDF and SVD models.
    """

    # --- Check for NLTK data ---
    with open('movie_recommender_data.pkl', 'rb') as f:
        data = pickle.load(f)

    return (
        data['final_df_hybrid'],
        data['vectors'],
        data['predicted_ratings'],
        data['user_map'],
        data['movie_map'],
        data['m'],
        data['C']
    )


# --- 3. HYBRID RECOMMENDATION FUNCTION (GLOBAL) ---

def get_hybrid_recommendations(user_id, movie_title, n_recommendations=10, final_df=None, vectors=None, SVD_preds=None,
                               user_map=None, movie_map=None, m=None, C=None):
    """
    Gets hybrid recommendations.
    Returns a list of dictionaries, each with title, tmdb_id, and score.
    """
    recommendations = []
    header = ""

    # --- 1. MODEL 1: CONTENT-BASED FILTERING (Get Candidates) ---
    try:
        movie_index = final_df[final_df['title'] == movie_title].index[0]
    except IndexError:
        return [{"title": "Error: Movie not found.", "tmdb_id": None, "score": 0}], ""

    movie_vector = vectors[movie_index]
    sim_scores = cosine_similarity(movie_vector, vectors)[0]
    candidate_list = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])[1:101]

    # --- 2. CHECK IF USER IS NEW ---
    if user_id not in user_map:
        header = f"User {user_id} is a new user. Recommending POPULAR similar movies (non-personalized):"

        candidate_indices = [i[0] for i in candidate_list]
        candidate_movies = final_df.iloc[candidate_indices].copy()

        candidate_movies['popularity_score'] = candidate_movies.apply(weighted_rating, axis=1, m=m, C=C)
        candidate_movies = candidate_movies.sort_values('popularity_score', ascending=False)

        for (idx, row) in candidate_movies.head(n_recommendations).iterrows():
            recommendations.append({
                "title": row['title'],
                "tmdb_id": row['tmdbId'],  # <-- UPDATED: Pass tmdbId
                "score": f"Popularity: {row['popularity_score']:.2f}"
            })

    else:  # --- 3. EXISTING USER (Hybrid Model) ---
        header = f"Hybrid recommendations for user {user_id} (based on '{movie_title}'):"

        ranked_candidates = []
        for (candidate_index, content_score) in candidate_list:
            row = final_df.iloc[candidate_index]
            candidate_movie_id = row['movieId']
            candidate_title = row['title']
            candidate_tmdb_id = row['tmdbId']  # <-- UPDATED: Get tmdbId

            predicted_rating = predict_rating(user_id, candidate_movie_id, SVD_preds, user_map, movie_map)

            ranked_candidates.append({
                "title": candidate_title,
                "tmdb_id": candidate_tmdb_id,  # <-- UPDATED: Pass tmdbId
                "score": predicted_rating
            })

        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        for rec in ranked_candidates[:n_recommendations]:
            rec["score"] = f"Predicted: {rec['score']:.2f}"
            recommendations.append(rec)

    return recommendations, header


# --- 4. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🎬 Hybrid Movie Recommender System")
st.markdown(
    "Combines **Content-Based Filtering** (what's in the movie) with **Collaborative Filtering** (what similar users like).")

# --- Check for required files ---
required_files = ['files/movies_metadata.csv', 'files/credits.csv', 'files/keywords.csv', 'files/links_small.csv', 'files/ratings_small.csv']
files_missing = False
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Error: Missing required file: `{file}`. Please add it to the app folder.")
        files_missing = True

# Only run the app if files are present
if not files_missing:
    # Load models (this runs only once)
    with st.spinner('Loading all models and data... This may take a minute.'):
        final_df_hybrid, vectors, SVD_preds, user_map, movie_map, m, C = load_models()
    st.success('Models loaded successfully!')

    # Get user input
    movie_list = final_df_hybrid['title'].unique()
    user_list = list(user_map.keys())

    col1, col2 = st.columns(2)

    with col1:
        selected_movie = st.selectbox(
            "Select a movie you like:",
            movie_list
        )

    with col2:
        user_id = st.number_input(
            f"Enter your User ID (e.g., 1 to {len(user_list)}):",
            min_value=-1,
            max_value=len(user_list),
            value=1
        )
        st.info("Enter **-1** or **0** for a 'New User' recommendation.")

    # Generate recommendations
    if st.button("Get Recommendations"):
        with st.spinner('Finding recommendations...'):
            recommendations, header = get_hybrid_recommendations(
                user_id,
                selected_movie,
                final_df=final_df_hybrid,
                vectors=vectors,
                SVD_preds=SVD_preds,
                user_map=user_map,
                movie_map=movie_map,
                m=m,
                C=C
            )

            st.subheader(header)
            st.markdown("---")

            # --- Display posters in a grid ---
            # Create 5 columns for the top 5
            cols = st.columns(5)
            for i in range(min(5, len(recommendations))):
                with cols[i]:
                    # --- Call the API to get the poster URL ---
                    poster_url = fetch_poster(recommendations[i]['tmdb_id'])
                    st.image(poster_url)
                    st.caption(f"{recommendations[i]['title']} ({recommendations[i]['score']})")
                    # Add a small delay to not spam the API
                    time.sleep(0.05)

                    # Create another 5 columns for the next 5
            if len(recommendations) > 5:
                cols = st.columns(5)
                for i in range(5, min(10, len(recommendations))):
                    with cols[i - 5]:
                        # --- Call the API to get the poster URL ---
                        poster_url = fetch_poster(recommendations[i]['tmdb_id'])
                        st.image(poster_url)
                        st.caption(f"{recommendations[i]['title']} ({recommendations[i]['score']})")
                        # Add a small delay
                        time.sleep(0.05)