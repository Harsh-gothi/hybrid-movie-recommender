import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import os

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK data not found. Downloading...")
    nltk.download('punkt')
    print("NLTK data downloaded.")

# Initialize stemmer
ps = PorterStemmer()

def stem_word_list(word_list):
    return [ps.stem(word) for word in word_list]

def remove_spaces(list_of_strings):
    return [item.replace(" ", "") for item in list_of_strings]

print("Starting data processing...")

# --- Load Data ---
print("Loading CSV files...")
movies_df = pd.read_csv('files/movies_metadata.csv', low_memory=False)
credits_df = pd.read_csv('files/credits.csv')
keywords_df = pd.read_csv('files/keywords.csv')
links_df = pd.read_csv('files/links_small.csv')
ratings_df = pd.read_csv('files/ratings_small.csv')

# --- A. CONTENT-BASED MODEL PREP ---
print("Processing movie metadata...")
movies_df = movies_df[['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count']]

movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
movies_df = movies_df.dropna(subset=['id'])
movies_df['id'] = movies_df['id'].astype('int')

credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')

df_merged = pd.merge(movies_df, credits_df, on='id')
df_final_content = pd.merge(df_merged, keywords_df, on='id')

def parse_literal(obj):
    try:
        return ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []

print("Parsing JSON fields...")
df_final_content['genres'] = df_final_content['genres'].apply(parse_literal).apply(lambda x: [i['name'] for i in x])
df_final_content['keywords'] = df_final_content['keywords'].apply(parse_literal).apply(
    lambda x: [i['name'] for i in x])
df_final_content['cast'] = df_final_content['cast'].apply(parse_literal).apply(lambda x: [i['name'] for i in x][:5])
df_final_content['crew'] = df_final_content['crew'].apply(parse_literal).apply(
    lambda x: [i['name'] for i in x if i.get('job') == 'Director'])

df_final_content['cast'] = df_final_content['cast'].apply(remove_spaces)
df_final_content['crew'] = df_final_content['crew'].apply(remove_spaces)
df_final_content['genres'] = df_final_content['genres'].apply(remove_spaces)
df_final_content['keywords'] = df_final_content['keywords'].apply(remove_spaces)

df_final_content['overview'] = df_final_content['overview'].fillna('')
df_final_content['overview'] = df_final_content['overview'].apply(lambda x: x.split())

print("Stemming words...")
df_final_content['overview'] = df_final_content['overview'].apply(stem_word_list)
df_final_content['genres'] = df_final_content['genres'].apply(stem_word_list)
df_final_content['keywords'] = df_final_content['keywords'].apply(stem_word_list)

df_final_content['tags'] = df_final_content['overview'] + \
                           df_final_content['genres'] + \
                           df_final_content['keywords'] + \
                           df_final_content['cast'] + \
                           df_final_content['crew']

final_df = df_final_content[['id', 'title', 'tags', 'vote_average', 'vote_count']].copy()

final_df['tags'] = final_df['tags'].apply(lambda x: " ".join(x))
final_df = final_df.drop_duplicates(subset=['tags'])

# --- Link with MovieLens IDs ---
print("Linking with MovieLens IDs...")
links_df = links_df[['movieId', 'tmdbId']]
links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
links_df = links_df.dropna(subset=['tmdbId'])
links_df['tmdbId'] = links_df['tmdbId'].astype('int')
links_df['movieId'] = pd.to_numeric(links_df['movieId'], errors='coerce')
links_df = links_df.dropna(subset=['movieId'])
links_df['movieId'] = links_df['movieId'].astype('int')

final_df_hybrid = pd.merge(final_df, links_df, left_on='id', right_on='tmdbId')
final_df_hybrid = final_df_hybrid.reset_index(drop=True)

# --- Train TF-IDF ---
print("Training TF-IDF model...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(final_df_hybrid['tags'])

# --- Popularity Score Params ---
C = final_df_hybrid['vote_average'].mean()
m = final_df_hybrid['vote_count'].quantile(0.90)

# --- B. COLLABORATIVE FILTERING (SVD) PREP ---
print("Building user-item matrix...")
ratings_df = ratings_df[['userId', 'movieId', 'rating']]
user_item_pivot = ratings_df.pivot(index='userId', columns='movieId', values='rating')
user_means = user_item_pivot.mean(axis=1).fillna(0)
user_item_pivot_demeaned = user_item_pivot.subtract(user_means, axis=0).fillna(0)
user_item_matrix = csr_matrix(user_item_pivot_demeaned.values)

print("Training SVD model...")
svd_model = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd_model.fit_transform(user_item_matrix)
item_factors = svd_model.components_

predicted_demeaned_ratings = user_factors.dot(item_factors)
predicted_ratings_original_scale = predicted_demeaned_ratings + user_means.values.reshape(-1, 1)

user_id_to_index = {user_id: index for index, user_id in enumerate(user_item_pivot.index)}
movie_id_to_index = {movie_id: index for index, movie_id in enumerate(user_item_pivot.columns)}

# --- SAVE EVERYTHING TO PICKLE ---
print("Saving to pickle file...")
data_to_save = {
    'final_df_hybrid': final_df_hybrid,
    'vectors': vectors,
    'predicted_ratings': predicted_ratings_original_scale,
    'user_map': user_id_to_index,
    'movie_map': movie_id_to_index,
    'm': m,
    'C': C
}

with open('movie_recommender_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("✅ SUCCESS! Pickle file created: movie_recommender_data.pkl")
print(f"File size: {os.path.getsize('movie_recommender_data.pkl') / (1024*1024):.2f} MB")