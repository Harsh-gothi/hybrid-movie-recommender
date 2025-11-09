# 🎬 Hybrid Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent movie recommendation engine that combines content-based and collaborative filtering to deliver personalized movie suggestions. Built with Streamlit and powered by machine learning algorithms including TF-IDF and Singular Value Decomposition (SVD).

![Demo Screenshot](images/img.png)

## 🌟 Overview

This project implements a sophisticated hybrid recommendation system that leverages both content similarity and user behavior patterns to suggest movies. The system intelligently adapts its recommendation strategy based on whether you're a new or existing user.

### Demo Screenshots

**For Existing Users (Personalized)**

![Existing User](images/existing_user.jpg)

**For New Users (Popularity-Based)**

![New User](images/new_user.jpg)

## ✨ Features

- **Hybrid Recommendation Engine:** Combines two powerful models:
  - **Content-Based Filtering (TF-IDF):** Analyzes movie metadata including plot, cast, director, genres, and keywords
  - **Collaborative Filtering (SVD):** Learns from user rating patterns to predict preferences
  
- **Personalized Recommendations:** Tailored suggestions based on your historical rating behavior and preferences

- **Cold-Start Solution:** New users receive intelligent recommendations based on popular movies with similar content to their selection

- **Interactive Web Interface:** Clean, responsive Streamlit UI for seamless user experience

- **Dynamic Visual Content:** Real-time movie poster fetching from TMDb API with fallback placeholders

- **Performance Optimized:** Efficient caching mechanisms to minimize loading times

## 🤖 How It Works

The system employs a two-stage hybrid approach:

### Stage 1: Content-Based Filtering (Candidate Generation)

1. **Feature Engineering:** Each movie is represented by a "tags" corpus combining:
   - Plot overview (stemmed words)
   - Genres
   - Keywords
   - Top 5 cast members
   - Director name

2. **Vectorization:** Text data is transformed using `TfidfVectorizer` to create numerical feature vectors

3. **Similarity Calculation:** When you select a movie, the system computes **cosine similarity** against all other movies to find the top 100 most similar candidates

### Stage 2: Collaborative Filtering (Re-ranking)

1. **Matrix Factorization:** A user-item rating matrix is decomposed using **Truncated SVD** to discover latent features

2. **Personalized Scoring:** The system predicts ratings for the 100 candidates specifically for your user profile

3. **Adaptive Strategy:**
   - **Existing Users:** Candidates are re-ranked using predicted ratings from the SVD model
   - **New Users:** Candidates are ranked by weighted popularity score (IMDb formula)

4. **Final Output:** Top 10 movies from the re-ranked list are displayed with posters

### Recommendation Formula

**Existing Users:**
```
Score = SVD_predicted_rating(user, movie)
```

**New Users (Weighted Rating):**
```
Score = (v/(v+m) × R) + (m/(v+m) × C)
where:
  v = number of votes for the movie
  m = minimum votes required (90th percentile)
  R = average rating of the movie
  C = mean rating across all movies
```

## 🛠️ Tech Stack

**Core Framework:**
- [Streamlit](https://streamlit.io/) - Web application framework

**Machine Learning:**
- [scikit-learn](https://scikit-learn.org/) - TF-IDF vectorization, SVD, cosine similarity
- [NumPy](https://numpy.org/) - Numerical computations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [SciPy](https://scipy.org/) - Sparse matrix operations

**Natural Language Processing:**
- [NLTK](https://www.nltk.org/) - Text preprocessing and stemming

**API Integration:**
- [TMDb API](https://www.themoviedb.org/documentation/api) - Movie poster retrieval
- [Requests](https://requests.readthedocs.io/) - HTTP client

**Environment Management:**
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable handling

## 📦 Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- TMDb API key (free)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Harsh-gothi/hybrid-movie-recommender.git
cd hybrid-movie-recommender
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```txt
streamlit
pandas
numpy
nltk
scikit-learn
scipy
requests
python-dotenv
```

### Step 4: Download Datasets

Download the [MovieLens and TMDb Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle.

Place these files in the project root directory:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links_small.csv`
- `ratings_small.csv`

### Step 5: Configure API Key

1. Sign up for a free API key at [TMDb](https://www.themoviedb.org/settings/api)
2. Create a `.env` file in the project root:

```bash
TMDB_API_KEY=your_api_key_here
```

**Important:** Add `.env` to your `.gitignore` to keep your API key secure.

## 🚀 Usage

### Running the Application

From the project directory with your virtual environment activated:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Recommender

1. **Select a Movie:** Choose a movie you like from the dropdown menu
2. **Enter User ID:** 
   - Enter a valid user ID (1 to 671) for personalized recommendations
   - Enter `-1` or `0` for new user recommendations
3. **Get Recommendations:** Click the button to generate your personalized movie list
4. **Explore Results:** Browse the top 10 recommended movies with posters and scores

### Example Use Cases

**Scenario 1: Existing User**
```
Selected Movie: "The Dark Knight"
User ID: 42
Result: 10 personalized movies based on your rating history
```

**Scenario 2: New User**
```
Selected Movie: "Inception"
User ID: -1
Result: 10 popular movies similar to Inception
```

## 📁 Project Structure

```
hybrid-movie-recommender/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not in repo)
├── .gitignore                 # Git ignore file
│
├── movies_metadata.csv        # Movie metadata (from Kaggle)
├── credits.csv                # Cast and crew data
├── keywords.csv               # Movie keywords
├── links_small.csv            # MovieLens to TMDb ID mapping
├── ratings_small.csv          # User ratings data
│
├── images/                    # Screenshots for README
│   ├── img.png
│   ├── existing_user.jpg
│   └── new_user.jpg
│
└── README.md                  # This file
```

## 📊 Dataset

This project uses the **MovieLens and TMDb Dataset** from Kaggle, which contains:

- **45,000+ movies** from TMDb with metadata
- **100,000+ ratings** from 700+ users
- Cast, crew, keywords, and genre information
- Links between MovieLens IDs and TMDb IDs

**Dataset Citation:**
```
Rounak Banik. (2017). The Movies Dataset. 
Retrieved from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
```

**Key Statistics:**
- Movies: ~45,000
- Users: 671
- Ratings: 100,004
- Time Period: 1995-2017
- Rating Scale: 0.5 to 5.0 stars

## 🔑 API Configuration

The application uses the TMDb API for fetching movie posters. The API has the following characteristics:

- **Rate Limit:** 40 requests per 10 seconds
- **Cost:** Free tier available
- **Response:** High-quality poster images (500px width)
- **Fallback:** Placeholder images for missing posters

The app includes built-in rate limiting (`time.sleep(0.05)`) and error handling to ensure reliable poster fetching.

## 🐛 Troubleshooting

### Common Issues

**1. "Missing required file" error**
```
Solution: Ensure all 5 CSV files are in the project root directory
```

**2. "NLTK data not found"**
```
Solution: The app will auto-download required NLTK data on first run
```

**3. "Invalid API key" error**
```
Solution: Verify your TMDb API key in the .env file is correct
```

**4. Movie posters not loading**
```
Solution: Check your internet connection and TMDb API key validity
```

**5. "Movie not found" error**
```
Solution: The selected movie might not be in the processed dataset. Try another movie.
```

### Performance Tips

- **First Load:** Initial model training takes 1-2 minutes but is cached
- **Subsequent Loads:** Near-instant thanks to Streamlit's `@st.cache_data`
- **Clear Cache:** Use `streamlit cache clear` if you update data files

### Debug Mode

Run with verbose output:
```bash
streamlit run app.py --logger.level=debug
```

## 🙏 Acknowledgments

- **Dataset:** [Rounak Banik](https://www.kaggle.com/rounakbanik) for the comprehensive movie dataset
- **API:** [The Movie Database (TMDb)](https://www.themoviedb.org/) for poster images
- **Framework:** [Streamlit](https://streamlit.io/) team for the amazing web framework
- **Inspiration:** Hybrid recommendation research papers and Netflix recommendation system

## 📧 Contact

For questions, suggestions, or collaboration:

- **GitHub:** [@Harsh-gothi](https://github.com/Harsh-gothi)
- **Email:** harshgothi6453@gmail.com
- **LinkedIn:** [Harsh Gothi](https://www.linkedin.com/in/harsh-gothi)

---

⭐ **If you found this project helpful, please consider giving it a star!**
