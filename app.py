import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# Load and Prepare Dataset
# -------------------------------
df = pd.read_csv("imdb-movies-dataset.csv")

# Clean missing values
df['Genre'] = df['Genre'].fillna('')
df['Description'] = df['Description'].fillna('')
df['Director'] = df['Director'].fillna('')
df['Cast'] = df['Cast'].fillna('')
df['Poster'] = df['Poster'].fillna('')
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

# Create combined features column
df["features"] = (
    df["Genre"] + " " +
    df["Description"] + " " +
    df["Director"] + " " +
    df["Cast"]
)

# TF-IDF & similarity matrix
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["features"])

def recommend(query, top_n=9):
    query_vec = vectorizer.transform([query])
    sim = cosine_similarity(query_vec, feature_matrix).flatten()
    idx = sim.argsort()[::-1][:top_n]
    return df.iloc[idx]


# -------------------------------
# Streamlit Modern UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>
        🎬 Movie Recommendation System
    </h1>
    <p style='text-align: center; font-size: 18px;'>
        Find movies by keywords like <b>alien battle sci-fi Tom Cruise</b>
    </p>
""", unsafe_allow_html=True)

# Search Box
query = st.text_input("🔍 Enter movie keywords, genre, actor, or description:", 
                      placeholder="Example: sci-fi alien battle Tom Cruise")

# Recommend when user types something
if query:
    results = recommend(query)

    st.write(f"### 🎯 Top Results for: **{query}**")

    # Display movies in 3 columns (like Netflix)
    cols = st.columns(3)

    for i, (_, movie) in enumerate(results.iterrows()):
        col = cols[i % 3]

        with col:
            # Poster
            try:
                response = requests.get(movie["Poster"])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=250)
            except:
                st.image("https://via.placeholder.com/250x350?text=No+Image")

            # Title
            st.markdown(
                f"<h4 style='margin-top: 10px;'>{movie['Title']}</h4>", 
                unsafe_allow_html=True
            )

            # Rating
            st.markdown(f"⭐ **Rating:** {movie['Rating']}")

            # Genre
            st.markdown(f"🎭 **Genre:** {movie['Genre']}")

            st.markdown("<hr>", unsafe_allow_html=True)
