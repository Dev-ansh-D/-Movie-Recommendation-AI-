# 🎬 Simple Movie Recommendation AI  
A lightweight **content-based movie recommender system** built using Python, Pandas, and TF-IDF vectorization.

*******This project is a **Streamlit web app**, so you must run it using the Streamlit command.*******

The AI recommends movies based on **genres, summary, cast, and director**, and also displays the **IMDb rating**.

This project uses natural language processing (NLP) and cosine similarity to match user queries like:


and returns the most relevant movies.

---

## 🚀 Features

### ✔ Content-Based Recommender
- No training labels needed  
- Works entirely using text similarity  

### ✔ Multi-Feature Matching
The recommender uses:
- **Genre**
- **Summary**
- **Director**
- **Cast**
- **TF-IDF keyword extraction**

### ✔ User-Friendly Query System
You can search using:
- Keywords (`alien`, `battle`, `time travel`)
- Genres (`action`, `sci-fi`, `romance`)
- Actors (`Tom Cruise`, `Emma Stone`)
- Plot descriptions (`space mission gone wrong`)

### ✔ Shows IMDb Ratings
Each recommendation displays:
- 🎬 Movie Title  
- ⭐ IMDb Rating  
- 🎭 Genre  

---

## 📂 Dataset
This project uses a custom IMDb dataset containing:

- `Title`
- `Genre`
- `Director`
- `Cast`
- `Summary`
- `Rating`
- Other metadata (votes, duration, certificate, etc.)

> You must place your dataset as:
