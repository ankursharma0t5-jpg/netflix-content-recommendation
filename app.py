import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬")

st.title("🎬 Netflix Recommendation System")
st.write("Find shows and movies similar to your favorites!")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df[["title", "type", "listed_in", "description"]].copy()
    df = df.dropna(subset=["title"])
    df["listed_in"] = df["listed_in"].fillna("")
    df["description"] = df["description"].fillna("")
    df["combined_features"] = df["listed_in"] + " " + df["description"]
    return df

@st.cache_resource
def build_model(features):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(features)
    similarity = cosine_similarity(matrix)
    return similarity

df = load_data()
similarity = build_model(df["combined_features"].tolist())

selected_title = st.selectbox(
    "Choose a movie or show:",
    sorted(df["title"].unique())
    )

if st.button("Recommend 🎯"):
    matches = df.index[df["title"] == selected_title].tolist()
    selected_index = matches[0]
    scores = list(enumerate(similarity[selected_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for index, score in scores:
        if index == selected_index:
            continue
        if df.iloc[index]["type"] != df.iloc[selected_index]["type"]:
            continue
        recommendations.append((index, score))
        if len(recommendations) == 5:
            break
        st.subheader("✨ Recommended for you")
        if recommendations:
            for index, score in recommendations:
                st.markdown(f"### 🍿 {df.iloc[index]['title']}")
                st.write(f"**Genre:** {df.iloc[index]['listed_in']}")
                st.write(f"**Similarity:** {score:.1%}")
                st.write(f"**Description:** {df.iloc[index]['description']}")
                st.divider()
        else:
            st.info("No recommendations found.")
