import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf=TfidfVectorizer(stop_words='english')
from difflib import get_close_matches
df=pd.read_csv("netflix_titles.csv")
print(df.head())
print("dataset shape:",df.shape)
print(df.columns)
print(df[['title', 'type', 'listed_in', 'description']].head())
print(df.isnull().sum())
print("duplicate rows:",df.duplicated().sum())
movies=df[["title","type","listed_in","description"]].copy()
print(movies.head())
movies['combined_features']=(
    movies['listed_in']+' '+movies['description']
)
print(movies[['title','combined_features','type']].head())
#TF-IDF use karenge, jo text ko numerical form mein convert karta hai.
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
similarity=cosine_similarity(tfidf_matrix)
print("similarity matrix shape:",similarity.shape)
def recommend(title):
    if title not in movies['title'].values:
        print("Title not found!")
        return
    index = movies[movies['title'] == title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]
    for i, score in scores:
        print(movies.iloc[i]['title'])
#recommend('The Queen')
#recommend('Stranger Things')
#recommend('The Crown')
#title = input("Enter a Netflix title: ")
#recommend(title)
"""agar user kisi TV Show ka naam de, toh recommendations mein movies bhi aa sakti hain.
isliye system ko improve karenge taaki user ko usi type ke results mil sakein."""
def recommend(title):
    matches = movies[movies['title'].str.lower() == title.lower()]
    if matches.empty:
        titles = movies['title'].dropna().tolist()
        suggestions = get_close_matches(
            title,
            titles,
            n=5,
            cutoff=0.4
            )
        
        if suggestions:
            print("Title not found. Did you mean:")
            for suggestion in suggestions:
                print("-", suggestion)
            else:
                print("No matching titles found.")
                return
            index = matches.index[0]
            selected_type = movies.loc[index, 'type']
            scores = list(enumerate(similarity[index]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            count = 0
            for i, score in scores:
                if i == index:
                    continue
                if movies.iloc[i]['type'] == selected_type:
                    print(movies.iloc[i]['title'], "-", round(score * 100, 2), "% match")
                    count += 1
                    if count == 5:
                        break
                    
                    
title = input("Enter a Netflix title: ")
recommend(title)

