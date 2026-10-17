# 🎬 Netflix Content-Based Recommendation System

A beginner-friendly Machine Learning project that recommends similar Netflix movies and TV shows based on their genres and descriptions.

## Features

- Recommends similar movies and TV shows.
- Uses TF-IDF Vectorization to convert text into numerical features.
- Uses Cosine Similarity to measure similarity between titles.
- Displays recommendations through a Streamlit web application.
- Shows genre, description, and similarity score.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Files

- `netflix_recommendation.py` — recommendation logic.
- `app.py` — Streamlit web application.

## Dataset

This project uses the Netflix Movies and TV Shows dataset.

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/shivamb/netflix-shows

Place `netflix_titles.csv` in the project folder before running the application.

## How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit
#Run the web application:
streamlit run app.py
