import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi utama untuk load data dan build model
def load_data_and_build_model():
    try:
        df = pd.read_csv('data/anime_MAL_clean.csv')

        # Pastikan kolom fitur_stem ada
        if 'fitur' not in df.columns:
            raise Exception("Kolom 'fitur' tidak ditemukan")

        df['fitur'] = df['fitur'].fillna('')

        # TF-IDF hanya dari fitur
        tfidf = TfidfVectorizer(
            max_features=6000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = tfidf.fit_transform(df['fitur'])

        return df, tfidf_matrix

    except FileNotFoundError:
        raise FileNotFoundError("File anime_dataset_clean.csv tidak ditemukan di folder data/")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

# Fungsi untuk kompatibilitas dengan app.py
def load_data():
    """Wrapper function untuk kompatibilitas"""
    df, _ = load_data_and_build_model()
    return df

def build_similarity_matrices(df):
    """Wrapper function untuk kompatibilitas"""
    _, tfidf_matrix = load_data_and_build_model()
    return tfidf_matrix

# Fungsi rekomendasi content-based
def get_recommendations(anime_title, df, tfidf_matrix, n_recommendations=5):
    # Cari index anime
    idx = df[df['judul'].str.lower() == anime_title.lower()].index

    if len(idx) == 0:
        return pd.DataFrame(), "Anime tidak ditemukan dalam database"

    idx = idx[0]

    # Hitung cosine similarity
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Ambil anime paling mirip (kecuali dirinya sendiri)
    similar_indices = sim_scores.argsort()[-n_recommendations-1:-1][::-1]

    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = sim_scores[similar_indices]

    return recommendations, None