import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Fungsi untuk membersihkan teks
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

# Load data
def load_data():
    try:
        df = pd.read_csv('data/anime_MAL_cleaned.csv')
        df['sinopsis_clean'] = df['sinopsis'].apply(clean_text)
        df['genre_clean'] = df['genre'].apply(clean_text)
        df['studio_clean'] = df['studio'].apply(clean_text)
        df['jenis_clean'] = df['jenis_tayangan'].apply(clean_text)
        
        # Gabungkan semua fitur dengan bobot
        df['combined_features'] = (
            df['sinopsis_clean'] + ' ' + 
            df['genre_clean'] + ' ' + df['genre_clean'] + ' ' +  # Genre diberi bobot 2x
            df['studio_clean'] + ' ' +
            df['jenis_clean']
        )
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError("File anime_MAL_cleaned.csv tidak ditemukan di folder data/")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# Build similarity matrices
def build_similarity_matrices(df):
    # TF-IDF untuk sinopsis
    tfidf_sinopsis = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    sinopsis_matrix = tfidf_sinopsis.fit_transform(df['sinopsis_clean'])
    
    # TF-IDF untuk genre
    tfidf_genre = TfidfVectorizer(
        max_features=500,
        stop_words='english'
    )
    genre_matrix = tfidf_genre.fit_transform(df['genre_clean'])
    
    # TF-IDF untuk studio
    tfidf_studio = TfidfVectorizer(
        max_features=200,
        stop_words='english'
    )
    studio_matrix = tfidf_studio.fit_transform(df['studio_clean'])
    
    # TF-IDF untuk jenis tayangan
    tfidf_jenis = TfidfVectorizer(
        max_features=50,
        stop_words='english'
    )
    jenis_matrix = tfidf_jenis.fit_transform(df['jenis_clean'])
    
    # TF-IDF untuk combined features
    tfidf_combined = TfidfVectorizer(
        max_features=6000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    combined_matrix = tfidf_combined.fit_transform(df['combined_features'])
    
    return {
        'sinopsis': sinopsis_matrix,
        'genre': genre_matrix,
        'studio': studio_matrix,
        'jenis': jenis_matrix,
        'combined': combined_matrix
    }

# Fungsi rekomendasi dengan multiple methods
def get_recommendations(anime_title, df, matrices, method='hybrid', n_recommendations=5):
    # Cari index anime
    idx = df[df['judul'].str.lower() == anime_title.lower()].index
    
    if len(idx) == 0:
        return None, "Anime tidak ditemukan dalam database", None
    
    idx = idx[0]
    
    # Pilih metode perhitungan similarity
    if method == 'sinopsis':
        sim_scores = cosine_similarity(matrices['sinopsis'][idx], matrices['sinopsis']).flatten()
        weight_info = "100% Sinopsis"
    elif method == 'genre':
        sim_scores = cosine_similarity(matrices['genre'][idx], matrices['genre']).flatten()
        weight_info = "100% Genre"
    elif method == 'combined':
        sim_scores = cosine_similarity(matrices['combined'][idx], matrices['combined']).flatten()
        weight_info = "Sinopsis + Genre (weighted)"
    else:  # hybrid
        # Hitung similarity untuk masing-masing fitur
        sim_sinopsis = cosine_similarity(matrices['sinopsis'][idx], matrices['sinopsis']).flatten()
        sim_genre = cosine_similarity(matrices['genre'][idx], matrices['genre']).flatten()
        sim_studio = cosine_similarity(matrices['studio'][idx], matrices['studio']).flatten()
        sim_jenis = cosine_similarity(matrices['jenis'][idx], matrices['jenis']).flatten()
        
        # Kombinasikan dengan bobot
        sim_scores = (
            0.60 * sim_sinopsis +
            0.25 * sim_genre +
            0.10 * sim_studio +
            0.05 * sim_jenis
        )
        weight_info = "60% Sinopsis + 25% Genre + 10% Studio + 5% Jenis"
    
    # Dapatkan index anime yang mirip (kecuali anime itu sendiri)
    similar_indices = sim_scores.argsort()[-n_recommendations-1:-1][::-1]
    
    # Buat dataframe hasil rekomendasi
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = sim_scores[similar_indices]
    
    return recommendations, None, weight_info