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
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError("File anime_MAL_cleaned.csv tidak ditemukan di folder data/")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# Build similarity matrices
def build_similarity_matrices(df):
    # Gabungkan semua fitur menjadi satu combined text (TANPA pembobotan)
    # Semua fitur digabung sebagai satu dokumen per anime, lalu TF-IDF dihitung
    df['combined_features'] = (
        df['sinopsis_clean'].fillna('') + ' ' +
        df['genre_clean'].fillna('') + ' ' +
        df['studio_clean'].fillna('') + ' ' +
        df['jenis_clean'].fillna('')
    )
    
    # Buat SATU combined matrix dari semua fitur yang sudah digabung
    tfidf_combined = TfidfVectorizer(
        max_features=6000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    combined_matrix = tfidf_combined.fit_transform(df['combined_features'])
    
    return {
        'combined': combined_matrix
    }

def get_recommendations(anime_title, df, matrices, method='combined', n_recommendations=5):
    # Cari index anime
    idx = df[df['judul'].str.lower() == anime_title.lower()].index
    
    if len(idx) == 0:
        return None, "Anime tidak ditemukan dalam database", None
    
    idx = idx[0]
    
    # Gunakan combined features (semua fitur digabung menjadi satu, tanpa pembobotan)
    sim_scores = cosine_similarity(matrices['combined'][idx], matrices['combined']).flatten()
    weight_info = "Gabungkan fitur (sinopsis+genre+studio+jenis) → TF-IDF → Cosine Similarity"
    
    # Dapatkan index anime yang mirip (kecuali anime itu sendiri)
    similar_indices = sim_scores.argsort()[-n_recommendations-1:-1][::-1]
    
    # Buat dataframe hasil rekomendasi
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = sim_scores[similar_indices]
    
    return recommendations, None, weight_info