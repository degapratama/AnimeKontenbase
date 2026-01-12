import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

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

        return df, tfidf_matrix, tfidf

    except FileNotFoundError:
        raise FileNotFoundError("File anime_dataset_clean.csv tidak ditemukan di folder data/")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

# Fungsi untuk kompatibilitas dengan app.py
def load_data():
    """Wrapper function untuk kompatibilitas"""
    df, _, _ = load_data_and_build_model()
    return df

def build_similarity_matrices(df):
    """Wrapper function untuk kompatibilitas"""
    _, tfidf_matrix, _ = load_data_and_build_model()
    return tfidf_matrix

# Fungsi untuk menghitung metrik evaluasi
def calculate_evaluation_metrics(recommendations, original_anime_idx, df, top_k=5):
    """
    Menghitung Precision, Recall, dan F1-Score untuk rekomendasi
    berdasarkan genre overlap
    """
    try:
        # Genre anime asli
        if 'genre' not in df.columns:
            return None
        
        original_genres = set()
        if pd.notna(df.iloc[original_anime_idx]['genre']):
            original_genres = set(str(df.iloc[original_anime_idx]['genre']).split(','))
            original_genres = {g.strip() for g in original_genres}
        
        # Hitung metrik untuk setiap rekomendasi
        precisions = []
        recalls = []
        f1_scores = []
        
        for _, rec in recommendations.iterrows():
            rec_genres = set()
            if pd.notna(rec['genre']):
                rec_genres = set(str(rec['genre']).split(','))
                rec_genres = {g.strip() for g in rec_genres}
            
            # Hitung intersection
            intersection = original_genres.intersection(rec_genres)
            
            # Precision = TP / (TP + FP)
            precision = len(intersection) / len(rec_genres) if len(rec_genres) > 0 else 0
            
            # Recall = TP / (TP + FN)
            recall = len(intersection) / len(original_genres) if len(original_genres) > 0 else 0
            
            # F1-Score = 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Hitung rata-rata
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'individual_precisions': precisions,
            'individual_recalls': recalls,
            'individual_f1_scores': f1_scores
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# Fungsi rekomendasi content-based
def get_recommendations(anime_title, df, tfidf_matrix, n_recommendations=5):
    # Cari index anime
    idx = df[df['judul'].str.lower() == anime_title.lower()].index

    if len(idx) == 0:
        return pd.DataFrame(), None, None, "Anime tidak ditemukan dalam database"

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
    
    # Hitung metrik evaluasi
    evaluation_metrics = calculate_evaluation_metrics(recommendations, idx, df, n_recommendations)

    return recommendations, evaluation_metrics, idx, None