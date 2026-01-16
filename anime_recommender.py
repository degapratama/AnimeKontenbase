import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk load data
def load_data():
    try:
        df = pd.read_csv('data/anime_MAL_clean.csv')
        df['fitur'] = df['fitur'].fillna('')
        return df
    except FileNotFoundError:
        raise FileNotFoundError("File anime_MAL_clean.csv tidak ditemukan di folder data/")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# Fungsi untuk build similarity matrix
def build_similarity_matrices(df):
    try:
        # Pastikan kolom fitur ada
        if 'fitur' not in df.columns:
            raise ValueError("Kolom 'fitur' tidak ditemukan dalam dataframe")
        
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=6000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(df['fitur'])
        return tfidf_matrix
    except Exception as e:
        raise Exception(f"Error building similarity matrix: {str(e)}")

# Fungsi untuk menghitung metrik evaluasi
def calculate_evaluation_metrics(recommendations, original_anime_idx, df):
    """
    Menghitung Precision, Recall, dan F1-Score untuk rekomendasi
    berdasarkan genre overlap
    """
    try:
        if 'genre' not in df.columns:
            return None
        
        # Genre anime asli
        original_genres = set()
        if pd.notna(df.iloc[original_anime_idx]['genre']):
            original_genres = set(str(df.iloc[original_anime_idx]['genre']).split(','))
            original_genres = {g.strip() for g in original_genres if g.strip()}
        
        if not original_genres:
            return None
        
        # Hitung metrik untuk setiap rekomendasi
        precisions = []
        recalls = []
        f1_scores = []
        
        for _, rec in recommendations.iterrows():
            rec_genres = set()
            if pd.notna(rec['genre']):
                rec_genres = set(str(rec['genre']).split(','))
                rec_genres = {g.strip() for g in rec_genres if g.strip()}
            
            if not rec_genres:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                continue
            
            # Hitung intersection
            intersection = original_genres.intersection(rec_genres)
            
            # Precision = TP / (TP + FP)
            precision = len(intersection) / len(rec_genres)
            
            # Recall = TP / (TP + FN)
            recall = len(intersection) / len(original_genres)
            
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

# Fungsi rekomendasi
def get_recommendations(anime_title, df, tfidf_matrix, n_recommendations=5):
    try:
        # Cari index anime
        matches = df[df['judul'].str.lower() == anime_title.lower()]
        
        if len(matches) == 0:
            # Coba cari partial match
            matches = df[df['judul'].str.contains(anime_title, case=False, na=False)]
            if len(matches) == 0:
                return pd.DataFrame(), None, None, "Anime tidak ditemukan dalam database"
        
        idx = matches.index[0]
        
        # Hitung cosine similarity
        sim_scores = cosine_similarity(
            tfidf_matrix[idx:idx+1],
            tfidf_matrix
        ).flatten()
        
        # Ambil anime paling mirip (kecuali dirinya sendiri)
        # Urutkan dari tertinggi ke terendah dan ambil n+1 pertama
        similar_indices = sim_scores.argsort()[-(n_recommendations + 1):][::-1]
        
        # Hilangkan anime itu sendiri dari hasil
        similar_indices = [i for i in similar_indices if i != idx][:n_recommendations]
        
        if not similar_indices:
            return pd.DataFrame(), None, None, "Tidak ditemukan anime yang mirip"
        
        # Buat dataframe rekomendasi
        recommendations = df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = sim_scores[similar_indices]
        
        # Hitung metrik evaluasi
        evaluation_metrics = calculate_evaluation_metrics(recommendations, idx, df)
        
        return recommendations, evaluation_metrics, idx, None
        
    except Exception as e:
        return pd.DataFrame(), None, None, f"Error dalam proses rekomendasi: {str(e)}"