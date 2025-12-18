import streamlit as st
import pandas as pd
from anime_recommender import load_data, build_similarity_matrices, get_recommendations

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Rekomendasi Anime",
    page_icon="🎌",
    layout="wide"
)

# Cache loading data
@st.cache_data
def cached_load_data():
    return load_data()

# Cache model TF-IDF
@st.cache_resource
def cached_build_matrices(df):
    return build_similarity_matrices(df)

# Load data
df = cached_load_data()
matrices = cached_build_matrices(df)

# Header
st.title("🎌 Sistem Rekomendasi Anime Berbasis Konten")
st.markdown("*Menemukan anime serupa menggunakan Combined Features Analysis*")
st.divider()

with st.sidebar:
    st.header("🔍 Filter & Pengaturan")
    
    # Info model rekomendasi
    st.subheader("⚙️ Model Rekomendasi")
    with st.expander("📖 Cara Kerja Combined Features", expanded=True):
        st.markdown("""
        ### 🔧 Fitur yang Digunakan:
        
        **1. Sinopsis (60%)**
           - Menganalisis cerita/plot anime
        
        **2. Genre (25%)**
           - Kategori/tipe anime
        
        **3. Studio (10%)**
           - Studio produksi anime
        
        **4. Jenis Tayangan (5%)**
           - TV, Movie, OVA, Special, dll
        
        ### 📊 Cara Perhitungan:
        
        **Step 1: Gabung Semua Fitur dalam 1 Tabel (tanpa pembobotan)**
        ```
        Combined Text = [Sinopsis] + [Genre] + [Studio] + [Jenis Tayangan]
        ```
        (Semua fitur hanya digabungkan—tidak ada pengulangan untuk memberi bobot)
        
        **Step 2: TF-IDF pada Combined Text**
        - Konversi gabungan teks menjadi vektor numerik
        - Term Frequency-Inverse Document Frequency 
        - Hanya 1 kali perhitungan untuk semua fitur
        
        **Step 3: Cosine Similarity**
        ```
        Similarity = cos(angle) antara dua vektor
        Hasil: 0 (berbeda) sampai 1 (sama persis)
        ```
        
        ✅ **Keuntungan:**
        - SATU perhitungan untuk semua fitur (tidak terpisah)
        - Fitur yang digabung jadi vektor tunggal
        - Cosine similarity dihitung dari vektor gabungan
        - Lebih efisien & konsisten
        """)

    
    st.divider()
    
    # Filter berdasarkan genre
    all_genres = set()
    for genres in df['genre'].dropna():
        all_genres.update([g.strip() for g in str(genres).split(',')])
    selected_genre = st.selectbox("Filter Genre", ["Semua Genre"] + sorted(list(all_genres)))
    
    # Filter berdasarkan jenis tayangan
    jenis_options = ["Semua Jenis"] + sorted(df['jenis_tayangan'].dropna().unique().tolist())
    selected_jenis = st.selectbox("Filter Jenis Tayangan", jenis_options)
    
    # Filter berdasarkan rating
    min_rating = st.slider("Rating Minimum", 0.0, 10.0, 0.0, 0.5)
    
    st.divider()
    st.markdown("### 📊 Statistik Dataset")
    st.metric("Total Anime", len(df))
    st.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")

# Aplikasikan filter
filtered_df = df.copy()
if selected_genre != "Semua Genre":
    filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, na=False, case=False)]
if selected_jenis != "Semua Jenis":
    filtered_df = filtered_df[filtered_df['jenis_tayangan'] == selected_jenis]
filtered_df = filtered_df[filtered_df['rating'] >= min_rating]

# Sederhana: satu halaman rekomendasi
st.header("🎯 Rekomendasi Sederhana")

col1, col2 = st.columns([3, 1])
with col1:
    anime_input = st.selectbox("Pilih anime yang Anda sukai:", options=filtered_df['judul'].tolist())
with col2:
    n_recommendations = st.number_input("Jumlah Rekomendasi", min_value=1, max_value=20, value=5)

if st.button("Dapatkan Rekomendasi"):
    with st.spinner("Mencari rekomendasi..."):
        recommendations, error, weight_info = get_recommendations(anime_input, df, matrices, method='combined', n_recommendations=n_recommendations)
        if error:
            st.error(error)
        else:
            st.success(f"Rekomendasi untuk: {anime_input}")
            st.info(weight_info)

            for _, row in recommendations.iterrows():
                st.markdown(f"**{row['judul']}** — ⭐ {row.get('rating', 'N/A')} — Kemiripan: {row['similarity_score']*100:.1f}%")
                st.markdown(f"Genre: {row.get('genre','-')} | Jenis: {row.get('jenis_tayangan','-')} | Studio: {row.get('studio','-')}")
                if pd.notna(row.get('poster_url')):
                    st.image(row['poster_url'], width=120)
                with st.expander("Sinopsis"):
                    st.write(row.get('sinopsis','-'))
                st.write("---")

# Simple database view (optional)
if st.checkbox("Tampilkan daftar anime", value=False):
    st.dataframe(filtered_df[['judul','rating','genre']].sort_values('rating', ascending=False))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Sistem Rekomendasi Anime menggunakan Hybrid Content-Based Filtering</p>
    <p><small>Dibuat dengan Python & Streamlit • TF-IDF + Cosine Similarity</small></p>
</div>
""", unsafe_allow_html=True)