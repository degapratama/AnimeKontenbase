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
st.title("Sistem Rekomendasi Anime Konten Base Recomendation")
st.markdown("*Menemukan anime serupa berdasarkan analisis sinopsis, genre, dan metadata*")
st.divider()

# Sidebar untuk filter dan metode
with st.sidebar:
    st.header("🔍 Filter & Pengaturan")
    
    # Pilih metode rekomendasi
    st.subheader("⚙️ Metode Rekomendasi")
    method = st.radio(
        "Pilih metode analisis:",
        ["hybrid", "sinopsis", "genre"],
        format_func=lambda x: {
            "hybrid": "🔥 Hybrid (Recommended)",
            "sinopsis": "📖 Sinopsis Only",
            "genre": "🎭 Genre Only"
        }[x],
        help="Hybrid menggabungkan sinopsis, genre, studio, dan jenis tayangan dengan bobot optimal"
    )
    
    # Info metode
    with st.expander("📖 Penjelasan Metode", expanded=False):
        if method == "hybrid":
            st.markdown("""
            **🔥 Hybrid (Recommended)**
            - Sinopsis: 60%
            - Genre: 25%
            - Studio: 10%
            - Jenis: 5%
            """)
        elif method == "sinopsis":
            st.markdown("""
            **📖 Sinopsis Only**
            - 100% berdasarkan sinopsis
            """)
        elif method == "genre":
            st.markdown("""
            **🎭 Genre Only**
            - 100% berdasarkan genre
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

# Tab navigasi
tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi", "📋 Database Anime", "📈 Analisis"])

with tab1:
    st.header("🎬 Cari Rekomendasi Anime")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        anime_input = st.selectbox(
            "Pilih anime yang Anda sukai:",
            options=filtered_df['judul'].tolist(),
            help="Pilih anime untuk mendapatkan rekomendasi serupa"
        )
    with col2:
        n_recommendations = st.number_input(
            "Jumlah Rekomendasi",
            min_value=1,
            max_value=20,
            value=5
        )
    
    if st.button("🔮 Dapatkan Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Menganalisis dan mencari anime serupa..."):
            recommendations, error, weight_info = get_recommendations(
                anime_input, 
                df, 
                matrices,
                method=method,
                n_recommendations=n_recommendations
            )
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Tampilkan anime yang dipilih
                selected_anime = df[df['judul'] == anime_input].iloc[0]
                
                st.success(f"✅ Menampilkan rekomendasi berdasarkan: **{anime_input}**")
                st.info(f"📊 Metode: {weight_info}")
                
                st.divider()
                
                # Lihat detail anime pilihan
                with st.expander("📖 Detail Anime Pilihan", expanded=False):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        if pd.notna(selected_anime['poster_url']):
                            st.image(selected_anime['poster_url'], use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/225x350?text=No+Image", use_container_width=True)
                    with col_b:
                        st.markdown(f"### {selected_anime['judul']}")
                        st.markdown(f"**Rating:** ⭐ {selected_anime['rating']}")
                        st.markdown(f"**Genre:** {selected_anime['genre']}")
                        st.markdown(f"**Jenis:** {selected_anime['jenis_tayangan']}")
                        st.markdown(f"**Studio:** {selected_anime['studio']}")
                        st.markdown(f"**Sinopsis:** {selected_anime['sinopsis']}")
                
                st.divider()
                st.subheader("✨ Rekomendasi untuk Anda")
                
                # Tampilkan rekomendasi dalam grid
                for idx, (rec_idx, row) in enumerate(recommendations.iterrows()):
                    with st.container(border=True):
                        col_poster, col_info = st.columns([1, 3])
                        
                        with col_poster:
                            if pd.notna(row['poster_url']):
                                st.image(row['poster_url'], use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/225x350?text=No+Image", use_container_width=True)
                        
                        with col_info:
                            similarity_percent = row['similarity_score'] * 100
                            
                            # Badge kemiripan
                            if similarity_percent >= 80:
                                badge = "🔥 Sangat Mirip"
                                color = "#00D084"
                            elif similarity_percent >= 60:
                                badge = "✨ Mirip"
                                color = "#0099FF"
                            elif similarity_percent >= 40:
                                badge = "👍 Cukup Mirip"
                                color = "#FF9500"
                            else:
                                badge = "📌 Agak Mirip"
                                color = "#A0A0A0"
                            
                            st.markdown(f"### {row['judul']}")
                            st.markdown(f"**{badge}** | Kemiripan: `{similarity_percent:.1f}%` | Rating: ⭐ `{row['rating']}`")
                            
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                st.markdown(f"**Jenis:** {row['jenis_tayangan']}")
                            with col_meta2:
                                st.markdown(f"**Musim:** {row['musim_tayang']}")
                            with col_meta3:
                                st.markdown(f"**Studio:** {row['studio']}")
                            
                            st.markdown(f"**Genre:** {row['genre']}")
                            
                            with st.expander("📖 Baca Sinopsis Lengkap"):
                                st.write(row['sinopsis'])
                        
                        st.divider()

with tab2:
    st.header("📚 Database Anime")
    st.markdown(f"**Total:** {len(filtered_df)} anime ditampilkan")
    
    # Search box
    search_query = st.text_input("🔍 Cari anime berdasarkan judul", "")
    
    if search_query:
        filtered_df = filtered_df[filtered_df['judul'].str.contains(search_query, case=False, na=False)]
        st.markdown(f"**Hasil pencarian:** {len(filtered_df)} anime")
    
    # Tampilkan dataframe
    display_df = filtered_df[['judul', 'rating', 'genre', 'jenis_tayangan', 'musim_tayang', 'studio']].copy()
    display_df = display_df.sort_values('rating', ascending=False)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "judul": "Judul",
            "rating": st.column_config.NumberColumn("Rating", format="⭐ %.2f"),
            "genre": "Genre",
            "jenis_tayangan": "Jenis",
            "musim_tayang": "Musim Tayang",
            "studio": "Studio"
        }
    )

with tab3:
    st.header("📊 Analisis Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anime", len(df))
    with col2:
        st.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")
    with col3:
        st.metric("Total Studio", df['studio'].nunique())
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Rating")
        rating_counts = pd.cut(df['rating'], bins=[0, 5, 7, 8, 9, 10], labels=['0-5', '5-7', '7-8', '8-9', '9-10']).value_counts().sort_index()
        st.bar_chart(rating_counts)
    
    with col2:
        st.subheader("🎬 Jenis Tayangan")
        jenis_counts = df['jenis_tayangan'].value_counts()
        st.bar_chart(jenis_counts)
    
    st.divider()
    
    st.subheader("🏆 Top 10 Anime Berdasarkan Rating")
    top_anime = df.nlargest(10, 'rating')[['judul', 'rating', 'genre']]
    st.dataframe(
        top_anime,
        use_container_width=True,
        hide_index=True,
        column_config={
            "judul": "Judul",
            "rating": st.column_config.NumberColumn("Rating", format="⭐ %.2f"),
            "genre": "Genre"
        }
    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Sistem Rekomendasi Anime menggunakan Hybrid Content-Based Filtering</p>
    <p><small>Dibuat dengan Python & Streamlit • TF-IDF + Cosine Similarity</small></p>
</div>
""", unsafe_allow_html=True)