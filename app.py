import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# Load data dengan error handling
try:
    df = cached_load_data()
    if df.empty:
        st.error("Dataframe kosong. Periksa file data/anime_dataset_clean.csv")
        st.stop()
    
    tfidf_matrix = cached_build_matrices(df)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Pastikan file anime_recommender.py ada di direktori yang sama")
    st.stop()

# Header
st.title("🎌 Sistem Rekomendasi Anime Content-Based")
st.divider()

with st.sidebar: 
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
        if not filtered_df.empty:
            anime_input = st.selectbox(
                "Pilih anime yang Anda sukai:",
                options=filtered_df['judul'].tolist(),
                help="Pilih anime untuk mendapatkan rekomendasi serupa"
            )
        else:
            st.warning("Tidak ada anime yang sesuai dengan filter.")
            anime_input = None
    with col2:
        n_recommendations = st.number_input(
            "Jumlah Rekomendasi",
            min_value=1,
            max_value=20,
            value=5
        )
    
    if anime_input and st.button("🔮 Dapatkan Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Menganalisis dan mencari anime serupa..."):
            recommendations, error = get_recommendations(
                anime_input, 
                df, 
                tfidf_matrix,
                n_recommendations=n_recommendations
            )
            
            if error:
                st.error(f"❌ {error}")
            elif recommendations.empty:
                st.error("Tidak ada rekomendasi yang ditemukan.")
            else:
                # Tampilkan anime yang dipilih
                selected_anime = df[df['judul'] == anime_input].iloc[0]
                st.success(f"✅ Menampilkan rekomendasi berdasarkan: **{anime_input}**")
                
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
                        if 'sinopsis' in selected_anime and pd.notna(selected_anime['sinopsis']):
                            st.markdown(f"**Sinopsis:** {selected_anime['sinopsis']}")
                
                st.divider()
                
                # Analisis Kemiripan
                with st.expander("📊 Visualisasi Rekomendasi", expanded=False):
                    # Baris pertama: 2 grafik besar
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.markdown("#### Perbandingan Rating")
                        # Bar chart horizontal untuk rating
                        
                        fig_rating = go.Figure(go.Bar(
                            x=recommendations['rating'],
                            y=recommendations['judul'].str[:25],
                            orientation='h',
                            marker=dict(
                                color=recommendations['rating'],
                                colorscale='Viridis',
                                showscale=False
                            ),
                            text=recommendations['rating'].round(2),
                            textposition='auto',
                        ))
                        fig_rating.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis_title="Rating",
                            yaxis_title="Judul Film",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_rating, use_container_width=True)
                    
                    with col_chart2:
                        st.markdown("#### Jaringan Kesamaan Anime")
                        # Network graph sederhana
                        
                        # Posisi node dalam lingkaran
                        n = len(recommendations)
                        angles = [i * 2 * 3.14159 / n for i in range(n)]
                        x_nodes = [2 * np.cos(angle) for angle in angles]
                        y_nodes = [2 * np.sin(angle) for angle in angles]
                        
                        # Node center (anime pilihan)
                        x_center, y_center = [0], [0]
                        
                        # Edge traces
                        edge_x, edge_y = [], []
                        for i in range(n):
                            edge_x.extend([0, x_nodes[i], None])
                            edge_y.extend([0, y_nodes[i], None])
                        
                        fig_network = go.Figure()
                        
                        # Edges
                        fig_network.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(color='rgba(125,125,125,0.3)', width=1),
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Recommendation nodes
                        fig_network.add_trace(go.Scatter(
                            x=x_nodes, y=y_nodes,
                            mode='markers+text',
                            marker=dict(
                                size=recommendations['similarity_score'] * 100,
                                color=recommendations['similarity_score'] * 100,
                                colorscale='Blues',
                                showscale=False,
                                line=dict(width=2, color='white')
                            ),
                            text=recommendations['judul'].str[:15],
                            textposition='top center',
                            textfont=dict(size=8),
                            hovertemplate='<b>%{text}</b><br>Kemiripan: %{marker.color:.1f}%<extra></extra>',
                            showlegend=False
                        ))
                        
                        # Center node
                        fig_network.add_trace(go.Scatter(
                            x=x_center, y=y_center,
                            mode='markers+text',
                            marker=dict(size=30, color='red', line=dict(width=2, color='white')),
                            text=[anime_input[:15]],
                            textposition='bottom center',
                            textfont=dict(size=10, color='red'),
                            hovertemplate='<b>Anime Pilihan</b><br>%{text}<extra></extra>',
                            showlegend=False
                        ))
                        
                        fig_network.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_network, use_container_width=True)
                    
                    st.divider()
                    
                    # Baris kedua: 2 grafik tambahan
                    col_chart3, col_chart4 = st.columns(2)
                    
                    with col_chart3:
                        st.markdown("#### Scatter Plot Rekomendasi")
                        # Scatter plot rating vs similarity
                        fig_scatter = go.Figure()
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=recommendations['similarity_score'] * 100,
                            y=recommendations['rating'],
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                color=recommendations['similarity_score'] * 100,
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(title="Kemiripan %")
                            ),
                            text=recommendations['judul'].str[:10],
                            textposition='top center',
                            textfont=dict(size=8),
                            hovertemplate='<b>%{text}</b><br>Kemiripan: %{x:.1f}%<br>Rating: %{y:.2f}<extra></extra>'
                        ))
                        
                        fig_scatter.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis_title="Kemiripan (%)",
                            yaxis_title="Rating",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col_chart4:
                        st.markdown("#### Distribusi Genre")
                        # Genre distribution
                        genre_count = {}
                        for genres in recommendations['genre'].dropna():
                            for genre in str(genres).split(','):
                                genre = genre.strip()
                                genre_count[genre] = genre_count.get(genre, 0) + 1
                        
                        if genre_count:
                            genre_df = pd.DataFrame(list(genre_count.items()), columns=['Genre', 'Count'])
                            genre_df = genre_df.sort_values('Count', ascending=True).tail(8)
                            
                            fig_genre = go.Figure(go.Bar(
                                x=genre_df['Count'],
                                y=genre_df['Genre'],
                                orientation='h',
                                marker=dict(
                                    color=genre_df['Count'],
                                    colorscale='Teal',
                                    showscale=False
                                ),
                                text=genre_df['Count'],
                                textposition='auto',
                            ))
                            fig_genre.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_title="Jumlah",
                                yaxis_title="Genre",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_genre, use_container_width=True)
                    
                    st.divider()
                    
                    # Baris ketiga: Distribusi kata
                    st.markdown("#### Distribusi Kata Kunci")
                    
                    # Ekstrak kata dari fitur_stem rekomendasi
                    word_freq = {}
                    for idx in recommendations.index:
                        if 'fitur' in df.columns and pd.notna(df.loc[idx, 'fitur']):
                            words = str(df.loc[idx, 'fitur']).split()
                            for word in words:
                                if len(word) > 3:  # Filter kata minimal 4 karakter
                                    word_freq[word] = word_freq.get(word, 0) + 1
                    
                    if word_freq:
                        # Ambil top 20 kata
                        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                        words_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
                        
                        # Buat word cloud style bar chart
                        fig_words = go.Figure(go.Bar(
                            x=words_df['Frekuensi'],
                            y=words_df['Kata'],
                            orientation='h',
                            marker=dict(
                                color=words_df['Frekuensi'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Frekuensi", x=1.15)
                            ),
                            text=words_df['Frekuensi'],
                            textposition='auto',
                            hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>'
                        ))
                        
                        fig_words.update_layout(
                            height=500,
                            margin=dict(l=0, r=80, t=0, b=0),
                            xaxis_title="Frekuensi Kemunculan",
                            yaxis_title="Kata Kunci",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(autorange="reversed")
                        )
                        
                        st.plotly_chart(fig_words, use_container_width=True)
                    else:
                        st.warning("Data fitur kata tidak tersedia untuk analisis")
                    
                    # Metrik statistik
                    st.divider()
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        avg_similarity = recommendations['similarity_score'].mean() * 100
                        st.metric("Rata-rata Kemiripan", f"{avg_similarity:.1f}%")
                    with col_m2:
                        max_similarity = recommendations['similarity_score'].max() * 100
                        st.metric("Kemiripan Tertinggi", f"{max_similarity:.1f}%")
                    with col_m3:
                        avg_rating = recommendations['rating'].mean()
                        st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
                    with col_m4:
                        total_recs = len(recommendations)
                        st.metric("Total Rekomendasi", total_recs)
                
                st.divider()
                
                st.subheader("✨ Rekomendasi untuk Anda")
                
                # Tampilkan rekomendasi dalam grid
                for idx, (_, row) in enumerate(recommendations.iterrows()):
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
                            elif similarity_percent >= 60:
                                badge = "✨ Mirip"
                            elif similarity_percent >= 40:
                                badge = "👍 Cukup Mirip"
                            else:
                                badge = "📌 Agak Mirip"
                            
                            st.markdown(f"### {row['judul']}")
                            st.markdown(f"**{badge}** | Kemiripan: `{similarity_percent:.1f}%` | Rating: ⭐ `{row['rating']:.2f}`")
                            
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                if 'jenis_tayangan' in row and pd.notna(row['jenis_tayangan']):
                                    st.markdown(f"**Jenis:** {row['jenis_tayangan']}")
                            with col_meta2:
                                if 'musim_tayang' in row and pd.notna(row['musim_tayang']):
                                    st.markdown(f"**Musim:** {row['musim_tayang']}")
                            with col_meta3:
                                if 'studio' in row and pd.notna(row['studio']):
                                    st.markdown(f"**Studio:** {row['studio']}")
                            
                            if 'genre' in row and pd.notna(row['genre']):
                                st.markdown(f"**Genre:** {row['genre']}")
                            
                            if 'sinopsis' in row and pd.notna(row['sinopsis']):
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
    display_columns = ['judul', 'rating', 'genre', 'jenis_tayangan']
    
    # Tambahkan kolom jika ada
    optional_columns = ['musim_tayang', 'studio']
    for col in optional_columns:
        if col in df.columns:
            display_columns.append(col)
    
    display_df = filtered_df[display_columns].copy()
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
        if 'studio' in df.columns:
            st.metric("Total Studio", df['studio'].nunique())
        else:
            st.metric("Total Genre", df['genre'].nunique() if 'genre' in df.columns else "N/A")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Rating")
        rating_counts = pd.cut(df['rating'], bins=[0, 5, 7, 8, 9, 10], labels=['0-5', '5-7', '7-8', '8-9', '9-10']).value_counts().sort_index()
        st.bar_chart(rating_counts)
    
    with col2:
        if 'jenis_tayangan' in df.columns:
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
    <p>Dibuat dengan Python & Streamlit • TF-IDF + Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)