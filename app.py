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

# Load data
df = cached_load_data()
tfidf_matrix = build_similarity_matrices(df)

# Header
st.title("🎌 Sistem Rekomendasi Anime Content-Based")
st.divider()

with st.sidebar: 
    st.divider()
    st.markdown("### 📊 Statistik Dataset")
    st.metric("Total Anime", len(df))
    st.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")
    st.divider()

# Gunakan data asli tanpa filter
filtered_df = df.copy()

# Tab navigasi
tab1 = st.tabs(["🎯 Rekomendasi"])[0]

with tab1:
    st.header("🎬 Cari Rekomendasi Anime")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        anime_input = st.selectbox(
            "Pilih anime yang Anda sukai:",
            options=filtered_df['judul'].tolist()
        )
    with col2:
        n_recommendations = st.number_input(
            "Jumlah Rekomendasi",
            min_value=1,
            max_value=20,
            value=5
        )
    
    if anime_input and st.button("🔮 Dapatkan Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Menganalisis dan mencari anime serupa..."):
            recommendations, evaluation_metrics, original_idx, error = get_recommendations(
                anime_input, 
                df, 
                tfidf_matrix,
                n_recommendations=n_recommendations
            )
            
            if not error and not recommendations.empty:
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
                    
                    # Baris kedua: 2 grafik tambahan
                    col_chart3, col_chart4 = st.columns(2)
                    
                    with col_chart3:
                        # Genre distribution
                        st.markdown("#### Distribusi Genre")
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
                    
                    with col_chart4:
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
                            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                            words_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
                            
                            # Buat word cloud style bar chart
                            fig_words = go.Figure(go.Bar(
                                x=words_df['Frekuensi'],
                                y=words_df['Kata'],
                                orientation='h',
                                marker=dict(
                                    color=words_df['Frekuensi'],
                                    colorscale='Viridis',
                                    showscale=False
                                ),
                                text=words_df['Frekuensi'],
                                textposition='auto',
                                hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>'
                            ))
                            
                            fig_words.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_title="Frekuensi Kemunculan",
                                yaxis_title="Kata Kunci",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(autorange="reversed")
                            )
                            
                            st.plotly_chart(fig_words, use_container_width=True)
                    
                    st.divider()
                    st.markdown("#### 📈 Evaluasi Metrik Rekomendasi")
                    
                    if evaluation_metrics:
                        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                        
                        with col_metrics1:
                            precision = evaluation_metrics['precision']
                            st.metric("Precision (Rata-rata)", f"{precision:.3f}")
                        
                        with col_metrics2:
                            recall = evaluation_metrics['recall']
                            st.metric("Recall (Rata-rata)", f"{recall:.3f}")
                        
                        with col_metrics3:
                            f1_score_val = evaluation_metrics['f1_score']
                            st.metric("F1-Score (Rata-rata)", f"{f1_score_val:.3f}")
                        
                        # Tampilkan chart untuk metrik per anime
                        st.markdown("##### Metrik per Anime Rekomendasi")
                        
                        # Buat dataframe untuk chart
                        metrics_df = pd.DataFrame({
                            'Anime': recommendations['judul'].str[:20],
                            'Precision': evaluation_metrics['individual_precisions'],
                            'Recall': evaluation_metrics['individual_recalls'],
                            'F1-Score': evaluation_metrics['individual_f1_scores']
                        })
                        
                        # Chart bar untuk metrik per anime
                        fig_metrics = go.Figure()
                        
                        fig_metrics.add_trace(go.Bar(
                            x=metrics_df['Anime'],
                            y=metrics_df['Precision'],
                            name='Precision',
                            marker_color='#1f77b4',
                            text=metrics_df['Precision'].round(3),
                            textposition='auto'
                        ))
                        
                        fig_metrics.add_trace(go.Bar(
                            x=metrics_df['Anime'],
                            y=metrics_df['Recall'],
                            name='Recall',
                            marker_color='#ff7f0e',
                            text=metrics_df['Recall'].round(3),
                            textposition='auto'
                        ))
                        
                        fig_metrics.add_trace(go.Bar(
                            x=metrics_df['Anime'],
                            y=metrics_df['F1-Score'],
                            name='F1-Score',
                            marker_color='#2ca02c',
                            text=metrics_df['F1-Score'].round(3),
                            textposition='auto'
                        ))
                        
                        fig_metrics.update_layout(
                            height=400,
                            title="Metrik Evaluasi per Anime Rekomendasi",
                            xaxis_title="Anime",
                            yaxis_title="Nilai",
                            barmode='group',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Metrik statistik lainnya
                    st.divider()
                    st.markdown("#### 📊 Ringkasan Statistik")
                    
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
                            
                            # Tampilkan metrik evaluasi jika tersedia
                            if evaluation_metrics and idx < len(evaluation_metrics['individual_precisions']):
                                precision = evaluation_metrics['individual_precisions'][idx]
                                recall = evaluation_metrics['individual_recalls'][idx]
                                f1 = evaluation_metrics['individual_f1_scores'][idx]
                            
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

