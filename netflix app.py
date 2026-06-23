import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Netflix Movie Finder",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Change main background to red */
    .stApp {
        background-color: #E50914;
    }

    /* Optional: also target the main block container */
    .block-container {
        background-color: #E50914;
    }

    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8);
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .cluster-tag {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.2rem;
    }
    .stButton button {
        background-color: white;
        color: #E50914;
        border-radius: 30px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #f0f0f0;
        transform: scale(1.02);
        color: #b20710;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">🎬 Find Your Next Netflix Show</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover hidden gems based on your taste preferences</div>', unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_combined_clustering_dataset.csv")
    return df

@st.cache_data
def process_data(df):
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    le = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    return df_clean, scaled_data, scaler

@st.cache_data
def train_clusters(scaled_data):
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans

# Load data
with st.spinner("Loading Netflix data..."):
    df_original = load_data()
    df_processed, scaled_data, scaler = process_data(df_original)
    clusters, kmeans = train_clusters(scaled_data)

# Sidebar - User Input
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", use_container_width=True)
    st.markdown("---")
    st.markdown("### 🎯 Tell us what you like")

    rating_range = st.slider(
        "⭐ Rating Range",
        min_value=0.0,
        max_value=10.0,
        value=(6.0, 8.0),
        step=0.5
    )

    popularity = st.slider(
        "🔥 Popularity (1-100)",
        min_value=1,
        max_value=100,
        value=30
    )

    release_year = st.slider(
        "📅 Release Year",
        min_value=2010,
        max_value=2025,
        value=(2018, 2023)
    )

    content_type = st.selectbox(
        "🎞️ Content Type",
        ["Any", "Movie", "TV Show"]
    )

    language = st.selectbox(
        "🌍 Language",
        ["Any", "English (en)", "Spanish (es)", "French (fr)", "Hindi (hi)", "Korean (ko)"]
    )

    st.markdown("---")
    find_btn = st.button("🔍 Find My Next Watch", use_container_width=True)

# Main content area
if find_btn:
    with st.spinner("Finding the perfect content for you..."):
        df_filtered = df_original.copy()
        df_filtered['Cluster'] = clusters

        if rating_range:
            df_filtered = df_filtered[(df_filtered['rating'] >= rating_range[0]) &
                                      (df_filtered['rating'] <= rating_range[1])]

        if popularity:
            df_filtered = df_filtered[df_filtered['popularity'] >= popularity]

        if release_year:
            df_filtered = df_filtered[(df_filtered['release_year'] >= release_year[0]) &
                                      (df_filtered['release_year'] <= release_year[1])]

        if content_type != "Any":
            df_filtered = df_filtered[df_filtered['type'] == content_type]

        if language != "Any":
            df_filtered = df_filtered[df_filtered['language'] == language.split('(')[-1].strip(')')]

        if len(df_filtered) == 0:
            st.warning("No exact matches found. Showing popular high-rated content instead.")
            df_filtered = df_original.copy()
            df_filtered['Cluster'] = clusters
            df_filtered = df_filtered[df_filtered['rating'] >= 7.0]
            df_filtered = df_filtered[df_filtered['popularity'] >= 20]

        top_recommendations = df_filtered.sort_values(['rating', 'popularity'], ascending=[False, False]).head(20)

        st.markdown("---")
        st.markdown("### 🎉 Your Personalized Recommendations")
        st.markdown(f"Found **{len(top_recommendations)}** great options for you!")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(top_recommendations.iterrows()):
            with cols[idx % 3]:
                with st.container():
                    rating_color = "green" if row['rating'] >= 7.5 else "orange" if row['rating'] >= 6.0 else "red"
                    st.markdown(f"""
                    <div style="background: #E50914; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.3); border-left: 4px solid {rating_color};">
                        <h4 style="margin: 0; font-size: 1rem; color: white;">{row['title']}</h4>
                        <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap;">
                            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem;">
                                ⭐ {row['rating']:.1f}
                            </span>
                            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem;">
                                📅 {int(row['release_year'])}
                            </span>
                            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem;">
                                {row['type']}
                            </span>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: rgba(255,255,255,0.85);">
                            {row['genres'][:80] + '...' if len(str(row['genres'])) > 80 else row['genres']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Cluster visualization
        st.markdown("---")
        st.markdown("### 📊 Your Content Profile")

        cluster_dist = df_filtered['Cluster'].value_counts().sort_index()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=cluster_dist.values,
                names=[f"Cluster {i}" for i in cluster_dist.index],
                title="Content Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            all_genres = []
            for genres in df_filtered['genres'].head(100):
                if isinstance(genres, str):
                    all_genres.extend([g.strip() for g in genres.split(',')])

            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts().head(10)

                fig = px.bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    title="Top Genres in Your Matches",
                    color=genre_counts.values,
                    color_continuous_scale="Reds",
                    labels={'x': 'Count', 'y': 'Genre'}
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        st.markdown("---")
        col3, col4, col5 = st.columns(3)

        with col3:
            avg_rating = df_filtered['rating'].mean()
            st.metric("⭐ Average Rating", f"{avg_rating:.2f}")

        with col4:
            avg_year = df_filtered['release_year'].mean()
            st.metric("📅 Average Release Year", f"{avg_year:.0f}")

        with col5:
            total_matches = len(df_filtered)
            st.metric("🎯 Total Matches", total_matches)

# Show some initial stats when no prediction
else:
    st.markdown("### 🌟 Discover Your Perfect Content")
    st.markdown("Use the sidebar to tell us your preferences and click **Find My Next Watch**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; padding: 1.5rem; color: white; height: 100%;">
            <h3>🎯 4 Clusters</h3>
            <p style="font-size: 0.9rem;">Content organized into 4 distinct groups based on characteristics</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    border-radius: 15px; padding: 1.5rem; color: white; height: 100%;">
            <h3>⭐ 32,000+ Titles</h3>
            <p style="font-size: 0.9rem;">Comprehensive Netflix catalog ready to explore</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    border-radius: 15px; padding: 1.5rem; color: white; height: 100%;">
            <h3>🔍 Smart Search</h3>
            <p style="font-size: 0.9rem;">Find exactly what matches your taste preferences</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔥 Popular Picks")

    popular_df = df_original.sort_values('popularity', ascending=False).head(12)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(popular_df.iterrows()):
        if idx < 8:
            with cols[idx % 4]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.15); border-radius: 10px; padding: 0.8rem; margin-bottom: 0.5rem;
                            border-left: 3px solid white;">
                    <div style="font-weight: 600; font-size: 0.9rem; color: white;">{row['title']}</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.75);">⭐ {row['rating']:.1f} · {row['type']}</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem; padding: 1rem;">
    Made with ❤️ | Netflix Content Discovery Tool
</div>
""", unsafe_allow_html=True)