import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --red:#E50914; --bg:#0A0A0A; --surface:#141414; --card:#1C1C1C;
    --border:#2C2C2C; --muted:#6B6B6B; --text:#F0F0F0; --gold:#F5C518;
}
html,body,.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem 4rem;max-width:1100px;margin:auto;}
.hero{text-align:center;padding:3.5rem 1rem 2rem;}
.hero-wordmark{font-size:.72rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--red);margin-bottom:.6rem;}
.hero-title{font-size:clamp(2.2rem,5vw,3.6rem);font-weight:800;line-height:1.1;color:var(--text);margin:0 0 .9rem;}
.hero-title span{color:var(--red);}
.hero-sub{font-size:1rem;color:var(--muted);max-width:520px;margin:0 auto;line-height:1.6;}
.section-label{font-size:.68rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--red);margin:2.5rem 0 1rem;display:flex;align-items:center;gap:.7rem;}
.section-label::after{content:'';flex:1;height:1px;background:var(--border);}
.stButton>button{width:100%;background:var(--red)!important;color:#fff!important;font-size:1rem!important;font-weight:700!important;letter-spacing:.04em!important;padding:.85rem 2rem!important;border:none!important;border-radius:8px!important;transition:background .2s ease!important;margin-top:1rem;}
.stButton>button:hover{background:#c4070f!important;}
.result-header{font-size:.65rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin:2.5rem 0 1.2rem;}
.rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:2rem;}
.rec-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem 1.1rem 1.1rem;position:relative;transition:border-color .2s ease,transform .15s ease;}
.rec-card:hover{border-color:var(--red);transform:translateY(-2px);}
.rec-rank{font-size:.62rem;font-weight:800;letter-spacing:.14em;color:var(--red);text-transform:uppercase;margin-bottom:.35rem;}
.rec-title{font-size:.95rem;font-weight:700;color:var(--text);line-height:1.25;margin-bottom:.4rem;}
.rec-meta{font-size:.72rem;color:var(--muted);line-height:1.5;}
.rec-score{position:absolute;top:10px;right:12px;background:var(--gold);color:#000;font-size:.65rem;font-weight:800;padding:2px 7px;border-radius:999px;}
.rec-type-badge{display:inline-block;padding:1px 8px;border-radius:4px;font-size:.65rem;font-weight:700;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.06em;}
.badge-movie{background:#1a1a4a;color:#7B9EFF;}
.badge-tv{background:#1a3a1a;color:#4CD964;}
.cluster-box{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--red);border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:1.5rem 0;font-size:.88rem;line-height:1.6;color:var(--text);}
.cluster-box strong{color:var(--red);}
div[data-testid="stSlider"]>div>div>div>div{background:var(--red)!important;}
div[data-testid="stSlider"] label{color:var(--text)!important;font-size:.82rem!important;font-weight:500!important;}
div[data-baseweb="select"]>div{background:var(--card)!important;border-color:var(--border)!important;color:var(--text)!important;}
div[data-baseweb="tag"]{background:var(--red)!important;border-radius:999px!important;}
div[data-testid="stRadio"] label{color:var(--text)!important;font-size:.88rem!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT CATALOG
# ─────────────────────────────────────────────────────────────────────────────
CATALOG = [
    ("Inception","Movie",["Action","Sci-Fi","Thriller"],2010,8.4,156,37119,160,839,"en",["mind-bending","intense","cerebral"]),
    ("The Dark Knight","Movie",["Action","Crime","Drama"],2008,9.0,169,32000,185,1005,"en",["intense","epic","dark"]),
    ("Interstellar","Movie",["Sci-Fi","Drama","Adventure"],2014,8.6,140,30000,165,701,"en",["cerebral","emotional","epic"]),
    ("Parasite","Movie",["Thriller","Drama","Comedy"],2019,8.5,112,17000,11,263,"ko",["intense","dark","social"]),
    ("The Shawshank Redemption","Movie",["Drama"],2000,9.3,90,28000,25,58,"en",["emotional","uplifting","classic"]),
    ("Pulp Fiction","Movie",["Crime","Drama","Thriller"],1994,8.9,95,27000,8,214,"en",["dark","quirky","classic"]),
    ("The Matrix","Movie",["Action","Sci-Fi"],1999,8.7,120,24000,63,463,"en",["mind-bending","action-packed","cerebral"]),
    ("Avengers: Endgame","Movie",["Action","Adventure","Sci-Fi"],2019,8.4,250,24000,356,2798,"en",["epic","emotional","action-packed"]),
    ("Spirited Away","Movie",["Animation","Family","Fantasy"],2001,8.6,80,14000,19,395,"ja",["magical","emotional","whimsical"]),
    ("The Lion King","Movie",["Animation","Family","Drama"],1994,8.5,95,20000,45,968,"en",["emotional","uplifting","classic"]),
    ("Joker","Movie",["Crime","Drama","Thriller"],2019,8.4,135,23000,55,1079,"en",["dark","intense","character-study"]),
    ("La La Land","Movie",["Romance","Drama","Musical"],2016,8.0,110,19000,30,447,"en",["emotional","romantic","uplifting"]),
    ("Get Out","Movie",["Horror","Thriller","Mystery"],2017,7.7,88,15000,4,255,"en",["tense","mind-bending","social"]),
    ("Everything Everywhere All at Once","Movie",["Action","Comedy","Drama"],2022,7.8,95,13000,14,74,"en",["whimsical","emotional","mind-bending"]),
    ("Mad Max: Fury Road","Movie",["Action","Adventure","Sci-Fi"],2015,8.1,105,22000,150,375,"en",["intense","action-packed","epic"]),
    ("The Grand Budapest Hotel","Movie",["Comedy","Drama","Mystery"],2014,8.1,85,18000,25,175,"en",["quirky","whimsical","stylish"]),
    ("Her","Movie",["Drama","Romance","Sci-Fi"],2013,8.0,75,16000,23,48,"en",["cerebral","emotional","romantic"]),
    ("Arrival","Movie",["Drama","Mystery","Sci-Fi"],2016,7.9,88,17000,47,203,"en",["cerebral","emotional","mind-bending"]),
    ("Knives Out","Movie",["Comedy","Crime","Mystery"],2019,7.9,97,14000,40,311,"en",["fun","quirky","mystery"]),
    ("The Irishman","Movie",["Crime","Drama"],2019,7.8,82,13000,159,8,"en",["classic","dark","character-study"]),
    ("Dune","Movie",["Adventure","Drama","Sci-Fi"],2021,8.0,143,16000,165,401,"en",["epic","cerebral","visual"]),
    ("Spider-Man: No Way Home","Movie",["Action","Adventure","Fantasy"],2021,8.3,230,19000,200,1900,"en",["fun","action-packed","emotional"]),
    ("Top Gun: Maverick","Movie",["Action","Drama"],2022,8.3,198,16000,170,1490,"en",["action-packed","emotional","uplifting"]),
    ("Oppenheimer","Movie",["Biography","Drama","History"],2023,8.5,165,18000,100,952,"en",["cerebral","epic","intense"]),
    ("Barbie","Movie",["Adventure","Comedy","Fantasy"],2023,6.9,189,14000,145,1442,"en",["fun","whimsical","social"]),
    ("Squid Game","TV Show",["Thriller","Drama","Action"],2021,8.0,210,15000,21,900,"ko",["intense","dark","social"]),
    ("Breaking Bad","TV Show",["Crime","Drama","Thriller"],2008,9.5,175,32000,3,58,"en",["intense","dark","cerebral"]),
    ("Stranger Things","TV Show",["Drama","Fantasy","Horror"],2016,8.7,195,24000,8,0,"en",["nostalgic","fun","tense"]),
    ("The Crown","TV Show",["Biography","Drama","History"],2016,8.7,120,18000,130,0,"en",["classic","character-study","drama"]),
    ("Dark","TV Show",["Drama","Mystery","Sci-Fi"],2017,8.8,98,12000,5,0,"de",["cerebral","mind-bending","dark"]),
    ("Money Heist","TV Show",["Action","Crime","Mystery"],2017,8.2,185,18000,4,0,"es",["intense","fun","action-packed"]),
    ("The Witcher","TV Show",["Action","Adventure","Fantasy"],2019,8.2,155,16000,10,0,"en",["epic","action-packed","fantasy"]),
    ("Ozark","TV Show",["Crime","Drama","Thriller"],2017,8.4,118,14000,5,0,"en",["dark","intense","character-study"]),
    ("The Office (US)","TV Show",["Comedy"],2005,9.0,145,28000,2,0,"en",["fun","quirky","uplifting"]),
    ("Friends","TV Show",["Comedy","Romance"],1994,8.9,150,26000,1,0,"en",["fun","romantic","classic"]),
    ("Succession","TV Show",["Drama"],2018,8.9,130,17000,5,0,"en",["dark","intense","character-study"]),
    ("Euphoria","TV Show",["Drama"],2019,8.4,155,16000,7,0,"en",["dark","emotional","intense"]),
    ("The Mandalorian","TV Show",["Action","Adventure","Sci-Fi"],2019,8.8,170,18000,25,0,"en",["fun","action-packed","epic"]),
    ("Narcos","TV Show",["Biography","Crime","Drama"],2015,8.8,138,16000,3,0,"en",["intense","dark","classic"]),
    ("Black Mirror","TV Show",["Drama","Sci-Fi","Thriller"],2011,8.8,125,18000,4,0,"en",["cerebral","dark","mind-bending"]),
    ("Mindhunter","TV Show",["Crime","Drama","Thriller"],2017,8.6,110,14000,8,0,"en",["dark","cerebral","intense"]),
    ("The Queen's Gambit","TV Show",["Drama"],2020,8.6,142,15000,30,0,"en",["character-study","emotional","uplifting"]),
    ("Lupin","TV Show",["Action","Crime","Mystery"],2021,7.5,140,11000,5,0,"fr",["fun","action-packed","quirky"]),
    ("Wednesday","TV Show",["Comedy","Fantasy","Horror"],2022,8.1,200,14000,25,0,"en",["quirky","dark","fun"]),
    ("Peaky Blinders","TV Show",["Crime","Drama"],2013,8.8,145,19000,3,0,"en",["intense","dark","classic"]),
    ("Soul","Movie",["Animation","Comedy","Drama"],2020,8.1,85,14000,150,121,"en",["emotional","uplifting","whimsical"]),
    ("Bird Box","Movie",["Drama","Horror","Sci-Fi"],2018,6.6,120,12000,19,100,"en",["tense","dark","thriller"]),
    ("Extraction","Movie",["Action","Thriller"],2020,6.7,110,11000,65,0,"en",["action-packed","intense","fun"]),
    ("Bridgerton","TV Show",["Drama","Romance"],2020,7.3,145,12000,7,0,"en",["romantic","fun","drama"]),
    ("You","TV Show",["Crime","Drama","Romance","Thriller"],2018,7.7,138,13000,5,0,"en",["dark","intense","thriller"]),
    ("Cobra Kai","TV Show",["Action","Drama","Sport"],2018,8.6,128,15000,5,0,"en",["nostalgic","fun","emotional"]),
    ("1899","TV Show",["Drama","Horror","Mystery"],2022,7.5,95,10000,6,0,"en",["mind-bending","dark","mystery"]),
    ("Glass Onion","Movie",["Comedy","Crime","Mystery"],2022,7.1,110,11000,40,0,"en",["fun","quirky","mystery"]),
    ("All Quiet on the Western Front","Movie",["Action","Drama","War"],2022,7.8,88,10000,20,0,"de",["intense","dark","emotional"]),
    ("Guillermo del Toro's Pinocchio","Movie",["Animation","Drama","Fantasy"],2022,7.6,70,9000,35,0,"en",["emotional","whimsical","dark"]),
    ("Outer Banks","TV Show",["Action","Adventure","Mystery"],2020,7.6,130,11000,6,0,"en",["fun","action-packed","mystery"]),
    ("Emily in Paris","TV Show",["Comedy","Drama","Romance"],2020,7.2,130,10000,4,0,"en",["fun","romantic","uplifting"]),
    ("Klaus","Movie",["Animation","Comedy","Drama"],2019,8.2,72,10000,40,10,"en",["emotional","uplifting","whimsical"]),
    ("Enola Holmes","Movie",["Adventure","Comedy","Mystery"],2020,6.6,100,10000,20,0,"en",["fun","mystery","uplifting"]),
    ("Avatar: The Way of Water","Movie",["Action","Adventure","Fantasy"],2022,7.6,195,11000,350,2320,"en",["epic","visual","action-packed"]),
    ("Dune: Part Two","Movie",["Adventure","Drama","Sci-Fi"],2024,8.5,190,14000,190,714,"en",["epic","cerebral","visual"]),
    ("The Bear","TV Show",["Comedy","Drama"],2022,8.6,115,12000,3,0,"en",["intense","character-study","emotional"]),
    ("Severance","TV Show",["Drama","Mystery","Sci-Fi","Thriller"],2022,8.7,120,11000,15,0,"en",["cerebral","mind-bending","dark"]),
    ("Beef","TV Show",["Comedy","Crime","Drama","Thriller"],2023,8.2,105,10000,5,0,"en",["dark","character-study","emotional"]),
]

GENRES_ALL = sorted(set(g for _,_,genres,*_ in CATALOG for g in genres))
MOODS_ALL  = sorted(set(m for *_,moods in CATALOG for m in moods))
LANGS = {"Any":None,"English":"en","Korean":"ko","Spanish":"es",
         "French":"fr","German":"de","Japanese":"ja"}

# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_model():
    rows = []
    for title,ctype,genres,year,rating,pop,votes,budget,rev,lang,mood in CATALOG:
        rows.append({"title":title,"type":ctype,"genres":genres,"year":year,
                     "rating":rating,"popularity":pop,"vote_count":votes,
                     "budget":budget,"revenue":rev,"language":lang,"moods":mood})
    df = pd.DataFrame(rows)
    X = df[["year","rating","popularity","vote_count","budget","revenue"]].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=5,random_state=42,n_init=10)
    df["cluster"] = km.fit_predict(Xs)
    return df, scaler, km

catalog_df, scaler, km_model = build_model()

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def recommend(genres_sel, moods_sel, content_type, lang_code,
              min_rating, min_year, popularity_pref, n=12):
    df = catalog_df.copy()
    pop_map = {"Low-key & obscure":40,"Mixed":100,"Mainstream hits":180}
    user_pop = pop_map.get(popularity_pref,100)
    user_vec = np.array([[min_year, min_rating, user_pop, 10000, 50, 200]],dtype=float)
    user_cluster = int(km_model.predict(scaler.transform(user_vec))[0])

    scores = []
    for _,row in df.iterrows():
        score = 0.0
        if genres_sel:
            score += len(set(genres_sel)&set(row["genres"]))*3.5
        if moods_sel:
            score += len(set(moods_sel)&set(row["moods"]))*2.5
        if content_type != "Both" and row["type"] != content_type:
            score -= 5
        if lang_code and row["language"] != lang_code:
            score -= 4
        if row["rating"] < min_rating:
            score -= 10
        if row["year"] < min_year:
            score -= 3
        if popularity_pref=="Low-key & obscure" and row["popularity"]<90:
            score += 2
        elif popularity_pref=="Mainstream hits" and row["popularity"]>130:
            score += 2
        if row["cluster"]==user_cluster:
            score += 2
        score += (row["rating"]-5)*0.8
        scores.append(score)

    df["score"] = scores
    return df.sort_values("score",ascending=False).head(n), user_cluster

CLUSTER_NAMES = {
    0:"Trending & Highly Engaged",1:"Classic & Enduring",
    2:"Premium Blockbusters",3:"Hidden Gems",4:"Niche & Cult Favourites",
}
CLUSTER_DESC = {
    0:"Your taste aligns with content generating real buzz — recent releases with strong audience engagement.",
    1:"You lean toward timeless classics with consistent critical and audience acclaim.",
    2:"You're drawn to high-production blockbuster-tier content with mass appeal.",
    3:"Your preferences point toward underseen titles with dedicated followings — quality over hype.",
    4:"You gravitate toward specialised, genre-defining, or cult content that rewards patient viewers.",
}

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-wordmark'>Netflix · AI Recommender</div>
    <h1 class='hero-title'>Find Your Next<br><span>Perfect Watch</span></h1>
    <p class='hero-sub'>Tell us what you're in the mood for — our ML model segments content
    and surfaces the titles most likely to resonate with you.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREFERENCE INPUTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Your Preferences</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    st.markdown("**What genres excite you?**")
    genres_sel = st.multiselect("Genres", GENRES_ALL, default=["Action","Thriller"],
                                label_visibility="collapsed", placeholder="Pick one or more genres…")
    st.markdown("**What's your mood?**")
    moods_sel = st.multiselect("Mood", MOODS_ALL, default=["intense","cerebral"],
                               label_visibility="collapsed", placeholder="Select vibes…")
    st.markdown("**Content type**")
    content_type = st.radio("Type",["Movie","TV Show","Both"],horizontal=True,
                            label_visibility="collapsed",index=2)

with col_right:
    lang_label = st.selectbox("**Language preference**", list(LANGS.keys()), index=0)
    lang_code = LANGS[lang_label]
    min_rating = st.slider("**Minimum rating** (out of 10)",
                           min_value=1.0, max_value=9.5, value=7.0, step=0.1, format="%.1f ⭐")
    min_year = st.slider("**Released no earlier than**",
                         min_value=1990, max_value=2023, value=2005, step=1, format="%d")
    popularity_pref = st.select_slider("**Popularity preference**",
                                       options=["Low-key & obscure","Mixed","Mainstream hits"],
                                       value="Mixed")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
_, btn_col, _ = st.columns([1,2,1])
with btn_col:
    predict_clicked = st.button("🎬  Get My Recommendations")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    if not genres_sel and not moods_sel:
        st.warning("Pick at least one genre or mood to get personalised results.")
        st.stop()

    with st.spinner("Analysing your preferences …"):
        results, user_cluster = recommend(genres_sel, moods_sel, content_type, lang_code,
                                          min_rating, min_year, popularity_pref, n=12)

    cluster_name = CLUSTER_NAMES.get(user_cluster,"Curated")
    cluster_desc = CLUSTER_DESC.get(user_cluster,"")
    st.markdown(f"""
    <div class='cluster-box'>
        <strong>Your content cluster → {cluster_name}</strong><br>{cluster_desc}
    </div>""", unsafe_allow_html=True)

    # Top pick banner
    top = results.iloc[0]
    genre_str = " · ".join(top["genres"])
    lang_display = {v:k for k,v in LANGS.items() if v}.get(top["language"], top["language"].upper())
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1C1C1C 0%,#2a1010 100%);
                border:1px solid #3a1a1a;border-left:4px solid #E50914;
                border-radius:12px;padding:1.6rem 2rem;margin:1.5rem 0 .5rem;">
        <div style="font-size:.62rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;
                    color:#E50914;margin-bottom:.4rem;">Top Pick for You</div>
        <div style="font-size:1.6rem;font-weight:800;color:#F0F0F0;margin-bottom:.3rem;">{top["title"]}</div>
        <div style="font-size:.82rem;color:#8A8A8A;">{top["type"]} &nbsp;·&nbsp; {top["year"]} &nbsp;·&nbsp; ⭐ {top["rating"]} &nbsp;·&nbsp; {genre_str} &nbsp;·&nbsp; {lang_display}</div>
    </div>""", unsafe_allow_html=True)

    # Results grid
    st.markdown("<div class='result-header'>All Recommendations</div>", unsafe_allow_html=True)
    cards = "<div class='rec-grid'>"
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        badge_cls = "badge-movie" if row["type"]=="Movie" else "badge-tv"
        badge_lbl = "Film" if row["type"]=="Movie" else "Series"
        genres_display = " · ".join(row["genres"][:3])
        pop_label = "🔥 Trending" if row["popularity"]>140 else ("💎 Hidden gem" if row["popularity"]<80 else "📺 Popular")
        cards += f"""
        <div class='rec-card'>
            <div class='rec-score'>★ {row['rating']}</div>
            <div class='rec-rank'>#{rank}</div>
            <span class='rec-type-badge {badge_cls}'>{badge_lbl}</span>
            <div class='rec-title'>{row['title']}</div>
            <div class='rec-meta'>{row['year']}<br>{genres_display}<br>{pop_label}</div>
        </div>"""
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    with st.expander("How these recommendations were made"):
        st.markdown(f"""
**Feature extraction:** Your preferences were converted into a numeric feature vector —
release era ({min_year}+), minimum rating ({min_rating}★), popularity preference (*{popularity_pref}*),
plus genre and mood overlap vectors.

**Cluster prediction:** A KMeans model (k=5) trained on the full content library placed
your profile in **Cluster {user_cluster} — {cluster_name}**.

**Scoring:** Each title received points for genre overlap (×3.5 per match), mood match (×2.5),
cluster proximity, language fit, and base quality from ratings.
Hard penalties applied for titles below your rating/year floor. Top {len(results)} surfaced.
        """)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#3a3a3a;font-size:.9rem;">
        ↑ Set your preferences above and hit <strong style='color:#E50914'>Get My Recommendations</strong>
    </div>""", unsafe_allow_html=True)
