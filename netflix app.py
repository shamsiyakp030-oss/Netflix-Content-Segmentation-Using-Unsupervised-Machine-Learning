import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Netflix AI Recommender", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root{--red:#E50914;--red-dark:#B20710;--bg:#060606;--card:#181818;--card-hover:#232323;--border:#2a2a2a;--muted:#777;--text:#FFFFFF;--gold:#F5C518;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--text);}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}

.hero-wrap{position:relative;width:100%;height:100vh;min-height:620px;overflow:hidden;display:flex;align-items:center;justify-content:center;}
.poster-bg{position:absolute;inset:-60px;display:flex;gap:10px;animation:slide-bg 80s linear infinite;will-change:transform;z-index:0;}
@keyframes slide-bg{0%{transform:translateX(0);}100%{transform:translateX(-50%);}}
.poster-col{display:flex;flex-direction:column;gap:10px;flex-shrink:0;}
.poster-col:nth-child(even){margin-top:-80px;animation:drift-up 8s ease-in-out infinite alternate;}
.poster-col:nth-child(odd){animation:drift-down 7s ease-in-out infinite alternate;}
@keyframes drift-up{to{transform:translateY(-20px);}}
@keyframes drift-down{to{transform:translateY(20px);}}
.poster-tile{width:120px;height:180px;border-radius:8px;flex-shrink:0;overflow:hidden;position:relative;background:#111;}
.poster-tile img{width:100%;height:100%;object-fit:cover;display:block;opacity:0.5;}

.hero-overlay-left{position:absolute;inset:0;background:linear-gradient(to right,rgba(6,6,6,0.97) 0%,rgba(6,6,6,0.65) 40%,rgba(6,6,6,0.2) 70%,transparent 100%);z-index:1;}
.hero-overlay-top{position:absolute;inset:0;background:linear-gradient(to bottom,rgba(6,6,6,0.6) 0%,transparent 30%,transparent 70%,rgba(6,6,6,0.97) 100%);z-index:1;}
.hero-content{position:relative;z-index:2;max-width:1200px;width:100%;padding:0 60px;display:flex;flex-direction:column;gap:1.6rem;}
.hero-eyebrow{display:flex;align-items:center;gap:12px;font-size:.72rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--red);}
.hero-eyebrow::before{content:'';width:32px;height:2px;background:var(--red);border-radius:2px;}
.hero-logo{font-family:'Bebas Neue',sans-serif;font-size:clamp(4rem,9vw,8rem);line-height:.9;color:var(--text);letter-spacing:.01em;text-shadow:0 4px 40px rgba(0,0,0,.9);}
.hero-logo em{color:var(--red);font-style:normal;}
.hero-tagline{font-size:1.05rem;color:rgba(255,255,255,.6);max-width:480px;line-height:1.65;}
.scroll-cta{display:flex;align-items:center;gap:10px;font-size:.8rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.35);margin-top:1rem;animation:bounce 2.2s ease-in-out infinite;}
@keyframes bounce{0%,100%{transform:translateY(0);}50%{transform:translateY(7px);}}
.scroll-arrow{width:18px;height:18px;border-right:2px solid rgba(255,255,255,.3);border-bottom:2px solid rgba(255,255,255,.3);transform:rotate(45deg);}

.main-wrap{background:var(--bg);padding:60px 60px 80px;max-width:1200px;margin:0 auto;}
.section-lbl{font-size:.65rem;font-weight:800;letter-spacing:.22em;text-transform:uppercase;color:var(--red);margin:3rem 0 1.6rem;display:flex;align-items:center;gap:14px;}
.section-lbl::after{content:'';flex:1;height:1px;background:linear-gradient(to right,var(--border),transparent);}

div[data-testid="stSlider"]>div>div>div>div{background:var(--red)!important;}
div[data-testid="stSlider"] label{color:rgba(255,255,255,.7)!important;font-size:.78rem!important;font-weight:600!important;letter-spacing:.1em!important;text-transform:uppercase!important;}
div[data-baseweb="select"]>div{background:#1e1e1e!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
div[data-baseweb="select"]>div:hover{border-color:#555!important;}
div[data-baseweb="menu"]{background:#1e1e1e!important;border:1px solid var(--border)!important;border-radius:10px!important;}
div[data-baseweb="option"]{background:#1e1e1e!important;color:var(--text)!important;}
div[data-baseweb="option"]:hover{background:#2a2a2a!important;}
div[data-baseweb="tag"]{background:var(--red)!important;border-radius:999px!important;color:#fff!important;}
div[data-testid="stRadio"] label{color:var(--text)!important;font-size:.88rem!important;}
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p{color:rgba(255,255,255,.6)!important;font-size:.72rem!important;font-weight:700!important;letter-spacing:.14em!important;text-transform:uppercase!important;margin-bottom:.5rem!important;}

.stButton>button{width:100%!important;background:var(--red)!important;color:#fff!important;font-family:'Bebas Neue',sans-serif!important;font-size:1.35rem!important;letter-spacing:.1em!important;padding:1rem 2.5rem!important;border:none!important;border-radius:6px!important;cursor:pointer!important;transition:all .2s ease!important;margin-top:1.5rem!important;}
.stButton>button:hover{background:var(--red-dark)!important;transform:translateY(-1px)!important;box-shadow:0 8px 30px rgba(229,9,20,.35)!important;}

.cluster-reveal{background:linear-gradient(135deg,#1a0a0a,#0f0f0f);border:1px solid #2a0808;border-left:4px solid var(--red);border-radius:0 12px 12px 0;padding:1.4rem 2rem;margin:2rem 0 1rem;display:flex;align-items:flex-start;gap:1.2rem;}
.cluster-reveal .c-icon{font-size:2.2rem;line-height:1;flex-shrink:0;}
.cluster-reveal .c-label{font-size:.6rem;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:var(--red);margin-bottom:.25rem;}
.cluster-reveal .c-name{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.05em;color:#fff;margin-bottom:.3rem;}
.cluster-reveal .c-desc{font-size:.85rem;color:rgba(255,255,255,.5);line-height:1.55;}

.top-pick{position:relative;background:#0f0f0f;border:1px solid #2a1010;border-radius:14px;overflow:hidden;margin-bottom:.5rem;display:grid;grid-template-columns:200px 1fr;min-height:280px;}
.top-pick-poster{position:relative;overflow:hidden;background:#111;}
.top-pick-poster img{width:100%;height:100%;object-fit:cover;display:block;}
.top-pick-body{padding:2rem 2.4rem;display:flex;flex-direction:column;justify-content:center;position:relative;}
.top-pick-body::before{content:'#1 PICK FOR YOU';position:absolute;top:0;right:0;background:var(--red);color:#fff;font-size:.6rem;font-weight:800;letter-spacing:.18em;padding:5px 14px;border-radius:0 0 0 8px;}
.top-pick-title{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;letter-spacing:.03em;color:#fff;margin:.4rem 0 .5rem;line-height:1;}
.top-pick-meta{font-size:.8rem;color:rgba(255,255,255,.45);display:flex;gap:14px;flex-wrap:wrap;margin-top:.5rem;}
.meta-dot{width:3px;height:3px;border-radius:50%;background:rgba(255,255,255,.25);display:inline-block;vertical-align:middle;margin:0 4px;}
.top-pick-rating{display:inline-flex;align-items:center;gap:6px;background:rgba(245,197,24,.12);border:1px solid rgba(245,197,24,.3);color:var(--gold);font-size:.85rem;font-weight:700;padding:4px 12px;border-radius:6px;margin-top:.8rem;width:fit-content;}

.results-lbl{font-size:.62rem;font-weight:800;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin:2rem 0 1rem;}
.rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:14px;}
.rec-card{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;position:relative;transition:border-color .2s,transform .18s;}
.rec-card:hover{border-color:rgba(229,9,20,.6);transform:translateY(-5px);}
.rec-card:first-child{border-color:rgba(229,9,20,.45);}
.rec-poster-wrap{position:relative;width:100%;padding-top:148%;overflow:hidden;background:#111;}
.rec-poster-wrap img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease;}
.rec-card:hover .rec-poster-wrap img{transform:scale(1.07);}
.rec-poster-overlay{position:absolute;inset:0;background:linear-gradient(to top,rgba(0,0,0,.88) 0%,rgba(0,0,0,.15) 45%,transparent 75%);}
.rec-rank-badge{position:absolute;top:8px;left:10px;font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:rgba(255,255,255,.95);line-height:1;text-shadow:0 2px 10px rgba(0,0,0,.9);}
.rec-star-badge{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.65);backdrop-filter:blur(6px);border:1px solid rgba(245,197,24,.4);color:var(--gold);font-size:.62rem;font-weight:800;padding:3px 8px;border-radius:5px;}
.rec-body{padding:.85rem 1rem 1rem;}
.rec-type{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.58rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.45rem;}
.type-m{background:rgba(123,158,255,.12);color:#7B9EFF;border:1px solid rgba(123,158,255,.2);}
.type-t{background:rgba(76,217,100,.12);color:#4CD964;border:1px solid rgba(76,217,100,.2);}
.rec-title{font-size:.88rem;font-weight:700;color:#fff;line-height:1.2;margin-bottom:.3rem;}
.rec-sub{font-size:.67rem;color:rgba(255,255,255,.3);line-height:1.5;}
.rbar-bg{background:rgba(255,255,255,.07);border-radius:3px;height:3px;overflow:hidden;margin-top:.5rem;}
.rbar-fill{height:3px;border-radius:3px;background:linear-gradient(to right,#E50914,#F5C518);}
.rec-pop{margin-top:.4rem;font-size:.64rem;color:rgba(255,255,255,.26);}

details{background:#0f0f0f;border:1px solid var(--border);border-radius:10px;margin-top:2rem;overflow:hidden;}
details summary{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:rgba(255,255,255,.4);padding:1rem 1.4rem;cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px;}
details summary::before{content:'▶';font-size:.55rem;}
details[open] summary::before{content:'▼';}
details .inner{padding:0 1.4rem 1.4rem;font-size:.85rem;color:rgba(255,255,255,.45);line-height:1.7;}
details .inner strong{color:rgba(255,255,255,.7);}
.empty-state{text-align:center;padding:4rem 1rem 2rem;color:#333;font-size:.88rem;}
.empty-state strong{color:var(--red);}
div[data-testid="stVerticalBlock"]>div{gap:0!important;}
div[data-testid="column"]{padding:0 8px!important;}
</style>
""", unsafe_allow_html=True)

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
    ("Outer Banks","TV Show",["Action","Adventure","Mystery"],2020,7.6,130,11000,6,0,"en",["fun","action-packed","mystery"]),
    ("Emily in Paris","TV Show",["Comedy","Drama","Romance"],2020,7.2,130,10000,4,0,"en",["fun","romantic","uplifting"]),
    ("Klaus","Movie",["Animation","Comedy","Drama"],2019,8.2,72,10000,40,10,"en",["emotional","uplifting","whimsical"]),
    ("Dune: Part Two","Movie",["Adventure","Drama","Sci-Fi"],2024,8.5,190,14000,190,714,"en",["epic","cerebral","visual"]),
    ("The Bear","TV Show",["Comedy","Drama"],2022,8.6,115,12000,3,0,"en",["intense","character-study","emotional"]),
    ("Severance","TV Show",["Drama","Mystery","Sci-Fi","Thriller"],2022,8.7,120,11000,15,0,"en",["cerebral","mind-bending","dark"]),
    ("Beef","TV Show",["Comedy","Crime","Drama","Thriller"],2023,8.2,105,10000,5,0,"en",["dark","character-study","emotional"]),
    ("Guillermo del Toro's Pinocchio","Movie",["Animation","Drama","Fantasy"],2022,7.6,70,9000,35,0,"en",["emotional","whimsical","dark"]),
    ("Avatar: The Way of Water","Movie",["Action","Adventure","Fantasy"],2022,7.6,195,11000,350,2320,"en",["epic","visual","action-packed"]),
    ("Enola Holmes","Movie",["Adventure","Comedy","Mystery"],2020,6.6,100,10000,20,0,"en",["fun","mystery","uplifting"]),
]

GENRES_ALL = sorted(set(g for _,_,genres,*_ in CATALOG for g in genres))
MOODS_ALL  = sorted(set(m for *_,moods in CATALOG for m in moods))
LANGS = {"Any":None,"English":"en","Korean":"ko","Spanish":"es","French":"fr","German":"de","Japanese":"ja"}

# ── Hand-picked Wikipedia page titles for entries where the obvious guess
#    ("title" or "title (film)") would land on a disambiguation/word page
#    instead of the actual movie/show article. ──────────────────────────────
WIKI_OVERRIDES = {
    "You": "You (TV series)",
    "Her": "Her (film)",
    "Dark": "Dark (TV series)",
    "Soul": "Soul (2020 film)",
    "Klaus": "Klaus (2019 film)",
    "1899": "1899 (TV series)",
    "Beef": "Beef (TV series)",
    "Wednesday": "Wednesday (TV series)",
    "Lupin": "Lupin (TV series)",
    "The Office (US)": "The Office (American TV series)",
    "Extraction": "Extraction (2020 film)",
    "Bird Box": "Bird Box (film)",
    "Severance": "Severance (TV series)",
    "The Bear": "The Bear (TV series)",
    "Guillermo del Toro's Pinocchio": "Pinocchio (2022 film)",
    "Get Out": "Get Out (film)",
    "Arrival": "Arrival (film)",
    "Joker": "Joker (2019 film)",
    "Barbie": "Barbie (film)",
    "Dune": "Dune (2021 film)",
    "Outer Banks": "Outer Banks (TV series)",
    "Glass Onion": "Glass Onion (film)",
    "All Quiet on the Western Front": "All Quiet on the Western Front (2022 film)",
    "Knives Out": "Knives Out (film)",
    "Enola Holmes": "Enola Holmes (film)",
    "Money Heist": "Money Heist",
    "Narcos": "Narcos",
}

# ── Hardcoded backup TMDB URLs — used only if the live Wikipedia lookup fails ─
POSTERS_FALLBACK = {
    "Inception":                        "https://image.tmdb.org/t/p/w342/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
    "The Dark Knight":                  "https://image.tmdb.org/t/p/w342/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "Interstellar":                     "https://image.tmdb.org/t/p/w342/gEU2QniE6E77NI6lZtvSDaADkop.jpg",
    "Parasite":                         "https://image.tmdb.org/t/p/w342/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    "The Shawshank Redemption":         "https://image.tmdb.org/t/p/w342/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
    "Pulp Fiction":                     "https://image.tmdb.org/t/p/w342/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    "The Matrix":                       "https://image.tmdb.org/t/p/w342/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    "Avengers: Endgame":                "https://image.tmdb.org/t/p/w342/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
    "Spirited Away":                    "https://image.tmdb.org/t/p/w342/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    "The Lion King":                    "https://image.tmdb.org/t/p/w342/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg",
    "Joker":                            "https://image.tmdb.org/t/p/w342/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
    "La La Land":                       "https://image.tmdb.org/t/p/w342/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
    "Get Out":                          "https://image.tmdb.org/t/p/w342/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg",
    "Everything Everywhere All at Once":"https://image.tmdb.org/t/p/w342/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg",
    "Mad Max: Fury Road":               "https://image.tmdb.org/t/p/w342/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg",
    "The Grand Budapest Hotel":         "https://image.tmdb.org/t/p/w342/eWdyYQreja6JGCzqHWXpWHDrrPo.jpg",
    "Her":                              "https://image.tmdb.org/t/p/w342/eCOtqtfvn7mxGqFCGPhpSMZEFEw.jpg",
    "Arrival":                          "https://image.tmdb.org/t/p/w342/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
    "Knives Out":                       "https://image.tmdb.org/t/p/w342/pThyQovXQrws2hmOnNBSstVTpxP.jpg",
    "The Irishman":                     "https://image.tmdb.org/t/p/w342/mbm8k3GFhXS0Rock7RotLX4C0iT.jpg",
    "Dune":                             "https://image.tmdb.org/t/p/w342/d5NXSklpcvkCgnJQ36VnhKXbz9N.jpg",
    "Spider-Man: No Way Home":          "https://image.tmdb.org/t/p/w342/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg",
    "Top Gun: Maverick":                "https://image.tmdb.org/t/p/w342/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
    "Oppenheimer":                      "https://image.tmdb.org/t/p/w342/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
    "Barbie":                           "https://image.tmdb.org/t/p/w342/iuFNMS8vlzmIGlOKOI88uAidAE4.jpg",
    "Squid Game":                       "https://image.tmdb.org/t/p/w342/dDlEmu3EZ0Pgg93K2SVNLCjCSvE.jpg",
    "Breaking Bad":                     "https://image.tmdb.org/t/p/w342/ggFHVNu6YYI5L9pCfOacjizRGt.jpg",
    "Stranger Things":                  "https://image.tmdb.org/t/p/w342/49WJfeN0moxb9IPfGn8AIqMGskD.jpg",
    "The Crown":                        "https://image.tmdb.org/t/p/w342/1M876KPjulVwppEpldhdc8V4o68.jpg",
    "Dark":                             "https://image.tmdb.org/t/p/w342/apbrbWs5wheR0GeNGM9nYFEXVd4.jpg",
    "Money Heist":                      "https://image.tmdb.org/t/p/w342/reEMJA1uzscCbkipeJt2fDNKScl.jpg",
    "The Witcher":                      "https://image.tmdb.org/t/p/w342/7vjaCdMw15FEbXyLQTVa04URsPm.jpg",
    "Ozark":                            "https://image.tmdb.org/t/p/w342/pCGyPVrI9Fzc8f5CBtJQI7RfBLd.jpg",
    "The Office (US)":                  "https://image.tmdb.org/t/p/w342/qWnJzyZhyy74gjpSjIXWmuk0ifX.jpg",
    "Friends":                          "https://image.tmdb.org/t/p/w342/f496cm9enuEsZkSPzCwnTESEK5s.jpg",
    "Succession":                       "https://image.tmdb.org/t/p/w342/xGmdnGr5etCNbJPZTGIiBBaWysa.jpg",
    "Euphoria":                         "https://image.tmdb.org/t/p/w342/3Q0hd3heuScwCUm33NePMpJDGd8.jpg",
    "The Mandalorian":                  "https://image.tmdb.org/t/p/w342/sWgBv7LV2PRoQgkxwlibdGXKz1S.jpg",
    "Narcos":                           "https://image.tmdb.org/t/p/w342/rTmal9fDbwh5F0waol2hq35U4ah.jpg",
    "Black Mirror":                     "https://image.tmdb.org/t/p/w342/7PRddO7z7mcPi21nZjpd8tQkFeA.jpg",
    "Mindhunter":                       "https://image.tmdb.org/t/p/w342/e7JlejGDCECLDxlRJOm9gJaFCH7.jpg",
    "The Queen's Gambit":               "https://image.tmdb.org/t/p/w342/zU0htwkhNvBQdVSIKB9s6hgVeFK.jpg",
    "Lupin":                            "https://image.tmdb.org/t/p/w342/sg4YaqOwCt7iCUyPohPkLJwTkfL.jpg",
    "Wednesday":                        "https://image.tmdb.org/t/p/w342/9PFonBhy4cQy7Jz20NpMygczOkv.jpg",
    "Peaky Blinders":                   "https://image.tmdb.org/t/p/w342/vUUqzWa2LnHIVqkaKVlVGkPaQca.jpg",
    "Soul":                             "https://image.tmdb.org/t/p/w342/hm58Jw4Lw8OIeECIq5qyPYhAeRJ.jpg",
    "Bird Box":                         "https://image.tmdb.org/t/p/w342/rGfGfgL2pEPCfhIvqHXieXFn7gp.jpg",
    "Extraction":                       "https://image.tmdb.org/t/p/w342/sGAHbowLCiPHbTKMPhkS7ej8o1T.jpg",
    "Bridgerton":                       "https://image.tmdb.org/t/p/w342/luoKpgVwi1E5nQsi7W0UuKHu2Rq.jpg",
    "You":                              "https://image.tmdb.org/t/p/w342/Bb4mHgurjxYhmpfAvTCRcaVQaXY.jpg",
    "Cobra Kai":                        "https://image.tmdb.org/t/p/w342/obLBdhLxheKg8Li1qZs5IgP6i21.jpg",
    "1899":                             "https://image.tmdb.org/t/p/w342/fnPITisBbCyGHwJLLaKpPuFJ1fI.jpg",
    "Glass Onion":                      "https://image.tmdb.org/t/p/w342/vYkHgXKmqUBiqpGGzAqNGSFiZov.jpg",
    "All Quiet on the Western Front":   "https://image.tmdb.org/t/p/w342/hSoDbApRWOFADvSPxIRPToNPkJE.jpg",
    "Outer Banks":                      "https://image.tmdb.org/t/p/w342/jgSZkFMbOijqEZFJVOqGw3bq8GG.jpg",
    "Emily in Paris":                   "https://image.tmdb.org/t/p/w342/aoAZgnmMzY9vVcgxHpOtCGxzBEm.jpg",
    "Klaus":                            "https://image.tmdb.org/t/p/w342/6BxOZuSaGBqQxhSAMGYEXXKO7Po.jpg",
    "Dune: Part Two":                   "https://image.tmdb.org/t/p/w342/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg",
    "The Bear":                         "https://image.tmdb.org/t/p/w342/sHFlbKS3WLqMnp9t2ghADIJFnuQ.jpg",
    "Severance":                        "https://image.tmdb.org/t/p/w342/nYDPmFBSFqMDFoRYmGj3MBXlFdD.jpg",
    "Beef":                             "https://image.tmdb.org/t/p/w342/lyOoQKBaqm4b7h2pSBCzJJkJvST.jpg",
    "Guillermo del Toro's Pinocchio":   "https://image.tmdb.org/t/p/w342/8pjWz2lt29KTUbUkffDNwkdKoMR.jpg",
    "Avatar: The Way of Water":         "https://image.tmdb.org/t/p/w342/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
    "Enola Holmes":                     "https://image.tmdb.org/t/p/w342/riYInlsq2kf1AWoGm80JQW5dLKp.jpg",
}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24, max_entries=200)
def fetch_poster_url(title: str, year: int, content_type: str) -> str:
    """Resolve a real poster/cover image for a title.

    Tries the Wikipedia REST summary API (no API key required) with a few
    sensible page-title variants, falls back to a hardcoded TMDB URL, and
    finally returns an empty string if nothing could be found (the UI then
    shows a styled placeholder instead of a broken image).
    """
    if title in WIKI_OVERRIDES:
        candidates = [WIKI_OVERRIDES[title]]
    elif content_type == "Movie":
        candidates = [title, f"{title} ({year} film)", f"{title} (film)"]
    else:
        candidates = [title, f"{title} (TV series)", f"{title} ({year} TV series)"]

    headers = {"User-Agent": "StreamlitNetflixRecommender/1.0 (educational demo)"}
    for page_title in candidates:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title, safe='')}"
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("type") == "disambiguation":
                    continue
                img = (data.get("originalimage") or {}).get("source") or (data.get("thumbnail") or {}).get("source")
                if img:
                    return img
        except Exception:
            continue

    return POSTERS_FALLBACK.get(title, "")


@st.cache_resource(show_spinner=False)
def build_poster_cache():
    cache = {}
    for t, ct, *_rest in CATALOG:
        year = _rest[1]  # genres, year, rating ... -> index 1 in _rest is year
        cache[t] = fetch_poster_url(t, year, ct)
    return cache


POSTER_CACHE = build_poster_cache()


def poster_for(title: str) -> str:
    return POSTER_CACHE.get(title) or POSTERS_FALLBACK.get(title, "")


# ── HERO POSTER BG — every tile uses a real poster image ──────────────────────
def make_poster_bg():
    titles = [r[0] for r in CATALOG]
    cols = [titles[i:i+5] for i in range(0,len(titles),5)]
    doubled = cols + cols
    html = ""
    for col in doubled:
        html += "<div class='poster-col'>"
        for t in col * 3:
            url = poster_for(t)
            if url:
                html += f"<div class='poster-tile'><img src='{url}' alt='' loading='lazy'/></div>"
            else:
                html += f"<div class='poster-tile' style='background:#1a1a1a;'></div>"
        html += "</div>"
    return html

poster_bg_html = make_poster_bg()

# ── ML MODEL ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_model():
    rows = [{"title":t,"type":ct,"genres":g,"year":y,"rating":r,"popularity":p,
             "vote_count":v,"budget":b,"revenue":rv,"language":l,"moods":m}
            for t,ct,g,y,r,p,v,b,rv,l,m in CATALOG]
    df = pd.DataFrame(rows)
    X = df[["year","rating","popularity","vote_count","budget","revenue"]].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=5,random_state=42,n_init=10)
    df["cluster"] = km.fit_predict(Xs)
    return df, scaler, km

catalog_df, scaler, km_model = build_model()

def recommend(genres_sel, moods_sel, content_type, lang_code, min_rating, min_year, popularity_pref, n=12):
    df = catalog_df.copy()
    pop_map={"Low-key & obscure":40,"Mixed":100,"Mainstream hits":180}
    user_pop=pop_map.get(popularity_pref,100)
    user_vec=np.array([[min_year,min_rating,user_pop,10000,50,200]],dtype=float)
    user_cluster=int(km_model.predict(scaler.transform(user_vec))[0])
    scores=[]
    for _,row in df.iterrows():
        s=0.0
        if genres_sel: s+=len(set(genres_sel)&set(row["genres"]))*3.5
        if moods_sel:  s+=len(set(moods_sel)&set(row["moods"]))*2.5
        if content_type!="Both" and row["type"]!=content_type: s-=5
        if lang_code and row["language"]!=lang_code: s-=4
        if row["rating"]<min_rating: s-=10
        if row["year"]<min_year: s-=3
        if popularity_pref=="Low-key & obscure" and row["popularity"]<90: s+=2
        elif popularity_pref=="Mainstream hits" and row["popularity"]>130: s+=2
        if row["cluster"]==user_cluster: s+=2
        s+=(row["rating"]-5)*0.8
        scores.append(s)
    df["score"]=scores
    return df.sort_values("score",ascending=False).head(n), user_cluster

CLUSTER_META={
    0:("🔥","Trending & Highly Engaged","Your taste aligns with content generating real buzz — recent releases with strong audience engagement."),
    1:("🏛️","Classic & Enduring","You lean toward timeless masterpieces with consistent critical acclaim and multi-generational appeal."),
    2:("💎","Premium Blockbusters","You're drawn to high-production blockbuster-tier content — big budgets, massive audiences, maximum impact."),
    3:("🔍","Hidden Gems","Your preferences reveal a taste for underseen titles with dedicated followings — quality over hype."),
    4:("🎯","Niche & Cult Favourites","You gravitate toward genre-defining cult content that rewards patient, discerning viewers."),
}

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="poster-bg">{poster_bg_html}</div>
    <div class="hero-overlay-left"></div>
    <div class="hero-overlay-top"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">AI-Powered Recommendations</div>
        <div class="hero-logo">Find Your<br>Next <em>Obsession</em></div>
        <p class="hero-tagline">Tell us what you're in the mood for. Our ML model segments content and surfaces titles most likely to become your next binge.</p>
        <div class="scroll-cta"><div class="scroll-arrow"></div>Set your preferences below</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FORM ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)
st.markdown("<div class='section-lbl'>Your Taste Profile</div>", unsafe_allow_html=True)

col_left,col_right = st.columns([1,1],gap="large")
with col_left:
    st.markdown("**What genres excite you?**")
    genres_sel=st.multiselect("Genres",GENRES_ALL,default=["Action","Thriller"],label_visibility="collapsed",placeholder="Pick genres…")
    st.markdown("**What's your mood tonight?**")
    moods_sel=st.multiselect("Mood",MOODS_ALL,default=["intense","cerebral"],label_visibility="collapsed",placeholder="Select vibes…")
    st.markdown("**Content type**")
    content_type=st.radio("Type",["Movie","TV Show","Both"],horizontal=True,label_visibility="collapsed",index=2)
with col_right:
    lang_label=st.selectbox("**Language**",list(LANGS.keys()),index=0)
    lang_code=LANGS[lang_label]
    min_rating=st.slider("**Minimum IMDb rating**",min_value=1.0,max_value=9.5,value=7.0,step=0.1,format="%.1f")
    min_year=st.slider("**Earliest release year**",min_value=1990,max_value=2023,value=2005,step=1)
    popularity_pref=st.select_slider("**Popularity appetite**",options=["Low-key & obscure","Mixed","Mainstream hits"],value="Mixed")

st.markdown("")
_,btn_col,_=st.columns([1,2,1])
with btn_col:
    clicked=st.button("▶  Find My Perfect Watch")

# ── RESULTS ───────────────────────────────────────────────────────────────────
if clicked:
    if not genres_sel and not moods_sel:
        st.warning("Select at least one genre or mood to continue.")
        st.stop()

    with st.spinner(""):
        results,uc=recommend(genres_sel,moods_sel,content_type,lang_code,min_rating,min_year,popularity_pref,n=12)

    icon,cname,cdesc=CLUSTER_META.get(uc,("🎬","Curated for You",""))
    st.markdown(f"""
    <div class="cluster-reveal">
        <div class="c-icon">{icon}</div>
        <div>
            <div class="c-label">Your Content Cluster</div>
            <div class="c-name">{cname}</div>
            <div class="c-desc">{cdesc}</div>
        </div>
    </div>""",unsafe_allow_html=True)

    # ── Top pick ──────────────────────────────────────────────────────────────
    top=results.iloc[0]
    top_url=poster_for(top["title"])
    bc_top="type-m" if top["type"]=="Movie" else "type-t"
    bl_top="Film" if top["type"]=="Movie" else "Series"
    genres_str=" · ".join(top["genres"])
    lang_disp={v:k for k,v in LANGS.items() if v}.get(top["language"],top["language"].upper())
    pop_top="🔥 Trending" if top["popularity"]>140 else ("💎 Hidden gem" if top["popularity"]<80 else "📺 Popular")

    if top_url:
        top_poster_html = f"""<img src="{top_url}" alt="{top['title']}"
                 onerror="this.parentElement.style.background='#1a0a0a';this.style.display='none'"/>"""
    else:
        top_poster_html = """<div style="position:absolute;inset:0;background:linear-gradient(160deg,#1a1a2e,#16213e,#0f3460);
                 display:flex;align-items:center;justify-content:center;font-size:2.6rem;">🎬</div>"""

    st.markdown(f"""
    <div class="top-pick">
        <div class="top-pick-poster">
            {top_poster_html}
        </div>
        <div class="top-pick-body">
            <span class="rec-type {bc_top}">{bl_top}</span>
            <div class="top-pick-title">{top["title"]}</div>
            <div class="top-pick-rating">★ {top["rating"]} / 10</div>
            <div class="top-pick-meta">
                <span>{top["year"]}</span><span class="meta-dot"></span>
                <span>{genres_str}</span><span class="meta-dot"></span>
                <span>{lang_disp}</span><span class="meta-dot"></span>
                <span>{pop_top}</span>
            </div>
        </div>
    </div>""",unsafe_allow_html=True)

    # ── Cards grid — every card gets its real poster ───────────────────────────
    st.markdown("<div class='results-lbl'>All Recommendations</div>",unsafe_allow_html=True)

    cards="<div class='rec-grid'>"
    for rank,(_,row) in enumerate(results.iterrows(),1):
        bc="type-m" if row["type"]=="Movie" else "type-t"
        bl="Film" if row["type"]=="Movie" else "Series"
        gd=" · ".join(row["genres"][:2])
        pl="🔥 Trending" if row["popularity"]>140 else ("💎 Hidden gem" if row["popularity"]<80 else "📺 Popular")
        bar_w=int((row["rating"]/10)*100)
        url=poster_for(row["title"])

        if url:
            poster_html = f"""<img src="{url}" alt="{row['title']}" loading="lazy"
                     onerror="this.style.display='none';document.getElementById('fb{rank}').style.display='flex'"/>
                <div id="fb{rank}" style="display:none;position:absolute;inset:0;
                     background:linear-gradient(160deg,#1a1a2e,#16213e,#0f3460);
                     align-items:center;justify-content:center;flex-direction:column;gap:8px;padding:12px;">
                    <div style="font-size:2.2rem">🎬</div>
                    <div style="font-size:.65rem;font-weight:700;color:rgba(255,255,255,.5);text-align:center;line-height:1.3">{row['title']}</div>
                </div>"""
        else:
            poster_html = f"""<div style="position:absolute;inset:0;
                     background:linear-gradient(160deg,#1a1a2e,#16213e,#0f3460);
                     display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;padding:12px;">
                    <div style="font-size:2.2rem">🎬</div>
                    <div style="font-size:.65rem;font-weight:700;color:rgba(255,255,255,.5);text-align:center;line-height:1.3">{row['title']}</div>
                </div>"""

        cards += f"""
        <div class='rec-card'>
            <div class='rec-poster-wrap'>
                {poster_html}
                <div class='rec-poster-overlay'></div>
                <div class='rec-rank-badge'>#{rank}</div>
                <div class='rec-star-badge'>★ {row['rating']}</div>
            </div>
            <div class='rec-body'>
                <span class='rec-type {bc}'>{bl}</span>
                <div class='rec-title'>{row['title']}</div>
                <div class='rec-sub'>{row['year']} · {gd}</div>
                <div class='rbar-bg'><div class='rbar-fill' style='width:{bar_w}%'></div></div>
                <div class='rec-pop'>{pl}</div>
            </div>
        </div>"""
    cards+="</div>"
    st.markdown(cards,unsafe_allow_html=True)

    st.markdown(f"""
    <details>
        <summary>How these recommendations were made</summary>
        <div class='inner'>
            <strong>Feature extraction</strong> — Your inputs encoded into a numeric vector:
            era ({min_year}+), rating floor ({min_rating}★), popularity appetite (<em>{popularity_pref}</em>),
            genre and mood overlap vectors from each title's metadata.<br><br>
            <strong>KMeans cluster prediction</strong> — KMeans (k=5) mapped your profile to
            <strong>Cluster {uc} — {cname}</strong>. Same-cluster titles receive a proximity bonus.<br><br>
            <strong>Personalised scoring</strong> — Genre overlap ×3.5 · Mood match ×2.5 ·
            Cluster proximity +2 · Rating quality ×0.8. Hard penalties for titles below your floor.
            Top {len(results)} surfaced and ranked.
        </div>
    </details>""",unsafe_allow_html=True)

else:
    st.markdown("<div class='empty-state'>Set your preferences and hit <strong>▶ Find My Perfect Watch</strong></div>",unsafe_allow_html=True)

st.markdown("</div>",unsafe_allow_html=True)
