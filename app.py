import streamlit as st
import pandas as pd
import numpy as np
import os, requests, glob, re
from scipy.stats import poisson
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
import google.generativeai as genai
from scraper_xg import get_understat_xg, get_market_values

# --- CONFIGURAZIONE CHIAVI (CLOUD SAFE) ---
# Prende la chiave dai Secrets di Streamlit Cloud
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Inizializzazione AI con gestione errore 404
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Cambiato in gemini-1.5-flash (senza modelli/ o v1beta)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Errore configurazione AI: {e}")
        gemini_model = None
else:
    st.error("⚠️ Chiave GEMINI_API_KEY non trovata nei Secrets di Streamlit!")
    gemini_model = None

API_KEY_ODDS = "a310fd7b74f24f2736a57c6caf768118"
API_KEY_DATA = "c299e4a676a54d48a642f20bca7f4480"

st.set_page_config(page_title="M4 STRATEGIC TERMINAL", layout="wide", initial_sidebar_state="expanded")

# --- GESTIONE TEMA ---
theme = st.sidebar.select_slider("⚙️ TEMA", options=["LIGHT", "DARK"], value="DARK")
if theme == "DARK":
    bg, card, txt, border, stat_bg, lbl = "#0b0e11", "#161b22", "#ffffff", "#30363d", "#0d1117", "#58a6ff"
else:
    bg, card, txt, border, stat_bg, lbl = "#f0f2f5", "#ffffff", "#1a1d23", "#e0e4e9", "#f8f9fa", "#0056b3"

# --- CSS PROFESSIONALE ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [data-testid="stapp"] {{ background-color: {bg} !important; color: {txt} !important; font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {bg}; }}
    .maradona-header {{
        background: linear-gradient(rgba(0, 45, 91, 0.8), rgba(0, 45, 91, 0.8)), 
                    url('https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&fit=crop&w=1600&q=80');
        background-size: cover; background-position: center; padding: 40px;
        border-radius: 0 0 20px 20px; text-align: center; margin: -60px -60px 30px -60px; color: white;
    }}
    .match-card {{ background-color: {card}; border-radius: 12px; padding: 20px; margin-bottom: 8px; border: 1px solid {border}; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    .team-name {{ font-size: 19px; font-weight: 800; color: #58a6ff; text-transform: uppercase; }}
    .match-date {{ font-size: 15px !important; color: #3b82f6 !important; font-weight: 700; display: block; margin-top: 8px; }}
    .stat-container {{ background-color: {stat_bg}; border: 1px solid {border}; border-radius: 8px; padding: 10px; text-align: center; height: 100%; }}
    .label-header {{ color: {lbl}; font-size: 15px !important; font-weight: 900; text-transform: uppercase; display: block; margin-bottom: 5px; border-bottom: 1px solid {border}; }}
    .val-p-green {{ color: #28a745; font-size: 17px; font-weight: 800; }}
    .val-p-red {{ color: #dc3545; font-size: 17px; font-weight: 800; }}
    div[data-testid="stDialog"] div[role="dialog"] {{ background-color: {card} !important; color: {txt} !important; border: 1px solid {border}; }}
    [data-testid="stVerticalBlock"] > div {{ gap: 0rem !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- MOTORE LOGICO ---
def clean_name(name):
    n = str(name).strip()
    m = {"Manchester United": "Man United", "Manchester City": "Man City", "Inter Milan": "Inter", "AC Milan": "Milan", "Atalanta BC": "Atalanta", "Hellas Verona": "Verona", "Lazio Roma": "Lazio"}
    n = m.get(n, n)
    for r in ["BC", "FC", "AC ", "AS ", "1907", "Calcio", "1900"]: n = n.replace(r, "")
    return n.strip()

@st.cache_data
def get_league_engine(camp_key):
    p = {"Serie A": "SerieA*", "Premier League": "Premier*", "La Liga": "LaLiga*", "Bundesliga": "Bundesliga*"}
    files = glob.glob(f"./database/{p.get(camp_key)}")
    if not files: return None
    df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in files])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date')
    df['HomeClean'] = df['HomeTeam'].apply(clean_name); df['AwayClean'] = df['AwayTeam'].apply(clean_name)
    xg_data = get_understat_xg(camp_key)
    mkt_values = get_market_values()
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
    stats = {}
    for t in pd.concat([df['HomeClean'], df['AwayClean']]).unique():
        h_h, a_h = df[df['HomeClean']==t], df[df['AwayClean']==t]
        if xg_data and t in xg_data: att, defe = xg_data[t]['xG_avg'], xg_data[t]['xGA_avg']
        else:
            att = ((h_h['FTHG'].mean()/avg_h)+(a_h['FTAG'].mean()/avg_a))/2 if not h_h.empty else 1.0
            defe = ((h_h['FTAG'].mean()/avg_a)+(a_h['FTHG'].mean()/avg_h))/2 if not a_h.empty else 1.0
        val = mkt_values.get(t, 50)
        stats[t] = {'att': att * (1 + (val/6000)), 'def': defe * (1 - (val/6000)), 'val': val}
    return stats, avg_h, avg_a, df

def get_full_poisson(h_e, a_e):
    h_p = [poisson.pmf(i, h_e) for i in range(8)]; a_p = [poisson.pmf(i, a_e) for i in range(8)]
    matrix = np.outer(h_p, a_p)
    def get_u(limit): return sum([matrix[i,j] for i in range(8) for j in range(8) if i+j < limit])
    return {"1": np.sum(np.tril(matrix, -1)), "X": np.sum(np.diag(matrix)), "2": np.sum(np.triu(matrix, 1)),
            "u15": get_u(1.5), "u25": get_u(2.5), "u35": get_u(3.5), "gg": (1-h_p[0])*(1-a_p[0])}

# --- POPUP STRATEGICO ---
@st.dialog("STRATEGIC ANALYSIS", width="large")
def show_details(h, a, m):
    if not gemini_model:
        st.error("Billy non ha un cervello attivo. Controlla la API Key.")
        return

    with st.spinner("Billy Walters (AI) sta analizzando..."):
        query = f"formazioni ufficiali infortunati {h} {a} 2026 sky sport"
        news = ""
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5): news += f" {r['body']}"
        except: news = "News non disponibili."
        
        prompt = f"""Sei Billy Walters, analista tecnico. RISPONDI IN ITALIANO.
        Analizza {h} vs {a}. AI Poisson: 1({m['1']:.0%}), X({m['X']:.0%}), 2({m['2']:.0%}).
        News: {news}.
        REGOLE: 1. Analisi formazioni (3 righe). 2. PRONOSTICO MASTER. 3. PROB PERCENTUALE."""
        
        try:
            response = gemini_model.generate_content(prompt)
            st.markdown(f"<div style='line-height:1.2; font-size:16px;'>{response.text.replace('*','')}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Errore Billy: {e}")

# --- UI PRINCIPALE ---
st.markdown('<div class="maradona-header"><h1>M4 STRATEGIC TERMINAL</h1><p>Powered by Billy Walters AI</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("🎩 Billy Walters Chat")
    camp_sel = st.selectbox("CAMPIONATO", ["Serie A", "Premier League", "La Liga", "Bundesliga"])
    if st.button("🔄 SINCRONIZZA TURNO"):
        l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
        st.session_state.live_data = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/matches?status=SCHEDULED", headers={'X-Auth-Token': API_KEY_DATA}).json().get('matches', [])
    if "live_data" in st.session_state and st.session_state.live_data:
        giornate = sorted(list(set([m['matchday'] for m in st.session_state.live_data])))
        g_sel = st.selectbox("FILTRA PER GIORNATA", giornate)

engine = get_league_engine(camp_sel)
if 'live_data' in st.session_state and engine:
    team_stats, avg_h, avg_a, df_full = engine
    matches = [m for m in st.session_state.live_data if m['matchday'] == g_sel]
    st.subheader(f"🏟️ {camp_sel.upper()} - GIORNATA {g_sel}")
    
    st.markdown("""<div class='table-head'>
        <div style='width:25%'>Incontro / Orario</div><div style='flex:1; text-align:center'>1X2</div>
        <div style='flex:1; text-align:center'>U/O 1.5</div><div style='flex:1; text-align:center'>U/O 2.5</div>
        <div style='flex:1; text-align:center'>U/O 3.5</div><div style='flex:1; text-align:center'>GG / NG</div><div style='width:5%'></div>
    </div>""", unsafe_allow_html=True)

    for idx, match in enumerate(matches):
        h_api, a_api = match['homeTeam']['shortName'], match['awayTeam']['shortName']
        h_cl, a_cl = clean_name(h_api), clean_name(a_api)
        dt = (datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=1)).strftime("%d/%m | %H:%M")
        h_s, a_s = team_stats.get(h_cl, {'att': 1.0, 'def': 1.0}), team_stats.get(a_cl, {'att': 1.0, 'def': 1.0})
        m = get_full_poisson(h_s['att'] * a_s['def'] * avg_h, a_s['att'] * h_s['def'] * avg_a)
        
        with st.container():
            st.markdown(f'<div class="match-card">', unsafe_allow_html=True)
            c_h, c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.2, 0.9, 0.9, 0.9, 1, 0.4])
            with c_h: st.markdown(f"<span class='team-name'>{h_api}<br>{a_api}</span><br><span class='match-date'>🕒 {dt.replace('Jan','Gen')}</span>", unsafe_allow_html=True)
            with c1: st.markdown(f"<div class='stat-container'><span class='label-header'>1 X 2</span><span class='val-p-green'>{m['1']:.0%} {m['X']:.0%} {m['2']:.0%}</span></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 1.5</span><span class='val-p-red'>{m['u15']:.0%}</span>/<span class='val-p-green'>{(1-m['u15']):.0%}</span></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 2.5</span><span class='val-p-red'>{m['u25']:.0%}</span>/<span class='val-p-green'>{(1-m['u25']):.0%}</span></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 3.5</span><span class='val-p-red'>{m['u35']:.0%}</span>/<span class='val-p-green'>{(1-m['u35']):.0%}</span></div>", unsafe_allow_html=True)
            with c5: st.markdown(f"<div class='stat-container'><span class='label-header'>GG / NG</span><span class='val-p-green'>{m['gg']:.0%}</span>/<span class='val-p-red'>{(1-m['gg']):.0%}</span></div>", unsafe_allow_html=True)
            with c6:
                st.write("<br>", unsafe_allow_html=True); st.button("🔍", key=f"ex_{idx}", on_click=show_details, args=(h_api, a_api, m))
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("👋 Terminale Pronto. Sincronizza per caricare la giornata.")
