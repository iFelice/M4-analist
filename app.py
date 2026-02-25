import streamlit as st
import pandas as pd
import numpy as np
import os, requests, glob, re
from scipy.stats import poisson
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
import google.generativeai as genai
from scraper_xg import get_understat_xg, get_market_values

# --- CONFIGURAZIONE CORE ---
GEMINI_API_KEY = "AIzaSyBPyuLxsTcTgqgndgLP3B8R_UpcrkuDA6E"
API_KEY_ODDS = "a310fd7b74f24f2736a57c6caf768118"
API_KEY_DATA = "c299e4a676a54d48a642f20bca7f4480"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="M4 Strategic Terminal", layout="wide")

# --- CSS: CLONAZIONE ESATTA REPLIT / NERDYTIPS DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stapp"] { background-color: #06090f !important; color: #f1f5f9 !important; font-family: 'Inter', sans-serif; }
    .stApp { background-color: #06090f; }

    /* Top Metrics */
    .metric-container { display: flex; gap: 15px; margin-bottom: 25px; }
    .metric-card {
        background: #0d1117; border: 1px solid #30363d; border-radius: 12px;
        padding: 15px 20px; flex: 1; text-align: left;
    }
    .metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight: 700; }
    .metric-value { color: #ffffff; font-size: 22px; font-weight: 800; display: block; }

    /* Table System */
    .table-header-row {
        display: flex; padding: 10px 20px; color: #8b949e; font-size: 11px;
        font-weight: 700; text-transform: uppercase; border-bottom: 1px solid #21262d;
    }
    .match-row {
        background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 20px; margin-bottom: 5px; display: flex; align-items: center;
    }
    .match-row:hover { border-color: #3b82f6; }

    .col-match { width: 22%; }
    .col-market { flex: 1; text-align: center; border-right: 1px solid #21262d; padding: 0 5px; }
    .col-market:last-of-type { border-right: none; }
    .col-action { width: 8%; text-align: right; }
    
    .team-txt { font-size: 15px; font-weight: 700; color: #ffffff; }
    .time-txt { font-size: 11px; color: #8b949e; }
    
    /* Prob Badges */
    .p-badge { 
        padding: 4px 6px; border-radius: 5px; font-weight: 700; font-size: 13px; 
        display: inline-block; min-width: 40px; margin: 0 2px;
    }
    .high { color: #39d353; }
    .low { color: #f85149; }
    .mid { color: #f1f5f9; }

    .edge-badge { 
        background: rgba(57, 211, 83, 0.1); color: #39d353; 
        padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 800;
        border: 1px solid rgba(57, 211, 83, 0.3);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #3b82f6 !important; color: white !important;
        font-size: 12px !important; border-radius: 6px !important; border: none !important;
    }
    
    /* Dialog Popup Dark */
    div[data-testid="stDialog"] div[role="dialog"] { background-color: #0d1117 !important; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTORE LOGICO ---
def clean_name(name):
    n = str(name).strip()
    m = {"Manchester United": "Man United", "Manchester City": "Man City", "Inter Milan": "Inter", "AC Milan": "Milan", "Atalanta BC": "Atalanta", "Hellas Verona": "Verona"}
    n = m.get(n, n)
    for r in ["BC", "FC", "AC ", "AS ", "1907", "Calcio"]: n = n.replace(r, "")
    return n.strip()

@st.cache_data
def get_engine(camp):
    files = glob.glob(f"./database/{camp.replace(' ','')}*")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['HomeTeam','AwayTeam','FTR']).sort_values('Date')
    df['HomeClean'] = df['HomeTeam'].apply(clean_name); df['AwayClean'] = df['AwayTeam'].apply(clean_name)
    
    xg_data = get_understat_xg(camp)
    mkt = get_market_values()
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
    stats = {}
    for t in pd.concat([df['HomeClean'], df['AwayClean']]).unique():
        h_h, a_h = df[df['HomeClean']==t], df[df['AwayClean']==t]
        if xg_data and t in xg_data: att, defe = xg_data[t]['xG_avg'], xg_data[t]['xGA_avg']
        else:
            att = ((h_h['FTHG'].mean()/avg_h)+(a_h['FTAG'].mean()/avg_a))/2 if not h_h.empty else 1.0
            defe = ((h_h['FTAG'].mean()/avg_a)+(a_h['FTHG'].mean()/avg_h))/2 if not a_h.empty else 1.0
        val = mkt.get(t, 50)
        stats[t] = {'att': att * (1 + (val/6000)), 'def': defe * (1 - (val/6000)), 'val': val}
    return stats, avg_h, avg_a, df

def get_poisson(h_e, a_e):
    h_p = [poisson.pmf(i, h_e) for i in range(8)]; a_p = [poisson.pmf(i, a_e) for i in range(8)]
    matrix = np.outer(h_p, a_p)
    def get_u(limit): return sum([matrix[i,j] for i in range(8) for j in range(8) if i+j < limit])
    return {"1": np.sum(np.tril(matrix, -1)), "X": np.sum(np.diag(matrix)), "2": np.sum(np.triu(matrix, 1)),
            "u15": get_u(1.5), "u25": get_u(2.5), "u35": get_u(3.5), "gg": (1-h_p[0])*(1-a_p[0])}

def get_badge_class(prob):
    if prob > 0.60: return "high"
    if prob < 0.25: return "low"
    return "mid"

# --- POPUP BILLY WALTERS (GEMINI VERSION) ---
@st.dialog("STRATEGIC ANALYSIS", width="large")
def show_details(h, a, m):
    with st.spinner("Billy Walters sta interrogando i database..."):
        query = f"formazioni ufficiali infortunati {h} {a} 2026 whoscored sky sport"
        news = ""
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5): news += f" {r['body']}"
        
        prompt = f"""Sei Billy Walters, analista betting cinico. Analizza {h} vs {a}. 
        AI Stats: 1({m['1']:.0%}), X({m['X']:.0%}), 2({m['2']:.0%}), O2.5({1-m['u25']:.0%}).
        News Web: {news}. Rispondi in italiano.
        1. Analisi formazioni/meteo (3 righe). 
        2. SENTENZA: [scelta migliore].
        3. PROB: [numero intero probabilità]."""
        
        try:
            response = gemini_model.generate_content(prompt)
            st.markdown(f"<div style='background:#1c2128; padding:20px; border-radius:10px; border-left:5px solid #3b82f6; white-space: pre-wrap;'>{response.text.replace('*','')}</div>", unsafe_allow_html=True)
        except: st.error("Errore API Gemini.")

# --- UI PRINCIPALE (REPLIT CLONE) ---
st.markdown("""
    <div class='main-header'>
        <div><h2 style='margin:0'>🛡️ M4 Strategic Terminal</h2><p style='color:#8b949e; font-size:13px'>Dati TIER 3 • Elaborazione Real-Time</p></div>
        <div class='tier-badge'>ONLINE EDITION</div>
    </div>
    """, unsafe_allow_html=True)

# 1. Top Cards
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="metric-card"><span class="metric-label">Match Caricati</span><span class="metric-value">20</span></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="metric-card"><span class="metric-label">Value Trovati</span><span class="metric-value">6</span></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="metric-card"><span class="metric-label">Best Edge</span><span class="metric-value">+14.2%</span></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="metric-card"><span class="metric-label">Avg GG %</span><span class="metric-value">54%</span></div>', unsafe_allow_html=True)

# 2. Sidebar e Sync
with st.sidebar:
    camp_sel = st.selectbox("CAMPIONATO", ["Serie A", "Premier League", "La Liga", "Bundesliga"])
    if st.button("🔄 SINCRONIZZA"):
        l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
        st.session_state.live_data = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/matches?status=SCHEDULED", headers={'X-Auth-Token': API_KEY_DATA}).json().get('matches', [])

# 3. Tabella Partite
engine = get_engine(camp_sel)
if 'live_data' in st.session_state and engine:
    team_stats, avg_h, avg_a, df_full = engine
    g_sel = st.session_state.live_data[0]['matchday']
    st.markdown(f"**{camp_sel.upper()} - GIORNATA {g_sel}**")
    
    st.markdown("""<div class='table-header-row'>
        <div style='width:22%'>Incontro / Orario</div><div style='flex:1;text-align:center'>1 X 2</div><div style='flex:1;text-align:center'>U/O 1.5</div>
        <div style='flex:1;text-align:center'>U/O 2.5</div><div style='flex:1;text-align:center'>U/O 3.5</div><div style='flex:1;text-align:center'>GG / NG</div><div style='width:8%'></div>
    </div>""", unsafe_allow_html=True)

    for idx, match in enumerate([m for m in st.session_state.live_data if m['matchday'] == g_sel]):
        h_raw, a_raw = match['homeTeam']['shortName'], match['awayTeam']['shortName']
        h_cl, a_cl = clean_name(h_raw), clean_name(a_raw)
        h_s, a_s = team_stats.get(h_cl, {'att':1,'def':1}), team_stats.get(a_cl, {'att':1,'def':1})
        m = get_poisson(h_s['att']*a_s['def']*avg_h, a_s['att']*h_s['def']*avg_a)
        dt = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%d %b | %H:%M")

        st.markdown(f"""
        <div class="match-row">
            <div class="col-match"><span class="team-txt">{h_raw} - {a_raw}</span><br><span class="time-txt">🕒 {dt}</span></div>
            <div class="col-market">
                <span class="p-badge {get_badge_class(m['1'])}">{m['1']:.0%}</span>
                <span class="p-badge {get_badge_class(m['X'])}">{m['X']:.0%}</span>
                <span class="p-badge {get_badge_class(m['2'])}">{m['2']:.0%}</span>
            </div>
            <div class="col-market"><span class="p-badge low">{m['u15']:.0%}</span><span class="p-badge high">{(1-m['u15']):.0%}</span></div>
            <div class="col-market"><span class="p-badge low">{m['u25']:.0%}</span><span class="p-badge high">{(1-m['u25']):.0%}</span></div>
            <div class="col-market"><span class="p-badge low">{m['u35']:.0%}</span><span class="p-badge high">{(1-m['u35']):.0%}</span></div>
            <div class="col-market"><span class="p-badge high">{m['gg']:.0%}</span><span class="p-badge low">{(1-m['gg']):.0%}</span></div>
        """, unsafe_allow_html=True)
        with st.columns([9.2, 0.8])[1]:
            st.button("INFO", key=f"btn_{idx}", on_click=show_details, args=(h_raw, a_raw, m))
        st.markdown("</div>", unsafe_allow_html=True)