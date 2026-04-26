import streamlit as st
import json
import pandas as pd
import numpy as np
import os, requests, glob, re
from scipy.stats import poisson
from datetime import datetime, timedelta, timezone
from duckduckgo_search import DDGS
from groq import Groq
from scraper_xg import get_understat_xg, get_market_values

# --- CONFIGURAZIONE CHIAVI (CLOUD SECRETS) ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
API_KEY_ODDS = "a310fd7b74f24f2736a57c6caf768118"
API_KEY_DATA = "c299e4a676a54d48a642f20bca7f4480"

JSONBIN_API_KEY = st.secrets.get("JSONBIN_API_KEY", "")
JSONBIN_BIN_ID = st.secrets.get("JSONBIN_BIN_ID", "")

try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    groq_client = None

st.set_page_config(page_title="M4 STRATEGIC TERMINAL", layout="wide", initial_sidebar_state="expanded")

# --- GESTIONE TEMA ---
theme = st.sidebar.select_slider("⚙️ TEMA", options=["LIGHT", "DARK"], value="LIGHT")
if theme == "DARK":
    bg, card, txt, border, stat_bg, lbl = "#0b0e11", "#161b22", "#ffffff", "#30363d", "#0d1117", "#58a6ff"
else:
    bg, card, txt, border, stat_bg, lbl = "#f0f2f5", "#ffffff", "#1a1d23", "#e0e4e9", "#f8f9fa", "#0056b3"

# --- CSS ---
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
    .match-card {{ background-color: {card}; border-radius: 12px; padding: 25px; margin-bottom: 8px; border: 1px solid {border}; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    .team-name {{ font-size: 19px; font-weight: 800; color: #58a6ff; text-transform: uppercase; }}
    .label-header {{ color: {lbl}; font-size: 15px !important; font-weight: 900; text-transform: uppercase; display: block; margin-bottom: 5px; border-bottom: 1px solid {border}; }}
    .match-date {{ font-size: 15px !important; color: #3b82f6 !important; font-weight: 700; display: block; margin-top: 5px; }}
    .stat-container {{ background-color: {stat_bg}; border: 1px solid {border}; border-radius: 8px; padding: 10px; text-align: center; height: 100%; }}
    .val-p-green {{ color: #28a745; font-size: 17px; font-weight: 800; }}
    .val-p-red {{ color: #dc3545; font-size: 17px; font-weight: 800; }}
    .top-mix-row {{ background-color: {card}; border: 1px solid {border}; border-radius: 8px; padding: 15px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }}
    </style>
    """, unsafe_allow_html=True)

# --- REGISTRO PREDIZIONI (CLOUD PERSISTENTE) ---
PREDICTIONS_FILE = "database/predictions.json"

def load_predictions():
    if JSONBIN_API_KEY and JSONBIN_BIN_ID:
        try:
            url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest"
            headers = {"X-Master-Key": JSONBIN_API_KEY}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                record = r.json().get("record", {})
                if isinstance(record, dict) and "data" in record: return record["data"]
                elif isinstance(record, list): return record
        except: pass
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data: return data["data"]
                elif isinstance(data, list): return data
        except: return []
    return []

def save_predictions(preds):
    if JSONBIN_API_KEY and JSONBIN_BIN_ID:
        try:
            url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
            headers = {"X-Master-Key": JSONBIN_API_KEY, "Content-Type": "application/json"}
            requests.put(url, json={"data": preds}, headers=headers, timeout=5)
        except: pass
    os.makedirs("database", exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({"data": preds}, f, ensure_ascii=False, indent=2)

def standardizza_mercato(testo):
    t = testo.lower()
    if "under 2.5" in t or "under2.5" in t: return "UNDER_2.5"
    if "over 2.5" in t or "over2.5" in t: return "OVER_2.5"
    if "under 3.5" in t: return "UNDER_3.5"
    if "over 3.5" in t: return "OVER_3.5"
    if "gg" in t or "goal/goal" in t or "entrambe segnano" in t: return "GG"
    if "ng" in t or "no goal" in t or "nessuna segna" in t: return "NG"
    if "pareggio" in t or "draw" in t or " x " in t: return "X"
    if "vittoria" in t or "vince" in t or "1 -" in t or "1-" in t: return "1"
    if "2 -" in t or "2-" in t: return "2"
    return "ALTRO"

def save_prediction_entry(match_id, h, a, camp, giornata, match_date, pronostico_sicuro, top3, prob_sicuro, risultati_attesi):
    preds = load_predictions()
    for p in preds:
        if p.get("match_id") == match_id: return  
    mercato_std = standardizza_mercato(pronostico_sicuro)
    preds.append({
        "match_id": match_id, "home": h, "away": a, "campionato": camp, "giornata": giornata,
        "data": match_date, "pronostico_sicuro": pronostico_sicuro, "mercato_standard": mercato_std,
        "top3": top3, "prob_sicuro": prob_sicuro, "risultati_attesi": risultati_attesi,
        "risultato_reale": None, "esito": None, "salvato_il": datetime.now().strftime("%d/%m/%Y %H:%M")
    })
    save_predictions(preds)

def aggiorna_risultati_reali(api_key):
    preds = load_predictions()
    aggiornate = 0
    pending = [p for p in preds if p.get("esito") in [None, "⏳"] and p.get("campionato")]
    if not pending: return 0
    l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
    for camp, comp in l_map.items():
        camp_pending = [p for p in pending if p["campionato"] == camp]
        if not camp_pending: continue
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{comp}/matches", headers={"X-Auth-Token": api_key}, params={"status": "FINISHED"})
            risultati_api = {m["id"]: m for m in r.json().get("matches", [])}
        except: continue
        for p in camp_pending:
            m_id = p.get("match_id")
            if not m_id or m_id not in risultati_api: p["esito"] = "⏳"; continue
            match = risultati_api[m_id]
            gh, ga = match["score"]["fullTime"]["home"], match["score"]["fullTime"]["away"]
            if gh is None: continue
            p["risultato_reale"] = f"{gh}-{ga}"
            p["esito"] = verifica_esito(p.get("mercato_standard", ""), gh, ga, p["home"], p["away"])
            aggiornate += 1
    if aggiornate > 0: save_predictions(preds)
    return aggiornate

def verifica_esito(mercato_std, gh, ga, home, away):
    totale = gh + ga; m = mercato_std.upper()
    if m == "UNDER_2.5": return "✅" if totale < 3 else "❌"
    if m == "OVER_2.5": return "✅" if totale > 2 else "❌"
    if m == "UNDER_3.5": return "✅" if totale < 4 else "❌"
    if m == "OVER_3.5": return "✅" if totale > 3 else "❌"
    if m == "GG": return "✅" if gh > 0 and ga > 0 else "❌"
    if m == "NG": return "✅" if gh == 0 or ga == 0 else "❌"
    if m == "X": return "✅" if gh == ga else "❌"
    if m == "1": return "✅" if gh > ga else "❌"
    if m == "2": return "✅" if ga > gh else "❌"
    return "⏳"

# --- MOTORE LOGICO ---
def clean_name(name):
    n = str(name).strip()
    m = {"Manchester United": "Man United", "Manchester City": "Man City", "Tottenham Hotspur": "Tottenham",
         "Inter Milan": "Inter", "AC Milan": "Milan", "Atalanta BC": "Atalanta", "Hellas Verona": "Verona"}
    n = m.get(n, n)
    for r in ["BC", "FC", "AC ", "AS ", "1907", "Calcio"]: n = n.replace(r, "")
    return n.strip()

@st.cache_data
def get_league_engine(camp_key):
    p = {"Serie A": "SerieA", "Premier League": "Premier", "La Liga": "LaLiga", "Bundesliga": "Bundesliga"}
    prefix = p.get(camp_key)
    if not prefix: return None
    files_storici = sorted(glob.glob(f"./database/{prefix}_20*.csv"))
    files_live = glob.glob(f"./database/{prefix}_Live.csv")
    if not files_storici and not files_live: return None
    dfs = []
    for f in files_storici:
        try: df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False); df_tmp['peso'] = 1.0; dfs.append(df_tmp)
        except: pass
    for f in files_live:
        try: df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False); df_tmp['peso'] = 4.0; dfs.append(df_tmp)
        except: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date')
    df['HomeClean'] = df['HomeTeam'].apply(clean_name); df['AwayClean'] = df['AwayTeam'].apply(clean_name)
    avg_h = np.average(df['FTHG'].dropna(), weights=df.loc[df['FTHG'].notna(), 'peso'])
    avg_a = np.average(df['FTAG'].dropna(), weights=df.loc[df['FTAG'].notna(), 'peso'])
    xg_data = get_understat_xg(camp_key); mkt_values = get_market_values()
    league_xg = league_xga = None
    if xg_data and len(xg_data) >= 10:
        league_xg = np.mean([v['xG_avg'] for v in xg_data.values()]); league_xga = np.mean([v['xGA_avg'] for v in xg_data.values()])
    stats = {}
    for t in pd.concat([df['HomeClean'], df['AwayClean']]).unique():
        h_h = df[df['HomeClean']==t]; a_h = df[df['AwayClean']==t]
        if xg_data and t in xg_data and league_xg and league_xga:
            att = xg_data[t]['xG_avg'] / league_xg; defe = xg_data[t]['xGA_avg'] / league_xga
        else:
            att_h = h_h['FTHG'].mean() / avg_h if not h_h.empty else 1.0; att_a = a_h['FTAG'].mean() / avg_a if not a_h.empty else 1.0
            def_h = h_h['FTAG'].mean() / avg_a if not h_h.empty else 1.0; def_a = a_h['FTHG'].mean() / avg_h if not a_h.empty else 1.0
            att = (att_h + att_a) / 2; defe = (def_h + def_a) / 2
        val = mkt_values.get(t, 50)
        stats[t] = {'att': att * (1 + (val/50000)), 'def': defe * (1 - (val/50000)), 'val': val}
    return stats, avg_h, avg_a, df

def get_full_poisson(h_e, a_e):
    h_p = [poisson.pmf(i, h_e) for i in range(8)]; a_p = [poisson.pmf(i, a_e) for i in range(8)]
    matrix = np.outer(h_p, a_p)
    def get_u(limit): return sum([matrix[i,j] for i in range(8) for j in range(8) if i+j < limit])
    return {"1": np.sum(np.tril(matrix, -1)), "X": np.sum(np.diag(matrix)), "2": np.sum(np.triu(matrix, 1)),
            "u15": get_u(1.5), "u25": get_u(2.5), "u35": get_u(3.5), "gg": (1-h_p[0])*(1-a_p[0])}

def calcola_segnali(risultati, infraset_giocate, infraset_programmate, stand, giornata=None, tutte_stand=None):
    mult_att = 1.0; mult_def = 1.0
    score_forma = 0; n_ris = 0
    for r in risultati:
        if r.endswith("(V)"): score_forma += 1; n_ris += 1
        elif r.endswith("(P)"): score_forma -= 1; n_ris += 1
        elif r.endswith("(X)"): n_ris += 1
    if n_ris > 0:
        forma_norm = score_forma / n_ris; delta_att = forma_norm * 0.08; delta_def = -forma_norm * 0.05
        mult_att += delta_att; mult_def += delta_def
    if infraset_giocate: mult_att -= 0.04; mult_def += 0.06
    if infraset_programmate: mult_att -= 0.02
    if stand:
        pg = stand.get("pg", 1); gf = stand.get("gf", 0); gs = stand.get("gs", 1)
        if (gf / pg) if pg > 0 else 1.0 > 1.8: mult_att += 0.03
        elif (gf / pg) if pg > 0 else 1.0 < 0.9: mult_att -= 0.03
        if (gs / pg) if pg > 0 else 1.0 < 0.8: mult_def -= 0.04
        elif (gs / pg) if pg > 0 else 1.0 > 1.5: mult_def += 0.04
    return max(0.78, min(1.22, mult_att)), max(0.78, min(1.22, mult_def)), ""

# --- DATI CONTESTUALI ---
def get_team_fd_id(team_name, camp_sel):
    for match in st.session_state.get("live_data", []):
        for team in [match["homeTeam"], match["awayTeam"]]:
            if clean_name(team.get("shortName", "") or team.get("name", "")).lower() == clean_name(team_name).lower(): return team["id"]
    return None

@st.cache_data(ttl=3600)
def get_ultimi_risultati_fd(team_id, camp_sel, n=5):
    comp = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}.get(camp_sel, "SA")
    try:
        r = requests.get(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers={"X-Auth-Token": API_KEY_DATA}, params={"status": "FINISHED", "limit": 15, "competitions": comp})
        risultati = []
        for match in r.json().get("matches", [])[-n:]:
            gh, ga, winner = match["score"]["fullTime"]["home"], match["score"]["fullTime"]["away"], match["score"]["winner"]
            esito = "V" if (match["homeTeam"]["id"] == team_id and winner == "HOME_TEAM") or (match["awayTeam"]["id"] == team_id and winner == "AWAY_TEAM") else ("X" if winner == "DRAW" else "P")
            risultati.append(f"{match['homeTeam'].get('shortName','?')} {gh}-{ga} {match['awayTeam'].get('shortName','?')} ({esito})")
        return risultati
    except: return []

@st.cache_data(ttl=3600)
def get_infraset_data(team_id, camp_code, match_date_str, now_utc_str):
    match_date = datetime.fromisoformat(match_date_str); now_utc = datetime.fromisoformat(now_utc_str)
    window_start = match_date - timedelta(days=7); giocate = []; programmate = []
    try:
        r_fin = requests.get(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers={"X-Auth-Token": API_KEY_DATA}, params={"status": "FINISHED", "limit": 10})
        for match in r_fin.json().get("matches", []):
            if match.get("competition", {}).get("code", "") == camp_code: continue
            try: match_dt = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
            except: continue
            if match_dt < window_start or match_dt >= match_date: continue
            gh, ga = match["score"]["fullTime"].get("home"), match["score"]["fullTime"].get("away")
            if gh is not None and ga is not None: giocate.append(f"{match_dt.strftime('%d/%m')} {match.get('competition',{}).get('name','')}: {gh}-{ga}")
        r_prg = requests.get(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers={"X-Auth-Token": API_KEY_DATA}, params={"status": "SCHEDULED,TIMED", "limit": 5})
        for match in r_prg.json().get("matches", []):
            if match.get("competition", {}).get("code", "") == camp_code: continue
            try: match_dt = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
            except: continue
            if match_dt >= match_date or match_dt <= now_utc: continue
            programmate.append(f"[PROG {match_dt.strftime('%d/%m')}] {match.get('competition',{}).get('name','')}")
    except: pass
    return giocate, programmate

def get_contesto_partita(h, a, camp_sel):
    h_id, a_id = get_team_fd_id(h, camp_sel), get_team_fd_id(a, camp_sel)
    contesto = {"h_risultati": get_ultimi_risultati_fd(h_id, camp_sel) if h_id else [], "a_risultati": get_ultimi_risultati_fd(a_id, camp_sel) if a_id else [], "h_infortunati": [], "a_infortunati": [], "h_infraset": [], "a_infraset": [], "h_infraset_prog": [], "a_infraset_prog": []}
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"{h} infortunati squalificati {camp_sel} 2026", max_results=2): contesto["h_infortunati"].append(r["body"][:300])
            for r in ddgs.text(f"{a} infortunati squalificati {camp_sel} 2026", max_results=2): contesto["a_infortunati"].append(r["body"][:300])
    except: pass
    camp_code = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}.get(camp_sel, "SA")
    match_date = now_utc = datetime.now(timezone.utc); match_id_found = None
    for mx in st.session_state.get("live_data", []):
        if clean_name(h) in clean_name(mx["homeTeam"].get("shortName", "") or mx["homeTeam"].get("name", "")):
            try: match_date = datetime.fromisoformat(mx["utcDate"].replace("Z", "+00:00")); match_id_found = mx["id"]
            except: pass
            break
    if h_id: contesto["h_infraset"], contesto["h_infraset_prog"] = get_infraset_data(h_id, camp_code, match_date.isoformat(), now_utc.isoformat())
    if a_id: contesto["a_infraset"], contesto["a_infraset_prog"] = get_infraset_data(a_id, camp_code, match_date.isoformat(), now_utc.isoformat())
    return contesto, match_id_found

# --- TOP MIX LOGIC ---
@st.cache_data(ttl=1800, show_spinner="Calcolando Top 10 Globale...")
def fetch_and_calc_top_mix():
    all_preds = []; missing_leagues = []
    leagues = ["Serie A", "Premier League", "La Liga", "Bundesliga"]
    l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
    for league in leagues:
        engine = get_league_engine(league)
        if not engine: 
            missing_leagues.append(league)
            continue
        team_stats, avg_h, avg_a, _ = engine
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[league]}/matches?status=TIMED,SCHEDULED", headers={'X-Auth-Token': API_KEY_DATA})
            matches = r.json().get('matches', [])
            if not matches: continue
            giornate = sorted(list(set([m['matchday'] for m in matches]))); g_next = giornate[0]
            matches_next = [m for m in matches if m['matchday'] == g_next]
        except: continue
        for match in matches_next:
            h = match['homeTeam'].get('shortName') or match['homeTeam'].get('name', '?'); a = match['awayTeam'].get('shortName') or match['awayTeam'].get('name', '?')
            h_cl, a_cl = clean_name(h), clean_name(a)
            h_s = team_stats.get(h_cl, {"att": 1.0, "def": 1.0}); a_s = team_stats.get(a_cl, {"att": 1.0, "def": 1.0})
            h_exp = h_s["att"] * a_s["def"] * avg_h; a_exp = a_s["att"] * h_s["def"] * avg_a
            m_poisson = get_full_poisson(h_exp, a_exp)
            mercati = {f"Vittoria {h}": m_poisson["1"], "Pareggio": m_poisson["X"], f"Vittoria {a}": m_poisson["2"], "Over 2.5": 1 - m_poisson["u25"], "Under 2.5": m_poisson["u25"], "GG": m_poisson["gg"], "NG": 1 - m_poisson["gg"]}
            best_mkt = max(mercati, key=mercati.get); best_prob = mercati[best_mkt]
            all_preds.append({"league": league, "giornata": g_next, "home": h, "away": a, "match_id": match.get("id"), "utcDate": match['utcDate'], "market": best_mkt, "prob": best_prob, "prob_val": round(best_prob * 100, 1)})
    
    # Ordina e ritorna i top 10, e la lista di chi è saltato
    return sorted(all_preds, key=lambda x: x['prob'], reverse=True)[:10], missing_leagues

# --- ANALISI RAPIDA ---
def analisi_rapida_giornata(matches, team_stats, avg_h, avg_a, camp_sel, classifica_sess, giornata_n):
    salvate = 0
    for match in matches:
        try:
            h = match['homeTeam'].get('shortName') or match['homeTeam'].get('name', '?'); a = match['awayTeam'].get('shortName') or match['awayTeam'].get('name', '?')
            h_cl, a_cl = clean_name(h), clean_name(a); m_id = match.get('id')
            if not m_id: continue
            try: match_date_str = (datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=2)).strftime("%d/%m/%Y %H:%M")
            except: match_date_str = ""
            h_s = team_stats.get(h_cl, {"att": 1.0, "def": 1.0}); a_s = team_stats.get(a_cl, {"att": 1.0, "def": 1.0})
            h_exp = h_s["att"] * a_s["def"] * avg_h; a_exp = a_s["att"] * h_s["def"] * avg_a
            m = get_full_poisson(h_exp, a_exp)
            mercati = {f"Vittoria {h}": m["1"], "Pareggio": m["X"], f"Vittoria {a}": m["2"], "Over 2.5": 1 - m["u25"], "Under 2.5": m["u25
