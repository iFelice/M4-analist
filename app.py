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
theme = st.sidebar.select_slider("⚙️ TEMA", options=["LIGHT", "DARK"], value="DARK")
if theme == "DARK":
    bg, card, txt, border, stat_bg, lbl, app_bg = "#0b0e11", "#161b22", "#ffffff", "#30363d", "#0d1117", "#58a6ff", "#0d1117"
else:
    bg, card, txt, border, stat_bg, lbl, app_bg = "#f0f2f5", "#ffffff", "#1a1d23", "#e0e4e9", "#f8f9fa", "#0056b3", "#ffffff"

# --- CSS RESTYLED ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    html, body, [data-testid="stapp"] {{ background-color: {app_bg} !important; color: {txt} !important; font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {app_bg}; }}
    .maradona-header {{
        background: linear-gradient(135deg, rgba(0, 40, 80, 0.95), rgba(0, 80, 150, 0.9)), 
                    url('https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&fit=crop&w=1600&q=80');
        background-size: cover; background-position: center; padding: 50px;
        border-radius: 0 0 24px 24px; text-align: center; margin: -60px -60px 40px -60px; color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}
    .match-card {{ 
        background-color: {card}; border-radius: 16px; padding: 20px; margin-bottom: 12px; 
        border: 1px solid {border}; box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: all 0.2s;
    }}
    .match-card:hover {{ border-color: #58a6ff; box-shadow: 0 6px 20px rgba(88, 166, 255, 0.15); }}
    .team-name {{ font-size: 20px; font-weight: 900; color: #58a6ff; text-transform: uppercase; letter-spacing: -0.5px; }}
    .label-header {{ color: {lbl}; font-size: 11px !important; font-weight: 800; text-transform: uppercase; display: block; margin-bottom: 8px; letter-spacing: 1px; }}
    .match-date {{ font-size: 13px !important; color: #8b949e !important; font-weight: 600; display: block; margin-top: 4px; }}
    .match-result {{ font-size: 20px; font-weight: 900; color: #3fb950; margin-top: 6px; display: block; letter-spacing: 1px; }}
    .stat-container {{ background-color: {stat_bg}; border: 1px solid {border}; border-radius: 10px; padding: 12px; text-align: center; height: 100%; }}
    .val-p-green {{ color: #3fb950; font-size: 18px; font-weight: 900; }}
    .val-p-red {{ color: #f85149; font-size: 18px; font-weight: 900; }}
    .val-q {{ color: #8b949e; font-size: 12px; font-weight: 600; display: block; margin-top: 3px; font-family: monospace; }}
    .top-mix-row {{ background-color: {card}; border: 1px solid {border}; border-radius: 12px; padding: 16px 20px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }}
    .pred-box {{ background-color: {card}; border-radius: 12px; padding: 16px; margin-bottom: 12px; border-left: 5px solid {border}; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    .pred-box-ok {{ border-left-color: #3fb950 !important; }}
    .pred-box-ko {{ border-left-color: #f85149 !important; }}
    .pred-box-wait {{ border-left-color: #8b949e !important; }}
    .pred-type {{ font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 3px 8px; border-radius: 6px; margin-left: 8px; letter-spacing: 0.5px; }}
    .type-mix {{ background-color: #d29922; color: #0d1117; }}
    .type-single {{ background-color: #58a6ff; color: #0d1117; }}
    
    /* AI Dialog Styles */
    .ai-section {{ margin-bottom: 20px; }}
    .ai-section-title {{ font-size: 12px; font-weight: 800; text-transform: uppercase; color: #8b949e; margin-bottom: 8px; letter-spacing: 1px; border-bottom: 1px solid {border}; padding-bottom: 4px; }}
    .ai-pronostico {{ background: linear-gradient(135deg, #1a1e23, #0d1117); border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin: 15px 0; }}
    .ai-pronostico-market {{ font-size: 24px; font-weight: 900; color: #3fb950; text-transform: uppercase; }}
    .ai-pronostico-prob {{ font-size: 40px; font-weight: 900; color: #ffffff; margin: 5px 0; }}
    .ai-confidence {{ font-size: 32px; font-weight: 900; color: #58a6ff; }}
    .ai-text {{ font-size: 14px; line-height: 1.6; color: {txt}; }}
    
    /* Streamlit overrides */
    [data-testid="stVerticalBlock"] > div {{ gap: 0rem !important; }}
    .stButton > button {{ border-radius: 8px; font-weight: 600; }}
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
    tipo = "Top Mix" if "Top Mix" in pronostico_sicuro else "Analisi"
    preds.append({
        "match_id": match_id, "home": h, "away": a, "campionato": camp, "giornata": giornata,
        "data": match_date, "pronostico_sicuro": pronostico_sicuro, "mercato_standard": mercato_std,
        "top3": top3, "prob_sicuro": prob_sicuro, "risultati_attesi": risultati_attesi,
        "risultato_reale": None, "esito": None, "tipo": tipo, "salvato_il": datetime.now().strftime("%d/%m/%Y %H:%M")
    })
    save_predictions(preds)

def aggiorna_risultati_reali(api_key):
    preds = load_predictions()
    aggiornate = 0
    pending = [p for p in preds if p.get("esito") in [None, "⏳"] and p.get("campionato") and p.get("giornata")]
    if not pending: return 0
    l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
    from collections import defaultdict
    grouped = defaultdict(list)
    for p in pending:
        grouped[(p["campionato"], p["giornata"])].append(p)
    for (camp, giornata), camp_pending in grouped.items():
        comp = l_map.get(camp)
        if not comp: continue
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{comp}/matches", headers={"X-Auth-Token": api_key}, params={"matchday": giornata, "status": "FINISHED"})
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
    files_base = glob.glob(f"./database/{prefix}.csv")
    if not files_storici and not files_live and not files_base: return None
    dfs = []
    for f in files_storici:
        try: df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False); df_tmp['peso'] = 1.0; dfs.append(df_tmp)
        except: pass
    for f in files_live:
        try: df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False); df_tmp['peso'] = 4.0; dfs.append(df_tmp)
        except: pass
    for f in files_base:
        try: df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False); df_tmp['peso'] = 3.0; dfs.append(df_tmp)
        except: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date')
    df['HomeClean'] = df['HomeTeam'].apply(clean_name); df['AwayClean'] = df['AwayTeam'].apply(clean_name)
    if 'peso' not in df.columns: df['peso'] = 1.0
    avg_h = np.average(df['FTHG'].dropna(), weights=df.loc[df['FTHG'].notna(), 'peso'])
    avg_a = np.average(df['FTAG'].dropna(), weights=df.loc[df['FTAG'].notna(), 'peso'])
    xg_data = get_understat_xg(camp_key); mkt_values = get_market_values()
    league_xg = league_xga = None
    if xg_data and len(xg_data) >= 10:
        league_xg = np.mean([v['xG_avg'] for v in xg_data.values()])
        league_xga = np.mean([v['xGA_avg'] for v in xg_data.values()])
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
        ratio_gf = (gf / pg) if pg > 0 else 1.0; ratio_gs = (gs / pg) if pg > 0 else 1.0
        if ratio_gf > 1.8: mult_att += 0.03
        elif ratio_gf < 0.9: mult_att -= 0.03
        if ratio_gs < 0.8: mult_def -= 0.04
        elif ratio_gs > 1.5: mult_def += 0.04
    return max(0.78, min(1.22, mult_att)), max(0.78, min(1.22, mult_def)), ""

# --- DATI CONTESTUALI (OTTIMIZZATO) ---
def get_team_fd_id(team_name, camp_sel):
    for match in st.session_state.get("live_data", []):
        for team in [match["homeTeam"], match["awayTeam"]]:
            nome = team.get("shortName", "") or team.get("name", "")
            if clean_name(nome).lower() == clean_name(team_name).lower(): return team["id"]
    return None

@st.cache_data(ttl=3600)
def get_team_context_fd(team_id, match_date_str, camp_code):
    match_date = datetime.fromisoformat(match_date_str)
    window_start = match_date - timedelta(days=7)
    form = []; infra_giocate = []; infra_programmate = []
    try:
        r = requests.get(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers={"X-Auth-Token": API_KEY_DATA}, params={"status": "FINISHED,TIMED,SCHEDULED", "limit": 15})
        if r.status_code == 429: return ["LIMITE_API"], [], []
        for match in r.json().get("matches", []):
            status = match.get("status", ""); comp_code = match.get("competition", {}).get("code", ""); comp_name = match.get("competition", {}).get("name", "")
            try: match_dt = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
            except: continue
            if status == "FINISHED" and comp_code == camp_code:
                gh = match["score"]["fullTime"]["home"]; ga = match["score"]["fullTime"]["away"]; winner = match["score"]["winner"]
                is_home = match["homeTeam"]["id"] == team_id
                if (is_home and winner == "HOME_TEAM") or (not is_home and winner == "AWAY_TEAM"): esito = "V"
                elif winner == "DRAW": esito = "X"
                else: esito = "P"
                form.append(f"{match['homeTeam'].get('shortName','?')} {gh}-{ga} {match['awayTeam'].get('shortName','?')} ({esito})")
            elif status == "FINISHED" and comp_code != camp_code:
                if match_dt >= window_start and match_dt < match_date:
                    gh = match["score"]["fullTime"].get("home"); ga = match["score"]["fullTime"].get("away")
                    if gh is not None: infra_giocate.append(f"{match_dt.strftime('%d/%m')} {comp_name}: {gh}-{ga}")
            elif status in ["TIMED", "SCHEDULED"] and comp_code != camp_code:
                if match_dt > datetime.now(timezone.utc) and match_dt < match_date:
                    infra_programmate.append(f"[PROG {match_dt.strftime('%d/%m')}] {comp_name}")
    except: pass
    return form[-5:], infra_giocate, infra_programmate

def get_contesto_partita(h, a, camp_sel):
    h_id, a_id = get_team_fd_id(h, camp_sel), get_team_fd_id(a, camp_sel)
    contesto = {"h_risultati": [], "a_risultati": [], "h_infortunati": [], "a_infortunati": [], "h_infraset": [], "a_infraset": [], "h_infraset_prog": [], "a_infraset_prog": []}
    camp_code = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}.get(camp_sel, "SA")
    match_date = now_utc = datetime.now(timezone.utc); match_id_found = None
    for mx in st.session_state.get("live_data", []):
        if clean_name(h) in clean_name(mx["homeTeam"].get("shortName", "") or mx["homeTeam"].get("name", "")):
            try: match_date = datetime.fromisoformat(mx["utcDate"].replace("Z", "+00:00")); match_id_found = mx["id"]
            except: pass
            break
    if h_id: contesto["h_risultati"], contesto["h_infraset"], contesto["h_infraset_prog"] = get_team_context_fd(h_id, match_date.isoformat(), camp_code)
    if a_id: contesto["a_risultati"], contesto["a_infraset"], contesto["a_infraset_prog"] = get_team_context_fd(a_id, match_date.isoformat(), camp_code)
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"{h} infortunati squalificati {camp_sel} 2026", max_results=2): contesto["h_infortunati"].append(r["body"][:300])
            for r in ddgs.text(f"{a} infortunati squalificati {camp_sel} 2026", max_results=2): contesto["a_infortunati"].append(r["body"][:300])
    except: pass
    return contesto, match_id_found

# --- TOP MIX LOGIC ---
@st.cache_data(ttl=1800, show_spinner="Calcolando Top 10 Globale...")
def fetch_and_calc_top_mix():
    all_preds = []; missing_leagues = []
    leagues = ["Serie A", "Premier League", "La Liga", "Bundesliga"]
    l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
    for league in leagues:
        engine = get_league_engine(league)
        if not engine: missing_leagues.append(league); continue
        team_stats, avg_h, avg_a, _ = engine
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[league]}/matches", headers={'X-Auth-Token': API_KEY_DATA}, params={"status": "TIMED,SCHEDULED"})
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
            mercati = {f"Vittoria {h}": m["1"], "Pareggio": m["X"], f"Vittoria {a}": m["2"], "Over 2.5": 1 - m["u25"], "Under 2.5": m["u25"], "GG": m["gg"], "NG": 1 - m["gg"]}
            best_mkt = max(mercati, key=mercati.get); best_prob = mercati[best_mkt]
            pronostico_sicuro = f"{best_mkt} - {best_prob:.0%} - analisi automatica Poisson"
            altri = sorted([(k, v) for k, v in mercati.items() if k != best_mkt], key=lambda x: -x[1])
            top3 = [f"{i+1}. {k} - {v:.0%}" for i, (k, v) in enumerate(altri[:3])]
            save_prediction_entry(m_id, h, a, camp_sel, giornata_n, match_date_str, pronostico_sicuro, top3, round(best_prob*100, 1), "")
            salvate += 1
        except: pass
    return salvate

# --- POPUP AI CON UI AVANZATA ---
@st.dialog("STRATEGIC ANALYSIS", width="large")
def show_details(h, a, m, camp_sel="Serie A"):
    if not groq_client: st.error("Billy non e' configurato."); return
    with st.spinner("Billy Walters sta analizzando..."):
        contesto, match_id = get_contesto_partita(h, a, camp_sel)
        h_ris = contesto.get("h_risultati", []); a_ris = contesto.get("a_risultati", [])
        limite_api = "LIMITE_API" in h_ris or "LIMITE_API" in a_ris
        senza_dati = (not h_ris or h_ris == ["LIMITE_API"]) and (not a_ris or a_ris == ["LIMITE_API"])
        if limite_api: st.warning("⏳ **Limite API raggiunto!** Attendi 60s. Analisi basata solo su xG/Poisson.")
        elif senza_dati: st.info("ℹ️ **Dati recenti non disponibili**. Analisi basata su xG/Poisson.")
            
        classifica_sess = st.session_state.get("classifica", {}); h_cl_key = clean_name(h); a_cl_key = clean_name(a)
        h_stand_s = classifica_sess.get(h_cl_key, {}); a_stand_s = classifica_sess.get(a_cl_key, {})
        giornata_corrente = st.session_state.get("live_data", [{}])[0].get("matchday") if st.session_state.get("live_data") else None
        h_mult_att, h_mult_def, h_note_seg = calcola_segnali(contesto.get("h_risultati", []), contesto.get("h_infraset", []), contesto.get("h_infraset_prog", []), h_stand_s, giornata=giornata_corrente, tutte_stand=classifica_sess)
        a_mult_att, a_mult_def, a_note_seg = calcola_segnali(contesto.get("a_risultati", []), contesto.get("a_infraset", []), contesto.get("a_infraset_prog", []), a_stand_s, giornata=giornata_corrente, tutte_stand=classifica_sess)
        engine_data = get_league_engine(camp_sel)
        if engine_data:
            team_stats_p, avg_h_p, avg_a_p, _ = engine_data
            h_s_p = team_stats_p.get(h_cl_key, {"att": 1.0, "def": 1.0}); a_s_p = team_stats_p.get(a_cl_key, {"att": 1.0, "def": 1.0})
            h_exp = h_s_p["att"] * h_mult_att * a_s_p["def"] * a_mult_def * avg_h_p; a_exp = a_s_p["att"] * a_mult_att * h_s_p["def"] * h_mult_def * avg_a_p
        else: h_exp = 1.3; a_exp = 1.1
        m_adj = get_full_poisson(h_exp, a_exp)
        p1 = m_adj['1']; pX = m_adj['X']; p2 = m_adj['2']
        po25 = 1 - m_adj['u25']; pu25 = m_adj['u25']; pgg = m_adj['gg']; png = 1 - m_adj['gg']
        mercati_calcolati = {f"Vittoria {h}": p1, "Pareggio": pX, f"Vittoria {a}": p2, "Over 2.5": po25, "Under 2.5": pu25, "GG": pgg, "NG": png}
        mercato_top = max(mercati_calcolati, key=mercati_calcolati.get); prob_top = mercati_calcolati[mercato_top]
        score_probs = sorted([(i, j, poisson.pmf(i, h_exp) * poisson.pmf(j, a_exp)) for i in range(6) for j in range(6)], key=lambda x: -x[2])
        risultati_str = "\n".join([f"  {h} {g[0]}-{g[1]} {a}: {g[2]:.1%}" for g in score_probs[:6]])
        prompt = f"""Sei Billy Walters. Analizza {h} vs {a}.
PROBABILITA' MODELLO POISSON: 1: {p1:.0%} | X: {pX:.0%} | 2: {p2:.0%} | O2.5: {po25:.0%} | U2.5: {pu25:.0%} | GG: {pgg:.0%} | NG: {png:.0%}
Il mercato matematicamente superiore è "{mercato_top}" al {prob_top:.0%}.
PRONOSTICO SICURO: DEVI scrivere OBBLIGATORIAMENTE "{mercato_top} - prob {prob_top:.0%} - motivazione". NON scegliere altri mercati.
TOP 3 MERCATI ALTERNATIVI: I 3 mercati con prob più alta dopo "{mercato_top}". LIVELLO DI CONFIDENZA: 1-10."""
        try:
            res = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=800)
            testo = res.choices[0].message.content
            
            # --- RENDERING GRAFICO AVANZATO DELL'AI ---
            sezioni = testo.split("\n")
            sezione_corrente = ""
            html_out = ""
            
            for riga in righe:
                rs = riga.strip()
                if not rs: continue
                
                # Riconoscimento sezioni
                if "PRONOSTICO SICURO" in rs.upper():
                    sezione_corrente = "sicuro"
                    html_out += "<div class='ai-section'>"
                    html_out += "<div class='ai-section-title'>🎯 PRONOSTICO SICURO</div>"
                    
                    # Estrai mercato e probabilità
                    mercato_ai = mercato_top
                    prob_ai = f"{prob_top:.0%}"
                    if rs.lower().startswith("pronostico sicuro"): rs = rs[len("pronostico sicuro"):].strip()
                    if rs.startswith(":"): rs = rs[1:].strip()
                    if "-" in rs:
                        parti = rs.split("-")
                        mercato_ai = parti[0].strip()
                        prob_ai = parti[1].strip() if len(parti) > 1 else prob_ai
                    
                    # Scegli colore in base al mercato
                    mkt_lower = mercato_ai.lower()
                    color = "#3fb950" if "under" in mkt_lower or "ng" in mkt_lower else "#f85149" if "over" in mkt_lower or "gg" in mkt_lower else "#58a6ff"
                    
                    html_out += f"""
                    <div class='ai-pronostico' style='border-color: {color};'>
                        <div class='ai-pronostico-market' style='color: {color};'>{mercato_ai}</div>
                        <div class='ai-pronostico-prob'>{prob_ai}</div>
                    </div>"""
                    
                    # Motivazione (il testo dopo la probabilità)
                    if len(parti) > 2:
                        motivazione = "-".join(parti[2:]).strip()
                        html_out += f"<div class='ai-text'>{motivazione}</div>"
                    html_out += "</div>"
                    continue
                    
                elif "TOP 3" in rs.upper() or "TOP3" in rs.upper():
                    sezione_corrente = "top3"
                    html_out += "<div class='ai-section'><div class='ai-section-title'>📊 TOP 3 ALTERNATIVE</div>"
                    continue
                elif "LIVELLO DI CONFIDENZA" in rs.upper():
                    sezione_corrente = "confidenza"
                    html_out += "<div class='ai-section'><div class='ai-section-title'>💪 LIVELLO DI CONFIDENZA</div>"
                    continue
                elif any(s in rs.upper() for s in ["STATO DI FORMA", "ANALISI TATTICA", "RAGIONAMENTO", "RISULTATI ATTESI"]):
                    if sezione_corrente: html_out += "</div>"
                    sezione_corrente = "testo"
                    titolo = rs.upper().replace("*", "")
                    html_out += f"<div class='ai-section'><div class='ai-section-title'>{titolo}</div>"
                    continue
                    
                # Contenuto
                if sezione_corrente == "top3" and rs[0].isdigit():
                    html_out += f"<div style='background:{stat_bg}; padding:8px 12px; border-radius:8px; margin-bottom:4px; border-left: 3px solid #58a6ff;'>{rs}</div>"
                elif sezione_corrente == "confidenza":
                    voto_match = re.search(r'(\d+)(?:/\d+)?', rs)
                    if voto_match:
                        voto = int(voto_match.group(1))
                        colore_conf = "#3fb950" if voto >= 7 else "#d29922" if voto >= 5 else "#f85149"
                        html_out += f"<div style='text-align:center;'><span class='ai-confidence' style='color:{colore_conf};'>{voto}/10</span><br>{rs}</div>"
                    else:
                        html_out += f"<div class='ai-text'>{rs}</div>"
                else:
                    html_out += f"<div class='ai-text'>{rs}</div>"
                    
            if sezione_corrente: html_out += "</div>"

            # Forza sfondo bianco per il popup in light mode
            if theme == "LIGHT":
                st.markdown("""<style>div[data-testid="stDialog"] > div > div { background-color: #ffffff !important; color: #1a1d23 !important; }</style>""", unsafe_allow_html=True)
                
            st.markdown(html_out, unsafe_allow_html=True)

            # Salvataggio predizione
            pronostico_sicuro = ""; top3 = []; prob_sicuro = 0.0
            # (Logica di estrazione invariata)
            in_sicuro = False; in_top3 = False
            for riga in testo.split("\n"):
                rs = riga.strip()
                if rs.startswith("PRONOSTICO SICURO"): in_sicuro = True; in_top3 = False; continue
                if rs.startswith("TOP 3"): in_top3 = True; in_sicuro = False; continue
                if any(rs.startswith(s) for s in ["RISULTATI", "LIVELLO"]): in_sicuro = False; in_top3 = False
                if in_sicuro and rs and not pronostico_sicuro:
                    pronostico_sicuro = rs[:150]; m_prob = re.search(r"(\d+)\%", rs)
                    if m_prob: prob_sicuro = int(m_prob.group(1))
                if in_top3 and rs and rs[0].isdigit(): top3.append(rs[:120])
                
            match_date_str = ""; m_id = match_id
            if not m_id:
                for mx in st.session_state.get("live_data", []):
                    if clean_name(h) in clean_name(mx["homeTeam"].get("shortName", "") or mx["homeTeam"].get("name","")):
                        m_id = mx.get("id")
                        try: match_date_str = (datetime.strptime(mx["utcDate"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=2)).strftime("%d/%m/%Y %H:%M")
                        except: pass
                        break
            if pronostico_sicuro and m_id:
                giornata_p = st.session_state.get("live_data", [{}])[0].get("matchday", 0) if st.session_state.get("live_data") else 0
                save_prediction_entry(m_id, h, a, camp_sel, giornata_p, match_date_str, pronostico_sicuro, top3, prob_sicuro, risultati_str)
        except Exception as e: st.error(f"Errore AI: {e}")

# --- UI PRINCIPALE ---
st.markdown('<div class="maradona-header"><h1>M4 STRATEGIC TERMINAL</h1><p>Professional Betting Intelligence v39.0</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("🎩 Billy Walters")
    camp_sel = st.selectbox("CAMPIONATO", ["Serie A", "Premier League", "La Liga", "Bundesliga"])
    try:
        xg_check = get_understat_xg(camp_sel)
        if xg_check and len(xg_check) > 0: st.markdown(f"<div style='font-size:12px; color:#3fb950; font-weight:700;'>✅ xG caricati ({len(xg_check)} squadre)</div>", unsafe_allow_html=True)
        else: st.markdown("<div style='font-size:12px; color:#f85149; font-weight:700;'>⚠️ xG non disponibili</div>", unsafe_allow_html=True)
    except: pass
    
    camp_cached = st.session_state.get("live_camp", None)
    has_data = bool("live_data" in st.session_state and st.session_state.get("live_data") and camp_cached == camp_sel)
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1: do_sync = st.button("🔄 SINCRONIZZA", disabled=has_data)
    with col_s2: do_refresh = st.button("↺ Refresh")
    
    if do_sync or do_refresh:
        l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
        try:
            resp = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/matches", headers={'X-Auth-Token': API_KEY_DATA})
            st.session_state.live_data = resp.json().get('matches', [])
            st.session_state.live_camp = camp_sel
            try:
                stand_resp = requests.get(f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/standings", headers={"X-Auth-Token": API_KEY_DATA})
                classifica = {}
                for row in stand_resp.json().get("standings", [])[0].get("table", []):
                    nome = row["team"].get("shortName") or row["team"].get("name", "")
                    classifica[clean_name(nome)] = {"pos": row["position"], "punti": row["points"], "pg": row["playedGames"], "gf": row["goalsFor"], "gs": row["goalsAgainst"], "forma": row.get("form", "")}
                st.session_state.classifica = classifica
            except: pass
        except Exception as e: st.sidebar.error(f"Errore sync: {e}")
        
    if "live_data" in st.session_state and st.session_state.live_data:
        giornate = sorted(list(set([m['matchday'] for m in st.session_state.live_data])))
        default_idx = 0; now_utc = datetime.now(timezone.utc); min_diff = timedelta(days=999)
        for i, g in enumerate(giornate):
            for m in st.session_state.live_data:
                if m['matchday'] == g:
                    try:
                        match_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                        diff = abs(match_dt - now_utc)
                        if diff < min_diff: min_diff = diff; default_idx = i
                    except: pass
        g_sel = st.selectbox("GIORNATA", giornate, index=default_idx)
    else: g_sel = None

engine = get_league_engine(camp_sel)
tab1, tab2, tab3 = st.tabs(["🏟️ PARTITE", "🌟 TOP MIX", "📒 REGISTRO"])

with tab1:
 if 'live_data' in st.session_state and engine and g_sel is not None:
    team_stats, avg_h, avg_a, df_full = engine
    matches = [m for m in st.session_state.live_data if m['matchday'] == g_sel]
    col_title, col_btn = st.columns([4, 1])
    with col_title: st.subheader(f"🏟️ {camp_sel.upper()} - GIORNATA {g_sel}")
    with col_btn:
        if st.button("⚡ Analisi Rapida"):
            with st.spinner("Calcolo..."): n = analisi_rapida_giornata(matches, team_stats, avg_h, avg_a, camp_sel, st.session_state.get("classifica", {}), g_sel)
            st.success(f"✅ {n} partite analizzate e salvate!")
            
    for idx, match in enumerate(matches):
        h_api = match['homeTeam'].get('shortName') or match['homeTeam'].get('name', '?'); a_api = match['awayTeam'].get('shortName') or match['awayTeam'].get('name', '?')
