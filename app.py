import streamlit as st
import pandas as pd
import numpy as np
import os, requests, glob, re
from scipy.stats import poisson
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
from groq import Groq
from scraper_xg import get_understat_xg, get_market_values

# --- CONFIGURAZIONE CHIAVI (CLOUD SECRETS) ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
API_KEY_ODDS = "a310fd7b74f24f2736a57c6caf768118"
API_KEY_DATA = "c299e4a676a54d48a642f20bca7f4480"

# Inizializzazione AI
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    groq_client = None

st.set_page_config(page_title="M4 STRATEGIC TERMINAL", layout="wide", initial_sidebar_state="expanded")

# --- GESTIONE TEMA ---
theme = st.sidebar.select_slider("⚙️ TEMA", options=["LIGHT", "DARK"], value="DARK")
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
    .val-sign {{ color: {txt} !important; font-weight: 800; font-size: 13px; margin-bottom: 2px; }}
    .val-p-green {{ color: #28a745; font-size: 17px; font-weight: 800; }}
    .val-p-red {{ color: #dc3545; font-size: 17px; font-weight: 800; }}
    .val-q {{ color: #856404; font-size: 14px; font-weight: 700; display: block; margin-top: 3px; font-family: monospace; }}
    [data-testid="stVerticalBlock"] > div {{ gap: 0rem !important; }}
    </style>
    """, unsafe_allow_html=True)

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
    p = {"Serie A": "SerieA", "Premier League": "Premier", "La Liga": "LaLiga", "Bundesliga": "Bundesliga", "Ligue 1": "Ligue1"}
    prefix = p.get(camp_key)
    if not prefix: return None

    # Carica i file storici e il file Live separatamente
    files_storici = sorted(glob.glob(f"./database/{prefix}_20*.csv"))
    files_live = glob.glob(f"./database/{prefix}_Live.csv")

    if not files_storici and not files_live: return None

    dfs = []

    # Stagioni storiche: peso 1x (base)
    for f in files_storici:
        try:
            df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            df_tmp['peso'] = 1.0
            dfs.append(df_tmp)
        except: pass

    # Stagione corrente (Live): peso 4x - conta molto di piu'
    for f in files_live:
        try:
            df_tmp = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            df_tmp['peso'] = 4.0
            # Duplica le righe per simulare il peso maggiore
            dfs.extend([df_tmp] * 4)
        except: pass

    if not dfs: return None

    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date')
    df['HomeClean'] = df['HomeTeam'].apply(clean_name)
    df['AwayClean'] = df['AwayTeam'].apply(clean_name)

    xg_data = get_understat_xg(camp_key)
    mkt_values = get_market_values()
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()

    stats = {}
    for t in pd.concat([df['HomeClean'], df['AwayClean']]).unique():
        h_h = df[df['HomeClean']==t]
        a_h = df[df['AwayClean']==t]

        # Se disponibili, usa xG dalla stagione corrente
        if xg_data and t in xg_data:
            att, defe = xg_data[t]['xG_avg'], xg_data[t]['xGA_avg']
        else:
            att_h = h_h['FTHG'].mean() / avg_h if not h_h.empty else 1.0
            att_a = a_h['FTAG'].mean() / avg_a if not a_h.empty else 1.0
            def_h = h_h['FTAG'].mean() / avg_a if not h_h.empty else 1.0
            def_a = a_h['FTHG'].mean() / avg_h if not a_h.empty else 1.0
            att = (att_h + att_a) / 2
            defe = (def_h + def_a) / 2

        val = mkt_values.get(t, 50)
        stats[t] = {'att': att * (1 + (val/6000)), 'def': defe * (1 - (val/6000)), 'val': val}

    return stats, avg_h, avg_a, df

def get_full_poisson(h_e, a_e):
    h_p = [poisson.pmf(i, h_e) for i in range(8)]
    a_p = [poisson.pmf(i, a_e) for i in range(8)]
    matrix = np.outer(h_p, a_p)
    def get_u(limit): return sum([matrix[i,j] for i in range(8) for j in range(8) if i+j < limit])
    return {"1": np.sum(np.tril(matrix, -1)), "X": np.sum(np.diag(matrix)), "2": np.sum(np.triu(matrix, 1)),
            "u15": get_u(1.5), "u25": get_u(2.5), "u35": get_u(3.5), "gg": (1-h_p[0])*(1-a_p[0])}


# --- STRATO INTERMEDIO: SEGNALI NUMERICI CONTESTUALI ---
def calcola_segnali(risultati, infraset_giocate, infraset_programmate, stand):
    """
    Calcola moltiplicatori da applicare ai gol attesi prima del Poisson.
    Ritorna (mult_att, mult_def, nota) dove:
      mult_att: corregge i gol attesi in attacco (>1 = squadra piu' in forma)
      mult_def: corregge i gol concessi (>1 = difesa piu' permeabile)
      nota: stringa con i segnali attivi per il prompt
    """
    mult_att = 1.0
    mult_def = 1.0
    note = []

    # --- FORMA RECENTE (ultimi 5 risultati) ---
    # Ogni risultato e' tipo "Roma 2-1 Juve (V)" o "(P)" o "(X)"
    score_forma = 0
    n_ris = 0
    for r in risultati:
        if r.endswith("(V)"):
            score_forma += 1; n_ris += 1
        elif r.endswith("(P)"):
            score_forma -= 1; n_ris += 1
        elif r.endswith("(X)"):
            n_ris += 1
    if n_ris > 0:
        # Normalizza: da [-1,+1] a moltiplicatore [0.92, 1.08]
        forma_norm = score_forma / n_ris  # tra -1 e +1
        delta_att = forma_norm * 0.08
        delta_def = -forma_norm * 0.05  # buona forma = difesa piu' solida
        mult_att += delta_att
        mult_def += delta_def
        if forma_norm > 0.3:
            note.append(f"forma positiva (+{score_forma}/{n_ris}): att +{delta_att:.0%}")
        elif forma_norm < -0.3:
            note.append(f"forma negativa ({score_forma}/{n_ris}): att {delta_att:.0%}")

    # --- STANCHEZZA INFRASETTIMANALE ---
    # Partita giocata = stanchezza accumulata
    if infraset_giocate:
        mult_att -= 0.04   # meno lucidita' offensiva
        mult_def += 0.06   # difesa piu' permeabile per stanchezza
        note.append(f"stanchezza ({len(infraset_giocate)} partita/e infraset.): att -4%, def +6%")
    # Partita in programma prima della gara = possibile turnover
    if infraset_programmate:
        mult_att -= 0.02
        note.append(f"impegno infraset. in programma: possibile turnover att -2%")

    # --- POSIZIONE IN CLASSIFICA ---
    if stand:
        pos = stand.get("pos", 10)
        pg = stand.get("pg", 1)
        gf = stand.get("gf", 0)
        gs = stand.get("gs", 1)
        # Rapporto gol: squadre che segnano molto tendono a segnare ancora
        ratio_gol = (gf / pg) if pg > 0 else 1.0
        if ratio_gol > 1.8:
            mult_att += 0.03
            note.append(f"prolifica ({gf} GF in {pg} pg): att +3%")
        elif ratio_gol < 0.9:
            mult_att -= 0.03
            note.append(f"poco prolifica ({gf} GF in {pg} pg): att -3%")
        # Difesa: squadre con pochi gol subiti
        ratio_gs = (gs / pg) if pg > 0 else 1.0
        if ratio_gs < 0.8:
            mult_def -= 0.04
            note.append(f"difesa solida ({gs} GS in {pg} pg): def -4%")
        elif ratio_gs > 1.5:
            mult_def += 0.04
            note.append(f"difesa permeabile ({gs} GS in {pg} pg): def +4%")

    # Clamp per sicurezza
    mult_att = max(0.80, min(1.20, mult_att))
    mult_def = max(0.80, min(1.20, mult_def))

    nota_str = " | ".join(note) if note else "nessun segnale contestuale significativo"
    return mult_att, mult_def, nota_str


# --- DATI CONTESTUALI DA FOOTBALL-DATA.ORG ---
def get_team_fd_id(team_name, camp_sel):
    """Trova l'ID squadra su football-data.org dalla lista partite in sessione"""
    matches = st.session_state.get("live_data", [])
    for match in matches:
        h = match["homeTeam"]
        a = match["awayTeam"]
        for team in [h, a]:
            nome = team.get("shortName", "") or team.get("name", "")
            if clean_name(nome).lower() == clean_name(team_name).lower():
                return team["id"]
    return None

def get_ultimi_risultati_fd(team_id, camp_sel, n=5):
    """Ultime N partite giocate da football-data.org"""
    l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1", "Ligue 1": "FL1"}
    comp = l_map.get(camp_sel, "SA")
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/teams/{team_id}/matches",
            headers={"X-Auth-Token": API_KEY_DATA},
            params={"status": "FINISHED", "limit": 15, "competitions": comp}
        )
        risultati = []
        for match in r.json().get("matches", [])[-n:]:
            home = match["homeTeam"].get("shortName") or match["homeTeam"].get("name", "?")
            away = match["awayTeam"].get("shortName") or match["awayTeam"].get("name", "?")
            gh = match["score"]["fullTime"]["home"]
            ga = match["score"]["fullTime"]["away"]
            winner = match["score"]["winner"]
            if match["homeTeam"]["id"] == team_id:
                esito = "V" if winner == "HOME_TEAM" else ("X" if winner == "DRAW" else "P")
            else:
                esito = "V" if winner == "AWAY_TEAM" else ("X" if winner == "DRAW" else "P")
            risultati.append(f"{home} {gh}-{ga} {away} ({esito})")
        return risultati
    except:
        return []

def get_contesto_partita(h, a, camp_sel):
    """Recupera ultime 5 partite + news infortunati"""
    h_id = get_team_fd_id(h, camp_sel)
    a_id = get_team_fd_id(a, camp_sel)

    contesto = {"h_risultati": [], "a_risultati": [], "h_infortunati": [], "a_infortunati": []}

    if h_id:
        contesto["h_risultati"] = get_ultimi_risultati_fd(h_id, camp_sel)
    if a_id:
        contesto["a_risultati"] = get_ultimi_risultati_fd(a_id, camp_sel)

    # Infortunati + partite infrasettimanali via DuckDuckGo
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"{h} infortunati squalificati {camp_sel} 2026", max_results=2):
                contesto["h_infortunati"].append(r["body"][:300])
            for r in ddgs.text(f"{a} infortunati squalificati {camp_sel} 2026", max_results=2):
                contesto["a_infortunati"].append(r["body"][:300])
    except:
        pass

    # Partite infrasettimanali (Champions, Europa League, Coppa Italia ecc.)
    contesto["h_infraset"] = []
    contesto["a_infraset"] = []
    from datetime import datetime, timedelta, timezone
    l_map2 = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
    camp_code = l_map2.get(camp_sel, "SA")

    # Trova la data della partita da analizzare dalla sessione
    match_date = None
    for mx in st.session_state.get("live_data", []):
        hn = clean_name(mx["homeTeam"].get("shortName") or mx["homeTeam"].get("name", ""))
        an = clean_name(mx["awayTeam"].get("shortName") or mx["awayTeam"].get("name", ""))
        if clean_name(h) in hn or hn in clean_name(h):
            try:
                match_date = datetime.fromisoformat(mx["utcDate"].replace("Z", "+00:00"))
            except:
                pass
            break

    # Se non trovata, usa ora come fallback
    now_utc = datetime.now(timezone.utc)
    if not match_date:
        match_date = now_utc

    # Finestra: dall'ultima giornata campionato (7 giorni prima della partita) fino alla partita
    window_start = match_date - timedelta(days=7)

    def get_infraset(team_id):
        giocate = []
        programmate = []
        try:
            # Partite finite in altre competizioni nella finestra
            r_fin = requests.get(
                f"https://api.football-data.org/v4/teams/{team_id}/matches",
                headers={"X-Auth-Token": API_KEY_DATA},
                params={"status": "FINISHED", "limit": 10}
            )
            for match in r_fin.json().get("matches", []):
                comp_code = match.get("competition", {}).get("code", "")
                comp_name = match.get("competition", {}).get("name", "")
                if comp_code == camp_code:
                    continue
                utc_date = match.get("utcDate", "")
                try:
                    match_dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                except:
                    continue
                if match_dt < window_start or match_dt >= match_date:
                    continue
                gh = match["score"]["fullTime"].get("home")
                ga = match["score"]["fullTime"].get("away")
                if gh is None or ga is None:
                    continue
                home = match["homeTeam"].get("shortName") or match["homeTeam"].get("name", "?")
                away = match["awayTeam"].get("shortName") or match["awayTeam"].get("name", "?")
                giocate.append(f"{match_dt.strftime('%d/%m')} {comp_name}: {home} {gh}-{ga} {away}")

            # Partite programmate in altre competizioni PRIMA della partita (impegno futuro)
            r_prg = requests.get(
                f"https://api.football-data.org/v4/teams/{team_id}/matches",
                headers={"X-Auth-Token": API_KEY_DATA},
                params={"status": "SCHEDULED,TIMED", "limit": 5}
            )
            for match in r_prg.json().get("matches", []):
                comp_code = match.get("competition", {}).get("code", "")
                comp_name = match.get("competition", {}).get("name", "")
                if comp_code == camp_code:
                    continue
                utc_date = match.get("utcDate", "")
                try:
                    match_dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                except:
                    continue
                if match_dt >= match_date or match_dt <= now_utc:
                    continue
                home = match["homeTeam"].get("shortName") or match["homeTeam"].get("name", "?")
                away = match["awayTeam"].get("shortName") or match["awayTeam"].get("name", "?")
                programmate.append(f"[IN PROGRAMMA {match_dt.strftime('%d/%m')}] {comp_name}: {home} vs {away}")
        except:
            pass
        return giocate, programmate

    if h_id:
        h_giot, h_prog = get_infraset(h_id)
        contesto["h_infraset"] = h_giot
        contesto["h_infraset_prog"] = h_prog
    else:
        contesto["h_infraset"] = []
        contesto["h_infraset_prog"] = []
    if a_id:
        a_giot, a_prog = get_infraset(a_id)
        contesto["a_infraset"] = a_giot
        contesto["a_infraset_prog"] = a_prog
    else:
        contesto["a_infraset"] = []
        contesto["a_infraset_prog"] = []

    return contesto

# --- POPUP AI ---
@st.dialog("STRATEGIC ANALYSIS", width="large")
def show_details(h, a, m, camp_sel="Serie A"):
    if not groq_client:
        st.error("Billy non e' configurato correttamente. Aggiungi GROQ_API_KEY nei secrets.")
        return
    with st.spinner("Billy Walters sta analizzando..."):
        # Dati strutturati da API-Football
        contesto = get_contesto_partita(h, a, camp_sel)

        # Fallback DuckDuckGo solo se API-Football non disponibile
        news_extra = ""
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(f"{h} {a} probabili formazioni 2026", max_results=3):
                    news_extra += f" {r['body']}"
        except:
            news_extra = ""

        # --- STRATO INTERMEDIO: correggi gol attesi con segnali contestuali ---
        classifica_sess = st.session_state.get("classifica", {})
        h_cl_key = clean_name(h)
        a_cl_key = clean_name(a)
        h_stand_s = classifica_sess.get(h_cl_key, {})
        a_stand_s = classifica_sess.get(a_cl_key, {})

        h_mult_att, h_mult_def, h_note_seg = calcola_segnali(
            contesto.get("h_risultati", []) if contesto else [],
            contesto.get("h_infraset", []) if contesto else [],
            contesto.get("h_infraset_prog", []) if contesto else [],
            h_stand_s
        )
        a_mult_att, a_mult_def, a_note_seg = calcola_segnali(
            contesto.get("a_risultati", []) if contesto else [],
            contesto.get("a_infraset", []) if contesto else [],
            contesto.get("a_infraset_prog", []) if contesto else [],
            a_stand_s
        )

        # Recupera stats squadre dall'engine
        engine_data = get_league_engine(camp_sel)
        if engine_data:
            team_stats_p, avg_h_p, avg_a_p, _ = engine_data
            h_s_p = team_stats_p.get(h_cl_key, {"att": 0.85, "def": 1.15})
            a_s_p = team_stats_p.get(a_cl_key, {"att": 0.85, "def": 1.15})
            # Applica moltiplicatori segnali ai gol attesi
            h_exp = h_s_p["att"] * h_mult_att * a_s_p["def"] * a_mult_def * avg_h_p
            a_exp = a_s_p["att"] * a_mult_att * h_s_p["def"] * h_mult_def * avg_a_p
        else:
            h_exp = 1.3
            a_exp = 1.1

        # Ricalcola Poisson con gol attesi corretti
        m_adj = get_full_poisson(h_exp, a_exp)

        p1 = m_adj['1']
        pX = m_adj['X']
        p2 = m_adj['2']
        po25 = 1 - m_adj['u25']
        po15 = 1 - m_adj['u15']
        pgg = m_adj['gg']

        # Verifica se xG sono disponibili
        xg_data_check = get_understat_xg(camp_sel)
        xg_status = f"xG attivi ({len(xg_data_check)} squadre)" if xg_data_check else "xG non disponibili - uso medie storiche"

        segnali_str = f"""
SEGNALI CONTESTUALI APPLICATI AL MODELLO:
- Fonte dati offensivi: {xg_status}
- {h}: {h_note_seg} → gol attesi {h_exp:.2f}
- {a}: {a_note_seg} → gol attesi {a_exp:.2f}"""

        # Calcola top 6 risultati più probabili dalla matrice Poisson
        h_p = [poisson.pmf(i, h_exp) for i in range(8)]
        a_p = [poisson.pmf(i, a_exp) for i in range(8)]
        score_probs = []
        for i in range(6):
            for j in range(6):
                score_probs.append((i, j, h_p[i] * a_p[j]))
        score_probs.sort(key=lambda x: x[2], reverse=True)
        top_scores = score_probs[:6]
        risultati_str = "\n".join([f"  {h} {g[0]}-{g[1]} {a}: {g[2]:.1%}" for g in top_scores])

        # Quote bookmaker dalla sessione
        q1, qX, q2 = 0.0, 0.0, 0.0
        qo25, qu25, qgg, qng = 0.0, 0.0, 0.0, 0.0
        h_cl = clean_name(h)

        # 1X2
        if "live_odds" in st.session_state and isinstance(st.session_state.live_odds, list):
            for mo in st.session_state.live_odds:
                if h_cl in clean_name(mo.get("home_team", "")):
                    try:
                        odds = {o["name"]: o["price"] for o in mo["bookmakers"][0]["markets"][0]["outcomes"]}
                        q1 = odds.get(mo["home_team"], 0.0)
                        qX = odds.get("Draw", odds.get("Tie", 0.0))
                        q2 = odds.get(mo["away_team"], 0.0)
                    except: pass

        # Over/Under 2.5
        if "live_odds_totals" in st.session_state and isinstance(st.session_state.live_odds_totals, list):
            for mo in st.session_state.live_odds_totals:
                if h_cl in clean_name(mo.get("home_team", "")):
                    try:
                        for bk in mo.get("bookmakers", []):
                            for mkt in bk.get("markets", []):
                                outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                                if "Over" in outcomes:
                                    qo25 = outcomes.get("Over", 0.0)
                                    qu25 = outcomes.get("Under", 0.0)
                                    break
                            if qo25 > 0: break
                    except: pass

        # GG/NG - mercato btts non disponibile nel piano attuale, skip

        # Calcolo value bet reale
        def value(prob, quota):
            if quota <= 0: return None
            return round((prob * quota - 1) * 100, 1)

        def fmt_value(v):
            if v is None: return "n/d"
            return f"{v:+.1f}%"

        v1   = value(p1,   q1)
        vX   = value(pX,   qX)
        v2   = value(p2,   q2)
        vo25 = value(po25, qo25)
        vu25 = value(1-po25, qu25)
        vgg  = value(pgg,  qgg)
        vng  = value(1-pgg, qng)

        quote_str = f"""
QUOTE BOOKMAKER (solo riferimento):
- 1 ({h}): {q1 or "n/d"} | X: {qX or "n/d"} | 2 ({a}): {q2 or "n/d"}
- Over 2.5: {qo25 or "n/d"}"""

        # Classifica
        classifica = st.session_state.get("classifica", {})
        h_cl_name = clean_name(h)
        a_cl_name = clean_name(a)
        h_stand = classifica.get(h_cl_name, {})
        a_stand = classifica.get(a_cl_name, {})
        if h_stand and a_stand:
            class_str = f"""
CLASSIFICA ATTUALE:
- {h}: {h_stand['pos']}° posto | {h_stand['punti']} punti | {h_stand['pg']} partite | GF {h_stand['gf']} GS {h_stand['gs']} | Forma: {h_stand['forma']}
- {a}: {a_stand['pos']}° posto | {a_stand['punti']} punti | {a_stand['pg']} partite | GF {a_stand['gf']} GS {a_stand['gs']} | Forma: {a_stand['forma']}"""
        else:
            class_str = ""

        # Risultati recenti, infortunati e infrasettimanali
        if contesto:
            h_ris_list = contesto["h_risultati"] if contesto["h_risultati"] else []
            a_ris_list = contesto["a_risultati"] if contesto["a_risultati"] else []
            h_ris = f"({len(h_ris_list)} partite) " + ", ".join(h_ris_list) if h_ris_list else "Non disponibili"
            a_ris = f"({len(a_ris_list)} partite) " + ", ".join(a_ris_list) if a_ris_list else "Non disponibili"
            h_inf = " | ".join(contesto["h_infortunati"]) if contesto["h_infortunati"] else "Nessuna info"
            a_inf = " | ".join(contesto["a_infortunati"]) if contesto["a_infortunati"] else "Nessuna info"
            h_infra = ", ".join(contesto.get("h_infraset", [])) or "Nessuna"
            a_infra = ", ".join(contesto.get("a_infraset", [])) or "Nessuna"
            h_prog = ", ".join(contesto.get("h_infraset_prog", [])) or "Nessuna"
            a_prog = ", ".join(contesto.get("a_infraset_prog", [])) or "Nessuna"
            dati_reali = f"""
RISULTATI RECENTI IN CAMPIONATO:
- Ultime 5 partite {h}: {h_ris}
- Ultime 5 partite {a}: {a_ris}
PARTITE INFRASETTIMANALI GIOCATE (tra ultima giornata e questa partita):
- {h}: {h_infra}
- {a}: {a_infra}
IMPEGNI INFRASETTIMANALI IN PROGRAMMA (prima di questa partita):
- {h}: {h_prog}
- {a}: {a_prog}
NOTIZIE INDISPONIBILI:
- {h}: {h_inf}
- {a}: {a_inf}"""
        else:
            dati_reali = f"CONTESTO WEB: {news_extra}"

        prompt = f"""Sei Billy Walters, il leggendario analista sportivo con 40 anni di esperienza. Analizza {h} vs {a}.

{class_str}
{dati_reali}
{segnali_str}

PROBABILITA' MODELLO POISSON (corrette per forma, stanchezza, xG):
- Vittoria {h}: {p1:.0%} | Pareggio: {pX:.0%} | Vittoria {a}: {p2:.0%}
- Over 2.5: {po25:.0%} | Under 2.5: {1-po25:.0%} | GG: {pgg:.0%} | NG: {1-pgg:.0%}

RISULTATI PIU' PROBABILI (modello Poisson, top 6):
{risultati_str}

{quote_str}

REGOLE:
- La stanchezza PENALIZZA la squadra che ha giocato infrasettimanale.
- Over 2.5 = 3+ gol totali (es. 2-1, 3-0). GG = entrambe segnano (es. 1-1, 2-1). Sono mercati diversi.
- NON menzionare mai Over 1.5.
- Indica sempre il numero esatto di partite disponibili nei risultati recenti.
- PRONOSTICO SICURO: prob >= 45% obbligatoria. Se nessun mercato supera il 45%, scrivi "Partita troppo equilibrata - nessun pronostico sicuro".

STRUTTURA (italiano, tono diretto, senza fronzoli):

STATO DI FORMA
Risultati recenti, trend, classifica, impatto infrasettimanali.

ANALISI TATTICA
GF/GS stagionali, indisponibili e impatto concreto sul match.

RAGIONAMENTO
Dai dati al pronostico. Le probabilita' Poisson confermano o contraddicono i dati reali?

RISULTATI ATTESI
Elenca i 3 risultati piu' probabili con la loro percentuale. Per ciascuno scrivi i mercati implicati.
Es: "1-0 (12%) → vittoria {h}, NG, Under 2.5"

PRONOSTICO SICURO
Mercato piu' probabile con prob >= 45%. Formato: "Mercato - prob X% - motivazione"

TOP 3 MERCATI ALTERNATIVI
I 3 mercati successivi per probabilita'. Formato: "N. Mercato - prob X% - motivazione"

LIVELLO DI CONFIDENZA
Voto 1-10 con motivazione breve.

IMPORTANTE: Usa SOLO i dati forniti."""

        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200
            )
            testo = res.choices[0].message.content.replace("**", "").replace("*", "")
            # Rendering con sezioni in evidenza
            righe = testo.split("\n")
            html = ""
            sezioni = ["STATO DI FORMA", "ANALISI TATTICA", "RAGIONAMENTO", "RISULTATI ATTESI",
                       "PRONOSTICO SICURO", "TOP 3 MERCATI", "LIVELLO DI CONFIDENZA"]
            for riga in righe:
                riga_strip = riga.strip()
                if any(riga_strip.startswith(s) for s in sezioni):
                    html += f"<div style='margin-top:16px; margin-bottom:4px; font-size:13px; font-weight:900; text-transform:uppercase; color:#3b82f6; letter-spacing:1px;'>{riga_strip}</div>"
                elif riga_strip:
                    html += f"<div style='font-size:15px; line-height:1.6; color:#1a1a1a; margin-bottom:4px;'>{riga_strip}</div>"
            # CSS per forzare popup bianco
            st.markdown("""<style>
            div[data-testid="stDialog"] > div > div {
                background-color: #ffffff !important;
                color: #1a1a1a !important;
            }
            div[data-testid="stDialog"] p,
            div[data-testid="stDialog"] div {
                color: #1a1a1a !important;
            }
            </style>""", unsafe_allow_html=True)
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Errore AI: {e}")

# --- UI PRINCIPALE ---
st.markdown('<div class="maradona-header"><h1>M4 STRATEGIC TERMINAL</h1><p>Intelligence Evolution v28.1</p></div>', unsafe_allow_html=True)

# Mapping odds sport key
map_odds = {
    "Serie A": "soccer_italy_serie_a",
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga"
}

with st.sidebar:
    st.title("🎩 Billy Walters Chat")
    camp_sel = st.selectbox("CAMPIONATO", ["Serie A", "Premier League", "La Liga", "Bundesliga"])

    # Status xG
    try:
        xg_check = get_understat_xg(camp_sel)
        if xg_check and len(xg_check) > 0:
            st.markdown(f"<div style='font-size:12px; color:#28a745; font-weight:700;'>✅ xG caricati ({len(xg_check)} squadre)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:12px; color:#dc3545; font-weight:700;'>⚠️ xG non disponibili — uso medie gol storiche</div>", unsafe_allow_html=True)
    except Exception as xg_err:
        st.markdown(f"<div style='font-size:12px; color:#dc3545; font-weight:700;'>⚠️ xG errore: {str(xg_err)[:60]}</div>", unsafe_allow_html=True)

    # Sincronizza solo se non ci sono dati per questo campionato, oppure se l'utente preme Refresh
    camp_cached = st.session_state.get("live_camp", None)
    has_data = "live_data" in st.session_state and st.session_state.live_data and camp_cached == camp_sel

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        do_sync = st.button("🔄 SINCRONIZZA TURNO", disabled=has_data)
    with col_s2:
        do_refresh = st.button("↺ Refresh", help="Forza nuovo aggiornamento quote e partite")

    if do_sync or do_refresh:
        l_map = {"Serie A": "SA", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1"}
        try:
            resp = requests.get(
                f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/matches?status=TIMED,SCHEDULED",
                headers={'X-Auth-Token': API_KEY_DATA}
            )
            st.session_state.live_data = resp.json().get('matches', [])
            st.session_state.live_camp = camp_sel

            # Classifica aggiornata
            try:
                stand_resp = requests.get(
                    f"https://api.football-data.org/v4/competitions/{l_map[camp_sel]}/standings",
                    headers={"X-Auth-Token": API_KEY_DATA}
                )
                standings_raw = stand_resp.json().get("standings", [])
                classifica = {}
                for group in standings_raw:
                    for row in group.get("table", []):
                        nome = clean_name(row["team"].get("shortName") or row["team"].get("name", ""))
                        classifica[nome] = {
                            "pos": row["position"],
                            "punti": row["points"],
                            "pg": row["playedGames"],
                            "gf": row["goalsFor"],
                            "gs": row["goalsAgainst"],
                            "forma": row.get("form", "")
                        }
                st.session_state.classifica = classifica
            except:
                st.session_state.classifica = {}

            # Odds - scarica h2h, totals (over/under) e btts (GG/NG)
            sport_key = map_odds[camp_sel]
            try:
                odds_h2h = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
                    params={"apiKey": API_KEY_ODDS, "regions": "eu", "markets": "h2h"}
                ).json()
                odds_totals = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
                    params={"apiKey": API_KEY_ODDS, "regions": "eu", "markets": "totals"}
                ).json()
                st.session_state.live_odds = odds_h2h if isinstance(odds_h2h, list) else []
                st.session_state.live_odds_totals = odds_totals if isinstance(odds_totals, list) else []
            except:
                st.session_state.live_odds = []
                st.session_state.live_odds_totals = []
        except Exception as e:
            st.sidebar.error(f"Errore sincronizzazione: {e}")

    if "live_data" in st.session_state and st.session_state.live_data:
        giornate = sorted(list(set([m['matchday'] for m in st.session_state.live_data])))
        g_sel = st.selectbox("GIORNATA", giornate)
    else:
        g_sel = None

engine = get_league_engine(camp_sel)

if 'live_data' in st.session_state and engine and g_sel is not None:
    team_stats, avg_h, avg_a, df_full = engine
    matches = [m for m in st.session_state.live_data if m['matchday'] == g_sel]
    st.subheader(f"🏟️ {camp_sel.upper()} - GIORNATA {g_sel}")

    for idx, match in enumerate(matches):
        h_api = match['homeTeam'].get('shortName') or match['homeTeam'].get('name', '?')
        a_api = match['awayTeam'].get('shortName') or match['awayTeam'].get('name', '?')
        h_cl, a_cl = clean_name(h_api), clean_name(a_api)
        dt = (datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=1)).strftime("%d/%m | %H:%M")

        q1, qX, q2 = 0.0, 0.0, 0.0
        qo25_card, qgg_card = 0.0, 0.0

        if 'live_odds' in st.session_state and isinstance(st.session_state.live_odds, list):
            for mo in st.session_state.live_odds:
                if h_cl in clean_name(mo.get('home_team', '')):
                    try:
                        odds = {o['name']: o['price'] for o in mo['bookmakers'][0]['markets'][0]['outcomes']}
                        q1 = odds.get(mo['home_team'], 0.0)
                        qX = odds.get('Draw', odds.get('Tie', 0.0))
                        q2 = odds.get(mo['away_team'], 0.0)
                    except: pass

        if 'live_odds_totals' in st.session_state and isinstance(st.session_state.live_odds_totals, list):
            for mo in st.session_state.live_odds_totals:
                if h_cl in clean_name(mo.get('home_team', '')):
                    try:
                        for bk in mo.get('bookmakers', []):
                            for mkt in bk.get('markets', []):
                                outs = {o['name']: o['price'] for o in mkt.get('outcomes', [])}
                                if 'Over' in outs:
                                    qo25_card = outs.get('Over', 0.0)
                                    break
                            if qo25_card > 0: break
                    except: pass

        # btts non disponibile nel piano attuale

        h_s = team_stats.get(h_cl, {'att': 0.85, 'def': 1.15})
        a_s = team_stats.get(a_cl, {'att': 0.85, 'def': 1.15})
        # Gol attesi base
        h_exp_base = h_s['att'] * a_s['def'] * avg_h
        a_exp_base = a_s['att'] * h_s['def'] * avg_a
        # Nessun contesto disponibile nella card - Poisson puro
        m = get_full_poisson(h_exp_base, a_exp_base)

        with st.container():
            st.markdown('<div class="match-card">', unsafe_allow_html=True)
            c_h, c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.2, 0.8, 0.8, 0.8, 1, 0.4])
            with c_h:
                st.markdown(f"<span class='team-name'>{h_api}<br>{a_api}</span><br><span class='match-date'>🕒 {dt}</span>", unsafe_allow_html=True)
            with c1:
                st.markdown(f"""<div class='stat-container'><span class='label-header'>Esito 1X2</span>
                <div style='display:flex; justify-content:space-around'>
                    <div><span class='val-sign'>1</span><br><span class='val-p-green'>{m['1']:.0%}</span><br><span class='val-q'>{q1}</span></div>
                    <div><span class='val-sign'>X</span><br><span class='val-p-green'>{m['X']:.0%}</span><br><span class='val-q'>{qX}</span></div>
                    <div><span class='val-sign'>2</span><br><span class='val-p-green'>{m['2']:.0%}</span><br><span class='val-q'>{q2}</span></div>
                </div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 1.5</span><span class='val-p-red'>{m['u15']:.0%}</span> / <span class='val-p-green'>{(1-m['u15']):.0%}</span></div>", unsafe_allow_html=True)
            with c3:
                q_o25_str = f"<br><span class='val-q'>Over: {qo25_card}</span>" if qo25_card > 0 else ""
                st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 2.5</span><span class='val-p-red'>{m['u25']:.0%}</span> / <span class='val-p-green'>{(1-m['u25']):.0%}</span>{q_o25_str}</div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='stat-container'><span class='label-header'>U/O 3.5</span><span class='val-p-red'>{m['u35']:.0%}</span> / <span class='val-p-green'>{(1-m['u35']):.0%}</span></div>", unsafe_allow_html=True)
            with c5:
                st.markdown(f"<div class='stat-container'><span class='label-header'>GG / NG</span><span class='val-p-green'>{m['gg']:.0%}</span> / <span class='val-p-red'>{(1-m['gg']):.0%}</span></div>", unsafe_allow_html=True)
            with c6:
                st.write("<br>", unsafe_allow_html=True)
                st.button("🔍", key=f"ex_{idx}", on_click=show_details, args=(h_api, a_api, m, camp_sel))
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("👋 Terminale Pronto. Sincronizza per caricare la giornata.")
