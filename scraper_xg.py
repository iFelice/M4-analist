import requests
import pandas as pd
from io import StringIO
import json
from bs4 import BeautifulSoup

# Headers anti-ban
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/"
}

# Mappatura nomi squadra FBref → nomi usati nell'app
FBREF_NAME_MAP = {
    # Serie A
    "Inter Milan": "Inter",
    "Milan": "Milan",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Atalanta": "Atalanta",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Monza": "Monza",
    "Genoa": "Genoa",
    "Lecce": "Lecce",
    "Hellas Verona": "Verona",
    "Udinese": "Udinese",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Parma": "Parma",
    "Como": "Como",
    "Venezia": "Venezia",
    # Premier League
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Tottenham": "Tottenham",
    "Newcastle Utd": "Newcastle",
    "Aston Villa": "Aston Villa",
    "West Ham": "West Ham",
    "Brighton": "Brighton",
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atlético Madrid": "Atletico Madrid",
    "Athletic Club": "Athletic Club",
    "Villarreal": "Villarreal",
    # Bundesliga
    "Bayern Munich": "Bayern",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "Leipzig",
    "Eintracht Frankfurt": "Frankfurt",
}

# URL FBref per campionato 2025-26
FBREF_URLS = {
    "Serie A":        "https://fbref.com/en/comps/11/2025-2026/stats/2025-2026-Serie-A-Stats",
    "Premier League": "https://fbref.com/en/comps/9/2025-2026/stats/2025-2026-Premier-League-Stats",
    "La Liga":        "https://fbref.com/en/comps/12/2025-2026/stats/2025-2026-La-Liga-Stats",
    "Bundesliga":     "https://fbref.com/en/comps/20/2025-2026/stats/2025-2026-Bundesliga-Stats",
}


def _parse_fbref_xg(league_name):
    """
    Legge la tabella xG da FBref per squadra.
    FBref espone una tabella 'Squad Standard Stats' con colonne xG e xGA.
    Ritorna {nome_squadra: {xG_avg, xGA_avg}} o None se fallisce.
    """
    url = FBREF_URLS.get(league_name)
    if not url:
        return None

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        # FBref usa header multi-livello — leggiamo tutte le tabelle
        tables = pd.read_html(StringIO(r.text), header=[0, 1])

        for table in tables:
            # Appiattisci i MultiIndex di colonne
            table.columns = [
                f"{a}_{b}".strip() if "Unnamed" not in str(a) else str(b).strip()
                for a, b in table.columns
            ]
            cols = list(table.columns)

            # Cerca colonne xG e xGA (FBref le chiama 'Expected_xG' e 'Expected_xGA')
            xg_col = next((c for c in cols if c == "Expected_xG" or c.endswith("_xG") and "xGA" not in c), None)
            xga_col = next((c for c in cols if "xGA" in c), None)
            squad_col = next((c for c in cols if "Squad" in c), None)
            mp_col = next((c for c in cols if c in ("Playing Time_MP", "MP", "Playing Time_Starts")), None)

            if not (xg_col and xga_col and squad_col):
                continue

            # Filtra righe valide
            df = table[[squad_col, xg_col, xga_col]].copy()
            if mp_col:
                df["MP"] = pd.to_numeric(table[mp_col], errors="coerce")
            else:
                df["MP"] = 20  # fallback

            df = df[df[squad_col].notna()]
            df = df[~df[squad_col].str.contains("Squad|vs", na=True)]
            df[xg_col] = pd.to_numeric(df[xg_col], errors="coerce")
            df[xga_col] = pd.to_numeric(df[xga_col], errors="coerce")
            df = df.dropna(subset=[xg_col, xga_col])

            if len(df) < 5:
                continue

            result = {}
            for _, row in df.iterrows():
                nome_fbref = str(row[squad_col]).strip()
                nome_app = FBREF_NAME_MAP.get(nome_fbref, nome_fbref)
                mp = max(float(row.get("MP", 20)), 1)
                result[nome_app] = {
                    "xG_avg":  round(float(row[xg_col])  / mp, 3),
                    "xGA_avg": round(float(row[xga_col]) / mp, 3),
                }

            if len(result) >= 10:
                return result

    except Exception as e:
        print(f"FBref xG errore: {e}")

    return None


def _parse_understat_xg(league_name):
    """Fallback: vecchio scraper Understat."""
    leagues = {
        "Serie A": "Serie_A",
        "Premier League": "EPL",
        "La Liga": "La_Liga",
        "Bundesliga": "Bundesliga",
    }
    url = f"https://understat.com/league/{leagues.get(league_name)}/2025"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.content, "lxml")
        scripts = soup.find_all("script")
        data_script = ""
        for s in scripts:
            if "teamsData" in s.text:
                data_script = s.text
                break
        start = data_script.find("('") + 2
        end = data_script.find("')")
        json_data = data_script[start:end].encode("utf-8").decode("unicode_escape")
        full_data = json.loads(json_data)
        parsed = {}
        for team_id in full_data:
            info = full_data[team_id]
            history = info["history"]
            if not history:
                continue
            matches = len(history)
            parsed[info["title"]] = {
                "xG_avg":  round(sum(float(x["xG"])  for x in history) / matches, 3),
                "xGA_avg": round(sum(float(x["xGA"]) for x in history) / matches, 3),
            }
        return parsed if parsed else None
    except Exception as e:
        print(f"Understat xG errore: {e}")
        return None


def get_understat_xg(league_name):
    """
    Tenta FBref prima, poi Understat come fallback.
    Interfaccia invariata: ritorna {nome: {xG_avg, xGA_avg}} o None.
    """
    result = _parse_fbref_xg(league_name)
    if result:
        print(f"xG caricati da FBref: {len(result)} squadre ({league_name})")
        return result

    result = _parse_understat_xg(league_name)
    if result:
        print(f"xG caricati da Understat (fallback): {len(result)} squadre ({league_name})")
        return result

    print(f"xG non disponibili per {league_name}")
    return None


def get_market_values():
    return {
        # Serie A
        "Inter": 600, "Milan": 550, "Juventus": 500, "Napoli": 450, "Atalanta": 400,
        "Roma": 350, "Lazio": 300, "Fiorentina": 250, "Bologna": 200, "Torino": 180,
        "Monza": 120, "Genoa": 110, "Lecce": 80, "Verona": 75, "Udinese": 90,
        "Cagliari": 70, "Empoli": 65, "Parma": 60, "Como": 55, "Venezia": 50,
        # Premier League
        "Man City": 900, "Arsenal": 850, "Liverpool": 900, "Chelsea": 750,
        "Man United": 600, "Tottenham": 500, "Newcastle": 450, "Aston Villa": 400,
        "West Ham": 300, "Brighton": 280,
        # La Liga
        "Real Madrid": 1100, "Barcelona": 1000, "Atletico Madrid": 700,
        "Athletic Club": 350, "Villarreal": 300,
        # Bundesliga
        "Bayern": 900, "Leverkusen": 600, "Dortmund": 550, "Leipzig": 450, "Frankfurt": 300,
    }
