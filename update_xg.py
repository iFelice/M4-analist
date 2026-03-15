"""
update_xg.py
Scarica gli xG per squadra da FBref e li salva in database/xg_<campionato>.json
Gira in GitHub Actions settimanalmente.
"""

import requests
import pandas as pd
import json
import os
import time
from io import StringIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

FBREF_URLS = {
    "serie_a":        "https://fbref.com/en/comps/11/2025-2026/stats/2025-2026-Serie-A-Stats",
    "premier_league": "https://fbref.com/en/comps/9/2025-2026/stats/2025-2026-Premier-League-Stats",
    "la_liga":        "https://fbref.com/en/comps/12/2025-2026/stats/2025-2026-La-Liga-Stats",
    "bundesliga":     "https://fbref.com/en/comps/20/2025-2026/stats/2025-2026-Bundesliga-Stats",
}

NAME_MAP = {
    "Inter Milan": "Inter", "Milan": "Milan", "Juventus": "Juventus",
    "Napoli": "Napoli", "Atalanta": "Atalanta", "Roma": "Roma",
    "Lazio": "Lazio", "Fiorentina": "Fiorentina", "Bologna": "Bologna",
    "Torino": "Torino", "Monza": "Monza", "Genoa": "Genoa",
    "Lecce": "Lecce", "Hellas Verona": "Verona", "Udinese": "Udinese",
    "Cagliari": "Cagliari", "Empoli": "Empoli", "Parma": "Parma",
    "Como": "Como", "Venezia": "Venezia", "Cremonese": "Cremonese",
    "Manchester City": "Man City", "Manchester Utd": "Man United",
    "Arsenal": "Arsenal", "Liverpool": "Liverpool", "Chelsea": "Chelsea",
    "Tottenham": "Tottenham", "Newcastle Utd": "Newcastle",
    "Aston Villa": "Aston Villa", "West Ham": "West Ham", "Brighton": "Brighton",
    "Real Madrid": "Real Madrid", "Barcelona": "Barcelona",
    "Atlético Madrid": "Atletico Madrid", "Athletic Club": "Athletic Club",
    "Villarreal": "Villarreal", "Bayern Munich": "Bayern",
    "Bayer Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "Leipzig", "Eintracht Frankfurt": "Frankfurt",
}

def fetch_xg(league_key, url):
    print(f"Fetching xG per {league_key}...")
    try:
        session = requests.Session()
        # Prima visita homepage per cookie
        session.get("https://fbref.com/", headers=HEADERS, timeout=15)
        time.sleep(4)

        r = session.get(url, headers=HEADERS, timeout=20)
        print(f"  HTTP {r.status_code}, bytes: {len(r.content)}")

        if r.status_code == 429:
            print("  Rate limited da FBref")
            return None
        if r.status_code != 200:
            return None

        tables = pd.read_html(StringIO(r.text), header=[0, 1])
        print(f"  Trovate {len(tables)} tabelle")

        for i, table in enumerate(tables):
            flat_cols = []
            for a, b in table.columns:
                a, b = str(a).strip(), str(b).strip()
                flat_cols.append(b if "Unnamed" in a else f"{a}_{b}")
            table.columns = flat_cols

            squad_col = next((c for c in flat_cols if c == "Squad"), None)
            xg_col    = next((c for c in flat_cols if c == "Expected_xG"), None)
            xga_col   = next((c for c in flat_cols if "xGA" in c and "Expected" in c), None)
            mp_col    = next((c for c in flat_cols if c in ("Playing Time_MP", "MP")), None)

            if not (squad_col and xg_col and xga_col):
                continue

            df = table[[squad_col, xg_col, xga_col]].copy()
            df = df[df[squad_col].notna()]
            df = df[~df[squad_col].astype(str).str.contains("Squad|vs", na=True)]
            df[xg_col]  = pd.to_numeric(df[xg_col],  errors="coerce")
            df[xga_col] = pd.to_numeric(df[xga_col], errors="coerce")
            df = df.dropna(subset=[xg_col, xga_col])

            if len(df) < 10:
                continue

            mp_series = pd.to_numeric(table.get(mp_col, pd.Series(dtype=float)), errors="coerce") if mp_col else None

            result = {}
            for idx, row in df.iterrows():
                nome = NAME_MAP.get(str(row[squad_col]).strip(), str(row[squad_col]).strip())
                mp = float(mp_series.iloc[idx]) if mp_series is not None and idx < len(mp_series) and pd.notna(mp_series.iloc[idx]) else 20
                mp = max(mp, 1)
                result[nome] = {
                    "xG_avg":  round(float(row[xg_col])  / mp, 3),
                    "xGA_avg": round(float(row[xga_col]) / mp, 3),
                }

            if len(result) >= 10:
                print(f"  OK: {len(result)} squadre")
                return result

        print("  Nessuna tabella xG trovata")
        return None

    except Exception as e:
        print(f"  Errore: {e}")
        return None


def main():
    os.makedirs("database", exist_ok=True)
    successi = 0

    for league_key, url in FBREF_URLS.items():
        data = fetch_xg(league_key, url)
        if data:
            path = f"database/xg_{league_key}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Salvato: {path}")
            successi += 1
        else:
            print(f"  SKIP {league_key}")
        time.sleep(6)  # Rispetta rate limit FBref

    print(f"\nCompletato: {successi}/{len(FBREF_URLS)} campionati")
    if successi == 0:
        print("ATTENZIONE: nessun xG scaricato - FBref potrebbe bloccare il runner")


if __name__ == "__main__":
    main()
