"""
update_xg.py
Scarica gli xG per squadra da FBref e li salva in database/xg_<campionato>.json
Gira in GitHub Actions settimanalmente.
"""

import requests
import pandas as pd
import json
import os
from io import StringIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://fbref.com/",
}

FBREF_URLS = {
    "serie_a":        "https://fbref.com/en/comps/11/2025-2026/stats/2025-2026-Serie-A-Stats",
    "premier_league": "https://fbref.com/en/comps/9/2025-2026/stats/2025-2026-Premier-League-Stats",
    "la_liga":        "https://fbref.com/en/comps/12/2025-2026/stats/2025-2026-La-Liga-Stats",
    "bundesliga":     "https://fbref.com/en/comps/20/2025-2026/stats/2025-2026-Bundesliga-Stats",
}

# Mapping nomi FBref → nomi usati nell'app
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
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} per {league_key}")
            return None

        tables = pd.read_html(StringIO(r.text), header=[0, 1])

        for table in tables:
            # Appiattisci MultiIndex colonne
            flat_cols = []
            for a, b in table.columns:
                a, b = str(a).strip(), str(b).strip()
                if "Unnamed" in a:
                    flat_cols.append(b)
                else:
                    flat_cols.append(f"{a}_{b}")
            table.columns = flat_cols

            # Cerca colonne chiave
            squad_col  = next((c for c in flat_cols if c == "Squad"), None)
            xg_col     = next((c for c in flat_cols if c == "Expected_xG"), None)
            xga_col    = next((c for c in flat_cols if c == "Expected_xGA"), None)
            mp_col     = next((c for c in flat_cols if c in ("Playing Time_MP", "MP")), None)

            if not (squad_col and xg_col and xga_col):
                continue

            df = table[[squad_col, xg_col, xga_col] + ([mp_col] if mp_col else [])].copy()
            df = df[df[squad_col].notna()]
            df = df[~df[squad_col].astype(str).str.contains("Squad|vs", na=True)]
            df[xg_col]  = pd.to_numeric(df[xg_col],  errors="coerce")
            df[xga_col] = pd.to_numeric(df[xga_col], errors="coerce")
            df = df.dropna(subset=[xg_col, xga_col])

            if len(df) < 10:
                continue

            result = {}
            for _, row in df.iterrows():
                nome_fbref = str(row[squad_col]).strip()
                nome_app   = NAME_MAP.get(nome_fbref, nome_fbref)
                mp = max(float(row[mp_col]) if mp_col and pd.notna(row.get(mp_col)) else 20, 1)
                result[nome_app] = {
                    "xG_avg":  round(float(row[xg_col])  / mp, 3),
                    "xGA_avg": round(float(row[xga_col]) / mp, 3),
                }

            if len(result) >= 10:
                print(f"  OK: {len(result)} squadre trovate")
                return result

        print(f"  Nessuna tabella xG trovata per {league_key}")
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
            out_path = f"database/xg_{league_key}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Salvato: {out_path}")
            successi += 1
        else:
            print(f"  SKIP: {league_key} non aggiornato")

    print(f"\nCompletato: {successi}/{len(FBREF_URLS)} campionati aggiornati")


if __name__ == "__main__":
    main()
