import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

def get_market_values():
    # Valori medi aggiornati (in milioni di €) per calibrare il peso tecnico
    # Nota: Questi possono essere aggiornati periodicamente
    values = {
        "Inter": 600, "Milan": 550, "Juventus": 500, "Napoli": 450, "Atalanta": 400,
        "Roma": 350, "Lazio": 300, "Fiorentina": 250, "Bologna": 200, "Torino": 180,
        "Monza": 120, "Genoa": 110, "Lecce": 80, "Verona": 75, "Udinese": 90,
        "Cagliari": 70, "Empoli": 65, "Parma": 60, "Como": 55, "Venezia": 50, "Pisa": 45
    }
    return values

def get_understat_xg(league_name):
    # Mappatura nomi campionati per Understat
    leagues = {
        "Serie A": "Serie_A",
        "Premier League": "EPL",
        "La Liga": "La_Liga",
        "Bundesliga": "Bundesliga"
    }
    
    url = f"https://understat.com/league/{leagues.get(league_name)}/2025"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, "lxml")
        
        # Understat nasconde i dati in tag <script> in formato JSON
        scripts = soup.find_all('script')
        data_script = ""
        for s in scripts:
            if 'teamsData' in s.text:
                data_script = s.text
                break
        
        # Pulizia del testo per ottenere solo il JSON
        start = data_script.find("('") + 2
        end = data_script.find("')")
        json_data = data_script[start:end].encode('utf-8').decode('unicode_escape')
        full_data = json.loads(json_data)
        
        parsed_stats = {}
        for id in full_data:
            team_info = full_data[id]
            name = team_info['title']
            # Calcoliamo la media xG e xGA (Expected Goals Against) per partita
            history = team_info['history']
            total_xg = sum(float(x['xG']) for x in history)
            total_xga = sum(float(x['xGA']) for x in history)
            matches = len(history)
            
            parsed_stats[name] = {
                'xG_avg': total_xg / matches,
                'xGA_avg': total_xga / matches
            }
            
        return parsed_stats
    except Exception as e:
        print(f"Errore Scraping xG: {e}")
        return None
    