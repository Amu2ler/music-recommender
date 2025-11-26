"""
Script de scraping complet pour Guts of Darkness
Génère le fichier sample_albums.csv avec tous les albums
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import random
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configuration
HEADERS = {
    "User-Agent": "MusicRecommenderBot/1.0 (Educational Project)"
}
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "sample_albums.csv"
START_ID = 0
END_ID = 25000  # Ajuste selon tes besoins
MAX_WORKERS = 10  # Nombre de threads parallèles
SAVE_INTERVAL = 100  # Sauvegarde tous les N albums

def generate_album_links(start_id: int, end_id: int) -> list:
    """Génère les URLs d'albums à scraper"""
    BASE_URL = "https://www.gutsofdarkness.com/god/objet.php?objet="
    urls = [f"{BASE_URL}{i}" for i in range(start_id, end_id + 1)]
    print(f"📋 {len(urls)} liens générés ({urls[0]} → {urls[-1]})")
    return urls

def parse_album(html: str, url: str) -> dict:
    """Parse une page d'album et extrait les métadonnées"""
    soup = BeautifulSoup(html, "html.parser")

    # Titre & Artiste
    h1 = soup.select_one("h1")
    title = h1.find("em").get_text(strip=True) if h1 and h1.find("em") else None
    artist = h1.get_text(" ", strip=True).replace(title, "").replace(">", "").strip() if h1 else None

    # Line-up alternatif
    lineup = soup.select_one("#objetLineup p")
    if lineup and not artist:
        artist = lineup.get_text(" ", strip=True)

    # Informations
    info_div = soup.select_one("#objet-informations")
    infos = []
    if info_div:
        infos = [p.get_text(" ", strip=True) for p in info_div.find_all("p")]

    # Chronique
    chronique_div = soup.select_one("div.objet-chronique")
    chronique_text = ""
    if chronique_div:
        ps = chronique_div.find_all("p")
        chronique_text = " ".join(p.get_text(" ", strip=True) for p in ps)
        chronique_text = re.sub(r"\s+", " ", chronique_text).strip()

    # Styles
    style_div = soup.select_one("div.objet-style")
    styles = [a.get_text(strip=True) for a in style_div.find_all("a")] if style_div else []

    # Note de la chronique
    sous_chronique_div = soup.select_one("div.objet-sous-chronique div.discrete-info")
    note_chronique = None
    if sous_chronique_div:
        pleines = len(sous_chronique_div.select("span.gfxNotePleine"))
        demi = len(sous_chronique_div.select("span.gfxNoteDemi"))
        note_chronique = pleines + 0.5 * demi

    # Albums similaires
    related_section = soup.select("div.mosaique a h1 em")
    same_spirit = [em.get_text(strip=True) for em in related_section] if related_section else []

    # Note moyenne
    vote_div = soup.select_one("div#objetVote")
    note_moyenne = None
    if vote_div:
        pleines = len(vote_div.select("span.gfxNotePleine"))
        demi = len(vote_div.select("span.gfxNoteDemi"))
        note_moyenne = pleines + 0.5 * demi

    # Tags
    tags_div = soup.select_one("div#contenuObjetTags")
    tags_text = tags_div.get_text(" ", strip=True) if tags_div else ""

    return {
        "album_name": title,
        "artist_name": artist,
        "lineup": lineup.get_text(" ", strip=True) if lineup else None,
        "informations": " ".join(infos),
        "chronique": chronique_text,
        "styles": ";".join(styles),
        "note_chronique": note_chronique,
        "note_moyenne": note_moyenne,
        "same_spirit": ";".join(same_spirit),
        "tags_text": tags_text,
        "source_url": url,
    }

def scrape_url(url: str) -> dict | None:
    """Scrape une URL et retourne les données parsées"""
    try:
        sleep(0.5 + random.random())  # Politesse
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        parsed = parse_album(r.text, url)
        if parsed["album_name"] and parsed["artist_name"]:
            return parsed
        return None
    except Exception as e:
        print(f"❌ Erreur sur {url}: {e}")
        return None

def main():
    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Charger les données existantes si le fichier existe
    existing_urls = set()
    if OUTPUT_FILE.exists():
        print(f"📂 Fichier existant trouvé : {OUTPUT_FILE}")
        df_existing = pd.read_csv(OUTPUT_FILE)
        existing_urls = set(df_existing["source_url"].tolist())
        print(f"✅ {len(existing_urls)} albums déjà scrapés")
    
    # Générer les URLs
    urls = generate_album_links(START_ID, END_ID)
    urls_to_scrape = [url for url in urls if url not in existing_urls]
    print(f"🎯 {len(urls_to_scrape)} nouvelles URLs à scraper")
    
    if not urls_to_scrape:
        print("✅ Aucune nouvelle URL à scraper !")
        return
    
    # Scraping multithread
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_url, url): url for url in urls_to_scrape}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
                print(f"[{i}/{len(urls_to_scrape)}] ✅ {result['artist_name']} - {result['album_name']}")
            else:
                print(f"[{i}/{len(urls_to_scrape)}] ⚠️ Échec")
            
            # Sauvegarde intermédiaire
            if i % SAVE_INTERVAL == 0:
                df_new = pd.DataFrame(results)
                if OUTPUT_FILE.exists():
                    df_old = pd.read_csv(OUTPUT_FILE)
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    df_combined.drop_duplicates(subset=["source_url"], inplace=True)
                    df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                else:
                    df_new.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                print(f"💾 Sauvegarde intermédiaire ({len(results)} nouveaux albums)")
                results = []  # Reset pour la prochaine batch
    
    # Sauvegarde finale
    if results:
        df_new = pd.DataFrame(results)
        if OUTPUT_FILE.exists():
            df_old = pd.read_csv(OUTPUT_FILE)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=["source_url"], inplace=True)
            df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        else:
            df_new.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    
    # Statistiques finales
    df_final = pd.read_csv(OUTPUT_FILE)
    print(f"\n🎉 Scraping terminé !")
    print(f"📊 Total : {len(df_final)} albums dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
