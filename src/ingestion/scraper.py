"""
Scraper pour Guts of Darkness
Extraction des métadonnées d'albums avec multithreading
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from .config import GUTS_BASE_URL, HEADERS, REQUEST_TIMEOUT, MAX_THREADS, RAW_ALBUMS_CSV, DATA_PROCESSED


def parse_album(html: str, url: str) -> Optional[Dict]:
    """Parse une page HTML d'album et extrait les métadonnées"""
    soup = BeautifulSoup(html, "html.parser")

    # === TITRE & ARTISTE ===
    h1 = soup.select_one("h1")
    title = h1.find("em").get_text(strip=True) if h1 and h1.find("em") else None
    artist = h1.get_text(" ", strip=True).replace(title or "", "").replace(">", "").strip() if h1 else None

    # Alternative : line-up
    lineup = soup.select_one("#objetLineup p")
    if lineup and not artist:
        artist = lineup.get_text(" ", strip=True)

    # === INFORMATIONS ===
    info_div = soup.select_one("#objet-informations")
    infos = []
    if info_div:
        infos = [p.get_text(" ", strip=True) for p in info_div.find_all("p")]

    # === CHRONIQUE ===
    chronique_div = soup.select_one("div.objet-chronique")
    chronique_text = ""
    if chronique_div:
        ps = chronique_div.find_all("p")
        chronique_text = " ".join(p.get_text(" ", strip=True) for p in ps)
        chronique_text = re.sub(r"\s+", " ", chronique_text).strip()

    # === STYLES ===
    style_div = soup.select_one("div.objet-style")
    styles = [a.get_text(strip=True) for a in style_div.find_all("a")] if style_div else []

    # === NOTE DE LA CHRONIQUE ===
    sous_chronique_div = soup.select_one("div.objet-sous-chronique div.discrete-info")
    note_chronique = None
    if sous_chronique_div:
        pleines = len(sous_chronique_div.select("span.gfxNotePleine"))
        demi = len(sous_chronique_div.select("span.gfxNoteDemi"))
        note_chronique = pleines + 0.5 * demi

    # === ALBUMS "DANS LE MÊME ESPRIT" ===
    related_section = soup.select("div.mosaique a h1 em")
    same_spirit = [em.get_text(strip=True) for em in related_section] if related_section else []

    # === NOTE MOYENNE ===
    vote_div = soup.select_one("div#objetVote")
    note_moyenne = None
    if vote_div:
        pleines = len(vote_div.select("span.gfxNotePleine"))
        demi = len(vote_div.select("span.gfxNoteDemi"))
        note_moyenne = pleines + 0.5 * demi

    # === TAGS ===
    tags_div = soup.select_one("div#contenuObjetTags")
    tags_text = ""
    if tags_div:
        tags_text = tags_div.get_text(" ", strip=True)

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


def scrape_url(url: str) -> Optional[Dict]:
    """Scrape une URL unique et retourne les données parsées"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None

        parsed = parse_album(r.text, url)
        if parsed and parsed["album_name"] and parsed["artist_name"]:
            return parsed
        return None
    except Exception as e:
        logger.debug(f"Erreur scraping {url}: {e}")
        return None


def generate_album_links(start_id: int, end_id: int) -> List[str]:
    """Génère une liste d'URLs d'albums à partir des IDs"""
    urls = [f"{GUTS_BASE_URL}{i}" for i in range(start_id, end_id + 1)]
    logger.info(f"{len(urls)} liens générés ({urls[0]} → {urls[-1]})")
    return urls


def scrape_albums(
    start_id: int = 0,
    end_id: int = 25000,
    output_path: str = None,
    max_workers: int = MAX_THREADS,
    save_interval: int = 100
) -> pd.DataFrame:
    """
    Scrape une plage d'albums avec multithreading
    
    Args:
        start_id: ID de départ
        end_id: ID de fin (inclus)
        output_path: Chemin du fichier CSV de sortie
        max_workers: Nombre de threads parallèles
        save_interval: Sauvegarder tous les N albums
    
    Returns:
        DataFrame avec les albums scrapés
    """
    if output_path is None:
        output_path = RAW_ALBUMS_CSV
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Charger les albums déjà scrapés
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        scraped_urls = set(df_existing["source_url"])
        logger.info(f"{len(scraped_urls)} albums déjà présents — ils seront ignorés.")
    else:
        scraped_urls = set()
        df_existing = None
        logger.info("Aucun dataset existant, scraping complet.")

    # Générer les URLs
    urls = generate_album_links(start_id, end_id)
    urls_to_scrape = [u for u in urls if u not in scraped_urls]
    logger.info(f"Lancement du scraping multithread ({len(urls_to_scrape)} nouvelles URLs)...")

    rows = []
    max_workers = min(max_workers, (os.cpu_count() or 1) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_url, url): url for url in urls_to_scrape}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                rows.append(result)
                logger.info(f"[{i}/{len(urls_to_scrape)}] {result['artist_name']} - {result['album_name']}")
            else:
                logger.debug(f"[{i}/{len(urls_to_scrape)}] Échec ou page vide")

            # Sauvegarde intermédiaire
            if i % save_interval == 0 and rows:
                temp_df = pd.DataFrame(rows)
                if df_existing is not None:
                    temp_df = pd.concat([df_existing, temp_df], ignore_index=True)
                    temp_df.drop_duplicates(subset=["source_url"], inplace=True)
                temp_df.to_csv(output_path, index=False, encoding="utf-8")
                logger.success(f"Sauvegarde intermédiaire ({len(temp_df)} albums) → {output_path}")
                rows = []

    # Fusion finale
    df = pd.DataFrame(rows)
    if df_existing is not None:
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.drop_duplicates(subset=["source_url"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.success(f"Scraping terminé : {len(df)} albums uniques → {output_path}")
    return df


if __name__ == "__main__":
    import sys
    logger.add(sys.stderr, level="INFO")
    
    # Scraper une petite plage pour test
    df = scrape_albums(start_id=1000, end_id=1100)
    print(f"\n✅ {len(df)} albums scrapés")
    print(df.head())

