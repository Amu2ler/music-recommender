import requests
import pandas as pd
from time import sleep
import random
import re
import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


HEADERS = {
    "User-Agent": "MusicRecommenderBot/0.1 (+mailto:ton.email@example.com)"
}

def parse_album(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # === TITRE & ARTISTE ===
    h1 = soup.select_one("h1")
    title = h1.find("em").get_text(strip=True) if h1 and h1.find("em") else None
    artist = h1.get_text(" ", strip=True).replace(title, "").replace(">", "").strip() if h1 else None

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
        vide = len(sous_chronique_div.select("span.gfxNoteVide"))
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

    # === TAGS (ajoutés par utilisateurs) ===
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
    

def generate_album_links(start_id: int, end_id: int):
    BASE_URL = "https://www.gutsofdarkness.com/god/objet.php?objet="
    urls = [f"{BASE_URL}{i}" for i in range(start_id, end_id + 1)]
    print(f" {len(urls)} liens générés ({urls[0]} → {urls[-1]})")
    return urls


def scrape_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        parsed = parse_album(r.text, url)
        if parsed["album_name"] and parsed["artist_name"]:
            return parsed
        return None
    except Exception:
        return None