import argparse
import json
import time
import re
import sys
from typing import Optional, Dict, List
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from loguru import logger

from .config import (
    GUTS_BASE_URL, REQUEST_TIMEOUT, REQUEST_DELAY_SEC,
    RAW_GUTS_JSON, DATA_RAW
)

HEADERS = {
    "User-Agent": "Academic project scraper (contact: example@example.com)"
}

def scrape_album(album_id: int) -> Optional[Dict]:
    """Scrape une page album de Guts of Darkness par ID."""
    url = f"{GUTS_BASE_URL}{album_id}"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
    except requests.RequestException as e:
        logger.warning(f"Req error for {url}: {e}")
        return None
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # éléments principaux (peuvent varier selon pages)
    h1 = soup.find("h1")
    h2 = soup.find("h2")
    chro = soup.find(id="chro") or soup.find("div", class_="chronique")

    if not (h1 and h2 and chro):
        return None

    def norm_txt(x: str) -> str:
        x = x.replace("\xa0", " ")
        x = re.sub(r"\s+", " ", x).strip()
        return x

    title = norm_txt(h1.get_text(strip=True))
    artist = norm_txt(h2.get_text(strip=True))
    description = norm_txt(chro.get_text(" ", strip=True))

    # année/genre : si présents dans la page (optionnel, dépend du site)
    year = None
    genre = None
    # Exemples d’extraction (faibles, dépend du HTML réel)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", soup.get_text(" ", strip=True))
    if year_match:
        try:
            year = int(year_match.group(0))
        except ValueError:
            year = None

    item = {
        "id": str(album_id),
        "title": title,
        "artist": artist,
        "year": year,
        "genre": genre,  # laissé à None si non trouvé
        "description": description,
        "source": "guts_of_darkness",
        "source_url": url
    }
    return item

def scrape_range(start: int, end: int, delay: float) -> List[Dict]:
    """Scrape une plage d'IDs [start, end) et retourne la liste d’albums valides."""
    rows: List[Dict] = []
    for i in range(start, end):
        item = scrape_album(i)
        if item:
            rows.append(item)
            logger.info(f"+ {i} → {item['artist']} — {item['title']}")
        else:
            logger.debug(f"- {i} (not found/parse)")
        time.sleep(delay)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Scraper Guts of Darkness albums.")
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--end", type=int, default=1020)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SEC)
    parser.add_argument("--out", type=str, default=str(RAW_GUTS_JSON))
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    logger.add(sys.stderr, level="INFO")
    logger.info(f"Scraping IDs [{args.start}, {args.end}) → {out_path}")

    rows = scrape_range(args.start, args.end, args.delay)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Saved {len(rows)} items → {out_path}")

if __name__ == "__main__":
    main()
