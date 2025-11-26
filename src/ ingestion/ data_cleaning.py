import argparse
import json
import re
import sys
from typing import List, Dict
from pathlib import Path
from loguru import logger

from .config import RAW_GUTS_JSON, CLEAN_ALBUMS_JSON, DATA_PROCESSED

def clean_text(t: str) -> str:
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_valid(a: Dict) -> bool:
    # Description suffisante pour éviter le bruit
    if not a.get("title") or not a.get("artist") or not a.get("description"):
        return False
    return len(a["description"]) >= 50

def main():
    parser = argparse.ArgumentParser(description="Nettoyage/normalisation des albums scrapés.")
    parser.add_argument("--in", dest="infile", type=str, default=str(RAW_GUTS_JSON))
    parser.add_argument("--out", dest="outfile", type=str, default=str(CLEAN_ALBUMS_JSON))
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.infile)
    out_path = Path(args.outfile)

    logger.add(sys.stderr, level="INFO")
    logger.info(f"Cleaning {in_path} → {out_path}")

    items: List[Dict] = json.loads(in_path.read_text(encoding="utf-8"))
    seen = set()
    cleaned: List[Dict] = []

    for a in items:
        a["title"] = clean_text(a.get("title", ""))
        a["artist"] = clean_text(a.get("artist", ""))
        a["description"] = clean_text(a.get("description", "")).lower()
        if a.get("genre"):
            a["genre"] = clean_text(a["genre"]).lower()

        key = (a["title"].lower(), a["artist"].lower())
        if key in seen:
            continue
        if is_valid(a):
            cleaned.append(a)
            seen.add(key)

    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Cleaned {len(cleaned)} items → {out_path}")

if __name__ == "__main__":
    main()
