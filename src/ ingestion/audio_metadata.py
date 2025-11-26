import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from mutagen import File as MutagenFile

from .config import AUDIO_DIR, AUDIO_META_JSON, DATA_PROCESSED

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}

def extract_audio_meta(audio_path: Path) -> Optional[Dict]:
    try:
        audio = MutagenFile(audio_path.as_posix())
    except Exception as e:
        logger.warning(f"Mutagen error on {audio_path}: {e}")
        return None

    if audio is None:
        return None

    info = getattr(audio, "info", None)

    def get_tag(tag):
        try:
            if hasattr(audio, "tags") and audio.tags and tag in audio.tags:
                return str(audio.tags.get(tag))
        except Exception:
            return None
        return None

    return {
        "path": audio_path.as_posix(),
        "filename": audio_path.name,
        "duration": round(getattr(info, "length", 0.0), 2) if info else None,
        "bitrate": getattr(info, "bitrate", None),
        "tags_title": get_tag("TIT2"),
        "tags_artist": get_tag("TPE1"),
    }

def main():
    parser = argparse.ArgumentParser(description="Extraction métadonnées audio locales.")
    parser.add_argument("--dir", type=str, default=str(AUDIO_DIR))
    parser.add_argument("--out", type=str, default=str(AUDIO_META_JSON))
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    base = Path(args.dir)
    out_path = Path(args.out)

    logger.add(sys.stderr, level="INFO")
    logger.info(f"Scanning audio in {base}")

    rows: List[Dict] = []
    if not base.exists():
        logger.warning(f"No audio dir: {base} (skip)")
    else:
        for p in base.rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS:
                meta = extract_audio_meta(p)
                if meta:
                    rows.append(meta)

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Saved {len(rows)} audio rows → {out_path}")

if __name__ == "__main__":
    main()
