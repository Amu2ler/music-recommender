from pathlib import Path

# Base directories (projet racine = 3 niveaux au-dessus de ce fichier)
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Fichiers I/O
RAW_GUTS_JSON = DATA_RAW / "guts_albums.json"
CLEAN_ALBUMS_JSON = DATA_PROCESSED / "albums_clean.json"
AUDIO_META_JSON = DATA_PROCESSED / "audio_metadata.json"

# Scraping
GUTS_BASE_URL = "https://www.gutsofdarkness.com/god/objet.php?objet="
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 1.0  # politesse

# Audio
AUDIO_DIR = DATA_RAW / "audio"
