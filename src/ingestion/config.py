"""Configuration centrale pour l'ingestion de données"""

from pathlib import Path

# Chemins
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Fichiers
RAW_ALBUMS_CSV = DATA_PROCESSED / "sample_albums.csv"
CLEAN_ALBUMS_CSV = DATA_PROCESSED / "sample_albums_clean.csv"
EMBEDDED_ALBUMS_PARQUET = DATA_PROCESSED / "sample_albums_embedded.parquet"

# Scraping
GUTS_BASE_URL = "https://www.gutsofdarkness.com/god/objet.php?objet="
HEADERS = {
    "User-Agent": "MusicRecommenderBot/1.0 (+mailto:contact@example.com)"
}
REQUEST_TIMEOUT = 10
MAX_THREADS = 32

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "music_embeddings"

