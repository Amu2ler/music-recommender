import subprocess
import sys
from pathlib import Path
from loguru import logger

# Exécute la phase 2 au complet, dans l’ordre
def run(cmd):
    logger.info("⇒ " + " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    logger.add(sys.stderr, level="INFO")
    root = Path(__file__).resolve().parents[2]

    run([sys.executable, str(root / "src" / "ingestion" / "guts_scraper.py"),
         "--start", "1000", "--end", "1050"])  # ~50 albums

    run([sys.executable, str(root / "src" / "ingestion" / "data_cleaning.py")])

    run([sys.executable, str(root / "src" / "ingestion" / "audio_metadata.py")])

    logger.success("✓ Phase 2 terminée (ingestion/clean/audio).")
