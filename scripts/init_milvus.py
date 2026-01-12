
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.ingestion.loader import load_database

DATA_PATH = "data/processed/sample_albums_embedded.parquet"

if __name__ == "__main__":
    print("Starting manual database initialization...")
    # set recreate=True to force fresh start
    load_database(DATA_PATH, recreate=True)
    print("Database initialization complete.")
