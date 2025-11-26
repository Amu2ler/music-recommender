"""
Module d'ingestion de données musicales
Scraping, nettoyage et préparation des données
"""

from .scraper import scrape_albums, scrape_url
from .cleaning import clean_dataset
from .embeddings import create_embeddings

__all__ = [
    "scrape_albums",
    "scrape_url", 
    "clean_dataset",
    "create_embeddings",
]

