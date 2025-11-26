"""
Script principal pour orchestrer le pipeline d'ingestion complet
"""

import sys
from pathlib import Path

from loguru import logger

# Importer les modules
from ingestion.scraper import scrape_albums
from ingestion.cleaning import clean_dataset
from ingestion.embeddings import create_embeddings
from vectorization.vector_store import MilvusStore


def run_full_pipeline(
    scrape: bool = True,
    start_id: int = 0,
    end_id: int = 1000,
    create_embeddings_flag: bool = True,
    insert_milvus: bool = True
):
    """
    Exécute le pipeline complet d'ingestion
    
    Args:
        scrape: Activer le scraping
        start_id: ID de début pour le scraping
        end_id: ID de fin pour le scraping
        create_embeddings_flag: Créer les embeddings
        insert_milvus: Insérer dans Milvus
    """
    logger.add(sys.stderr, level="INFO")
    
    # Phase 1 : Scraping
    if scrape:
        logger.info("=" * 60)
        logger.info("PHASE 1 : SCRAPING")
        logger.info("=" * 60)
        df_raw = scrape_albums(start_id=start_id, end_id=end_id)
        logger.success(f"✅ Scraping terminé : {len(df_raw)} albums")
    
    # Phase 2 : Nettoyage
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 : NETTOYAGE")
    logger.info("=" * 60)
    df_clean = clean_dataset()
    logger.success(f"✅ Nettoyage terminé : {len(df_clean)} albums")
    
    # Phase 3 : Embeddings
    if create_embeddings_flag:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3 : CRÉATION DES EMBEDDINGS")
        logger.info("=" * 60)
        df_embedded = create_embeddings()
        logger.success(f"✅ Embeddings créés : {len(df_embedded)} albums")
    
    # Phase 4 : Insertion dans Milvus
    if insert_milvus:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4 : INSERTION DANS MILVUS")
        logger.info("=" * 60)
        
        # Charger les données avec embeddings
        import pandas as pd
        from ingestion.config import EMBEDDED_ALBUMS_PARQUET
        df_embedded = pd.read_parquet(EMBEDDED_ALBUMS_PARQUET)
        
        # Connexion et création de la collection
        store = MilvusStore()
        store.connect()
        store.create_collection(drop_existing=False)
        store.create_index()
        
        # Insertion
        store.insert_from_dataframe(df_embedded)
        logger.success(f"✅ {len(df_embedded)} albums insérés dans Milvus")
        
        store.disconnect()
    
    logger.info("\n" + "=" * 60)
    logger.success("🎉 PIPELINE COMPLET TERMINÉ !")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion musicale")
    parser.add_argument("--no-scrape", action="store_true", help="Skip scraping")
    parser.add_argument("--start", type=int, default=0, help="Start ID for scraping")
    parser.add_argument("--end", type=int, default=1000, help="End ID for scraping")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings")
    parser.add_argument("--no-milvus", action="store_true", help="Skip Milvus insertion")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        scrape=not args.no_scrape,
        start_id=args.start,
        end_id=args.end,
        create_embeddings_flag=not args.no_embeddings,
        insert_milvus=not args.no_milvus
    )

