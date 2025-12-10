"""
🎵 MUSIC RECOMMENDER - UNIFIED PIPELINE 🎵
Script principal pour lancer tout le processus de bout en bout.
"""
import argparse
from src.ingestion.scraper import run_scraper
from src.vectorization.embedder import run_vectorization
from src.vectorization.reducer import generate_2d_map
from src.ingestion.loader import load_database
import os

def main():
    parser = argparse.ArgumentParser(description="Music Recommender Pipeline")
    parser.add_argument("--steps", type=str, default="all", help="Steps to run: all, scrape, vectorize, reduce, load")
    parser.add_argument("--input-csv", type=str, default="data/processed/sample_albums.csv", help="Input CSV file")
    parser.add_argument("--output-parquet", type=str, default="data/processed/sample_albums_embedded.parquet", help="Output Parquet file")
    parser.add_argument("--map-output", type=str, default="data/processed/albums_2d.parquet", help="Output 2D Map file")
    parser.add_argument("--styles-output", type=str, default="data/processed/styles.json", help="Output styles file")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 STARTING PIPELINE")
    print("="*60)
    
    # 1. Scraping (Optional/Check)
    if args.steps in ["all", "scrape"]:
        print("\n[1/4] 🕷️  Checking Data / Scraping")
        # Par défaut, on ne relance pas le scraping complet sauf si demandé explicitement
        # Ici on vérifie juste que le fichier existe, ou on le génère s'il manque
        if not os.path.exists(args.input_csv):
            print("Input file missing, running scraper...")
            run_scraper(output_path=args.input_csv, end_id=100) # Petite limite par défaut
        else:
            print(f"✅ Data file found: {args.input_csv}")

    # 2. Vectorization (Cleaning + Embedding)
    if args.steps in ["all", "vectorize"]:
        print("\n[2/4] 🧠 Vectorization (Cleaning + Embedding)")
        run_vectorization(args.input_csv, args.output_parquet)

    # 3. Dimensionality Reduction (UMAP)
    if args.steps in ["all", "reduce"]:
        print("\n[3/4] 🗺️  Dimensionality Reduction (2D Map)")
        generate_2d_map(args.output_parquet, args.map_output, args.styles_output)

    # 4. Loading to Milvus
    if args.steps in ["all", "load"]:
        print("\n[4/4] 🔌 Loading to Milvus")
        # On recrée la collection si on lance tout le pipeline
        recreate = (args.steps == "all")
        load_database(args.output_parquet, recreate=recreate)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
