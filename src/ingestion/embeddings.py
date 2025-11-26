"""
Création d'embeddings textuels pour les albums
Utilise SentenceTransformers pour encoder les descriptions
"""

import sys
from typing import Optional

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from .config import CLEAN_ALBUMS_CSV, EMBEDDED_ALBUMS_PARQUET, EMBEDDING_MODEL, DATA_PROCESSED


def create_embeddings(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model_name: str = EMBEDDING_MODEL,
    text_column: str = "text_full"
) -> pd.DataFrame:
    """
    Crée des embeddings pour chaque album
    
    Args:
        input_path: Chemin du CSV nettoyé (par défaut: CLEAN_ALBUMS_CSV)
        output_path: Chemin du fichier Parquet avec embeddings (par défaut: EMBEDDED_ALBUMS_PARQUET)
        model_name: Nom du modèle SentenceTransformer
        text_column: Colonne contenant le texte à encoder
    
    Returns:
        DataFrame avec la colonne 'embedding' ajoutée
    """
    if input_path is None:
        input_path = CLEAN_ALBUMS_CSV
    if output_path is None:
        output_path = EMBEDDED_ALBUMS_PARQUET
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Charger les données nettoyées
    logger.info(f"Chargement depuis {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"{len(df)} albums chargés")
    
    # Vérifier que la colonne texte existe
    if text_column not in df.columns:
        raise ValueError(f"Colonne '{text_column}' introuvable. Colonnes disponibles : {df.columns.tolist()}")
    
    # Charger le modèle
    logger.info(f"Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Créer les embeddings
    logger.info("Création des embeddings...")
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Ajouter au DataFrame
    df["embedding"] = [emb.tolist() for emb in embeddings]
    
    # Sauvegarder en Parquet (format efficace pour les arrays)
    logger.info(f"Sauvegarde vers {output_path}...")
    df.to_parquet(output_path, index=False)
    
    logger.success(f"✅ Embeddings créés : {len(df)} albums, dim={len(embeddings[0])} → {output_path}")
    
    return df


if __name__ == "__main__":
    logger.add(sys.stderr, level="INFO")
    df = create_embeddings()
    print(f"\n✅ {len(df)} albums avec embeddings")
    print(f"Dimensions : {len(df['embedding'][0])}")
    print(df[["artist_name", "album_name"]].head())

