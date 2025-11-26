"""
Nettoyage et préparation des données d'albums
Fusion des colonnes textuelles pour l'embedding
"""

import re
import sys
from typing import Optional

import pandas as pd
from loguru import logger

from .config import RAW_ALBUMS_CSV, CLEAN_ALBUMS_CSV, DATA_PROCESSED


def clean_text(t) -> str:
    """Nettoie un texte : espaces, caractères spéciaux, etc."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t)  # espaces multiples → un seul
    t = re.sub(r"[^\w\s,.!?;:()-]", "", t)  # enlever caractères bizarres
    return t.strip().lower()


def clean_dataset(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Nettoie le dataset d'albums :
    - Suppression des doublons
    - Nettoyage du texte
    - Fusion des colonnes textuelles en un seul champ
    
    Args:
        input_path: Chemin du CSV brut (par défaut: RAW_ALBUMS_CSV)
        output_path: Chemin du CSV nettoyé (par défaut: CLEAN_ALBUMS_CSV)
    
    Returns:
        DataFrame nettoyé
    """
    if input_path is None:
        input_path = RAW_ALBUMS_CSV
    if output_path is None:
        output_path = CLEAN_ALBUMS_CSV
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    logger.info(f"Chargement depuis {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"{len(df)} lignes brutes importées")

    # Supprimer doublons et NaN
    df.drop_duplicates(subset=["source_url"], inplace=True)
    df.dropna(subset=["album_name", "artist_name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Nettoyage de texte
    text_columns = ["album_name", "artist_name", "styles", "chronique", "informations"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Fusion en un seul champ texte complet pour l'embedding
    df["text_full"] = (
        df["artist_name"].fillna("") + " " +
        df["album_name"].fillna("") + " " +
        df["styles"].fillna("") + " " +
        df["chronique"].fillna("") + " " +
        df["informations"].fillna("")
    )
    
    # Nettoyer le champ fusionné
    df["text_full"] = df["text_full"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

    # Sauvegarder
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.success(f"✅ Données nettoyées : {len(df)} lignes → {output_path}")
    
    return df


if __name__ == "__main__":
    logger.add(sys.stderr, level="INFO")
    df = clean_dataset()
    print(f"\n✅ {len(df)} albums nettoyés")
    print(df[["artist_name", "album_name", "text_full"]].head())

