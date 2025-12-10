"""
Module de vectorisation (Cleaning + Embedding)
"""
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

def clean_text(t):
    """Nettoie une chaîne de texte"""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t)  # Espaces multiples → un seul
    t = re.sub(r"[^\w\s,.!?;:()-]", "", t)  # Enlever caractères bizarres
    return t.strip().lower()

def preprocess_data(input_csv: str) -> pd.DataFrame:
    """Charge et nettoie les données brutes"""
    print(f"Chargement de {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Supprimer les doublons
    if "source_url" in df.columns:
        df.drop_duplicates(subset=["source_url"], inplace=True)
    
    # Supprimer les lignes sans album_name ou artist_name
    df.dropna(subset=["album_name", "artist_name"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Nettoyer les colonnes textuelles
    text_columns = ["album_name", "artist_name", "styles", "chronique", "informations", "tags_text"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            
    # Créer le champ texte complet pour l'embedding
    # On gère le cas où certaines colonnes n'existent pas
    parts = []
    if "artist_name" in df.columns: parts.append(df["artist_name"])
    if "album_name" in df.columns: parts.append(df["album_name"])
    if "styles" in df.columns: parts.append(df["styles"])
    if "chronique" in df.columns: parts.append(df["chronique"])
    if "informations" in df.columns: parts.append(df["informations"])
    if "tags_text" in df.columns: parts.append(df["tags_text"])
    
    # Concaténation vectiorielle (plus rapide)
    df["text_full"] = parts[0]
    for p in parts[1:]:
        df["text_full"] = df["text_full"] + " " + p.fillna("")
        
    return df

def generate_embeddings(df: pd.DataFrame, model_name=MODEL_NAME) -> pd.DataFrame:
    """Génère les embeddings pour le DataFrame donné"""
    print(f"Chargement du modèle '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    print(f"Encodage de {len(df)} textes...")
    texts = df["text_full"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    df["embedding"] = [emb.tolist() for emb in embeddings]
    return df

def run_vectorization(input_csv: str, output_parquet: str) -> pd.DataFrame:
    """Pipeline complet de vectorisation"""
    df = preprocess_data(input_csv)
    df = generate_embeddings(df)
    
    # Sauvegarde
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Données vectorisées sauvegardées dans {output_parquet}")
    
    return df
