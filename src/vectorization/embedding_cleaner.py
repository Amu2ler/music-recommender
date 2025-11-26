# src/vectorization/embedding_cleaner.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Charge le dataset brut à partir d’un CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées depuis {csv_path}")
    return df


def clean_text(text: str) -> str:
    """Nettoie un texte (supprime espaces multiples, caractères spéciaux inutiles, etc.)."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Combine les champs pertinents pour créer un texte d’entrée du modèle."""
    texts = []
    for _, row in df.iterrows():
        combined = f"{row['artist_name']} - {row['album_name']} {row.get('chronique', '')} {row.get('styles', '')}"
        texts.append(clean_text(combined))
    return texts


def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Crée les embeddings à partir des textes."""
    print(f"🧠 Chargement du modèle {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    print("✅ Embeddings générés.")
    return embeddings


def save_with_embeddings(df: pd.DataFrame, embeddings: np.ndarray, output_path: str):
    """Sauvegarde le DataFrame enrichi avec les embeddings."""
    df["embedding"] = embeddings.tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Données sauvegardées dans {output_path}")
