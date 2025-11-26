"""
Pipeline complet : Nettoyage + Embeddings + Chargement Milvus
Pour le dataset complet de 14k albums
"""
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("data/processed")
INPUT_CSV = DATA_DIR / "sample_albums.csv"
OUTPUT_PARQUET = DATA_DIR / "sample_albums_embedded.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # Pour l'encodage par batch
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"

print("=" * 60)
print("🎵 PIPELINE COMPLET - MUSIC RECOMMENDER")
print("=" * 60)

# ============================================================================
# ÉTAPE 1 : NETTOYAGE DES DONNÉES
# ============================================================================
print("\n📋 ÉTAPE 1/4 : Chargement et nettoyage des données")
print("-" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"✅ {len(df)} lignes chargées depuis {INPUT_CSV}")

# Supprimer les doublons
df.drop_duplicates(subset=["source_url"], inplace=True)
print(f"✅ Après déduplication : {len(df)} lignes")

# Supprimer les lignes sans album_name ou artist_name
df.dropna(subset=["album_name", "artist_name"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✅ Après nettoyage : {len(df)} lignes")

# Fonction de nettoyage de texte
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t)  # Espaces multiples → un seul
    t = re.sub(r"[^\w\s,.!?;:()-]", "", t)  # Enlever caractères bizarres
    return t.strip().lower()

# Nettoyer les colonnes textuelles
text_columns = ["album_name", "artist_name", "styles", "chronique", "informations", "tags_text"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

print(f"✅ Colonnes textuelles nettoyées")

# Créer le champ texte complet pour l'embedding
df["text_full"] = (
    df["artist_name"] + " " +
    df["album_name"] + " " +
    df["styles"] + " " +
    df["chronique"] + " " +
    df["informations"] + " " +
    df["tags_text"]
)
print(f"✅ Champ 'text_full' créé")

# ============================================================================
# ÉTAPE 2 : GÉNÉRATION DES EMBEDDINGS
# ============================================================================
print("\n🧠 ÉTAPE 2/4 : Génération des embeddings")
print("-" * 60)

print(f"📥 Chargement du modèle '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)
print(f"✅ Modèle chargé")

print(f"🔄 Encodage de {len(df)} textes (batch size: {BATCH_SIZE})...")
texts = df["text_full"].tolist()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"✅ Embeddings générés : shape {embeddings.shape}")

# Ajouter les embeddings au DataFrame
df["embedding"] = [emb.tolist() for emb in embeddings]

# Sauvegarder en Parquet
print(f"💾 Sauvegarde dans {OUTPUT_PARQUET}...")
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"✅ Fichier Parquet sauvegardé ({OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.2f} MB)")

# ============================================================================
# ÉTAPE 3 : CONNEXION À MILVUS
# ============================================================================
print("\n🔌 ÉTAPE 3/4 : Connexion à Milvus")
print("-" * 60)

print(f"Connexion à Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ Connecté à Milvus")
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
    exit(1)

# Supprimer l'ancienne collection si elle existe
if utility.has_collection(COLLECTION_NAME):
    print(f"⚠️ Collection '{COLLECTION_NAME}' existe déjà. Suppression...")
    utility.drop_collection(COLLECTION_NAME)
    print("✅ Ancienne collection supprimée")

# Créer la nouvelle collection
print(f"📦 Création de la collection '{COLLECTION_NAME}'...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="album_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="artist_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, description="Music Embeddings - Full Dataset")
collection = Collection(COLLECTION_NAME, schema=schema)
print("✅ Collection créée")

# ============================================================================
# ÉTAPE 4 : INSERTION DES DONNÉES
# ============================================================================
print("\n📤 ÉTAPE 4/4 : Insertion dans Milvus")
print("-" * 60)

print(f"Préparation de {len(df)} entités...")
data_to_insert = [
    df["album_name"].tolist(),
    df["artist_name"].tolist(),
    df["embedding"].tolist()
]

print("🔄 Insertion en cours...")
collection.insert(data_to_insert)
collection.flush()
print(f"✅ {collection.num_entities} entités insérées")

# Créer l'index
print("🔧 Création de l'index...")
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 1024}  # Plus de clusters pour 14k entités
}
collection.create_index("embedding", index_params)
print("✅ Index créé")

# Charger la collection en mémoire
print("📥 Chargement de la collection...")
collection.load()
print("✅ Collection chargée et prête")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 60)
print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")
print("=" * 60)
print(f"📊 Statistiques finales :")
print(f"   - Albums traités : {len(df)}")
print(f"   - Embeddings générés : {len(df)} vecteurs de dimension 384")
print(f"   - Entités dans Milvus : {collection.num_entities}")
print(f"   - Fichier Parquet : {OUTPUT_PARQUET}")
print("\n✅ Le système est prêt à être utilisé !")
print("   - API : http://127.0.0.1:8000")
print("   - UI : http://localhost:8501")
print("=" * 60)
