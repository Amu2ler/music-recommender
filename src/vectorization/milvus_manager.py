from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import numpy as np


# === 1. Connexion à Milvus ===
def connect_milvus(host="localhost", port="19530"):
    connections.connect("default", host=host, port=port)
    print(f"✅ Connecté à Milvus ({host}:{port})")


# === 2. Création du schéma et de la collection ===
def create_collection(collection_name="music_embeddings", dim=384, drop_if_exists=True):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="album_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="artist_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Embeddings musicaux GutsOfDarkness")

    # Supprime si existe déjà
    if drop_if_exists and collection_name in utility.list_collections():
        Collection(collection_name).drop()
        print(f"🗑 Ancienne collection '{collection_name}' supprimée.")

    collection = Collection(collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' créée.")
    return collection


# === 3. Chargement du dataset ===
def load_dataset(path):
    df = pd.read_parquet(path)
    print(f"📦 {len(df)} lignes chargées depuis {path}")
    if isinstance(df["embedding"].iloc[0], str):
        df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x), dtype=np.float32))
    return df


# === 4. Insertion des données ===
def insert_data(collection, df):
    # Nettoyer avant insertion
    df = df.dropna(subset=["album_name", "artist_name", "embedding"]).copy()
    df["album_name"] = df["album_name"].fillna("").astype(str)
    df["artist_name"] = df["artist_name"].fillna("").astype(str)

    data = [
        df["album_name"].tolist(),
        df["artist_name"].tolist(),
        df["embedding"].tolist()
    ]

    collection.insert(data)
    collection.flush()
    print(f"✅ {len(df)} morceaux insérés dans Milvus (après nettoyage).")



# === 5. Création d’un index vectoriel ===
def create_index(collection):
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index("embedding", index_params)
    print("✅ Index créé avec succès.")


# === 6. Recherche vectorielle ===
def search_album(collection, query, model, top_k=5):
    query_vec = model.encode([query]).astype(np.float32)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    collection.load()
    print("✅ Collection chargée en mémoire pour la recherche.")

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["album_name", "artist_name"]
    )

    print(f"\n🎯 Résultats pour la requête : '{query}'\n")
    for hits in results:
        for hit in hits:
            print(f"🎧 {hit.entity.get('album_name')} — {hit.entity.get('artist_name')} (score: {hit.distance:.3f})")

    return results

