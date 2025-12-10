"""
Module de chargement dans Milvus
"""
import pandas as pd
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
import sys

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"

def connect_milvus():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ Connected to Milvus.")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        # On ne quitte pas brutalement ici pour laisser le pipeline gérer l'erreur
        raise e

def create_collection(drop_existing=False):
    # Check if collection exists
    if utility.has_collection(COLLECTION_NAME):
        if drop_existing:
            print(f"⚠️ Collection '{COLLECTION_NAME}' already exists. Dropping it...")
            utility.drop_collection(COLLECTION_NAME)
        else:
            return Collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="album_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="artist_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields, description="Music Embeddings from GutsOfDarkness")
    collection = Collection(COLLECTION_NAME, schema=schema)
    print("✅ Collection created.")
    return collection

def load_data(data_path):
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return None
        
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ensure embeddings are list of floats
    if "embedding" not in df.columns:
        print("❌ Column 'embedding' missing in parquet file.")
        return None

    # Convert string embeddings if necessary
    if isinstance(df["embedding"].iloc[0], str):
        df["embedding"] = df["embedding"].apply(lambda x: eval(x))
    
    # Ensure they are lists
    if isinstance(df["embedding"].iloc[0], np.ndarray):
        df["embedding"] = df["embedding"].apply(lambda x: x.tolist())

    print(f"✅ Loaded {len(df)} rows.")
    return df

def insert_data(collection, df):
    print("Inserting data into Milvus...")
    
    if df is None:
        print("No data to insert")
        return

    data_to_insert = [
        df["album_name"].tolist(),
        df["artist_name"].tolist(),
        df["embedding"].tolist()
    ]
    
    collection.insert(data_to_insert)
    collection.flush()
    print(f"✅ Inserted {collection.num_entities} entities.")

def create_index(collection):
    print("Creating index...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    collection.load()
    print("✅ Index created and collection loaded.")

def load_database(data_path: str, recreate: bool = False):
    """Fonction principale pour charger la donnée dans Milvus"""
    connect_milvus()
    collection = create_collection(drop_existing=recreate)
    df = load_data(data_path)
    if df is not None:
        insert_data(collection, df)
        create_index(collection)
