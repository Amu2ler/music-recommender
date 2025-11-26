import pandas as pd
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
import sys

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"
DATA_PATH = "data/processed/sample_albums_embedded.parquet"

def connect_milvus():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ Connected to Milvus.")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        sys.exit(1)

def create_collection():
    # Check if collection exists
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists. Dropping it...")
        utility.drop_collection(COLLECTION_NAME)

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

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Ensure embeddings are list of floats
    if "embedding" not in df.columns:
        print("❌ Column 'embedding' missing in parquet file.")
        sys.exit(1)

    # Convert string embeddings if necessary (e.g. "[0.1, ...]")
    if isinstance(df["embedding"].iloc[0], str):
        print("Converting string embeddings to list...")
        df["embedding"] = df["embedding"].apply(lambda x: eval(x))
    
    # Ensure they are lists (Milvus expects lists for float vectors)
    if isinstance(df["embedding"].iloc[0], np.ndarray):
        df["embedding"] = df["embedding"].apply(lambda x: x.tolist())

    print(f"✅ Loaded {len(df)} rows.")
    return df

def insert_data(collection, df):
    print("Inserting data into Milvus...")
    
    # Prepare data for insertion (column-based)
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

def main():
    connect_milvus()
    collection = create_collection()
    df = load_data()
    insert_data(collection, df)
    create_index(collection)
    print("\n🎉 Data loading complete!")

if __name__ == "__main__":
    main()
