from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import numpy as np

def ingest_data():
    print("Connecting to Milvus...")
    try:
        connections.connect("default", host="127.0.0.1", port="19530")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    collection_name = "music_embeddings"
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection '{collection_name}'...")
        utility.drop_collection(collection_name)

    print("Creating collection...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="artist", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields, description="Embeddings musicaux GutsOfDarkness")
    collection = Collection(collection_name, schema=schema)

    print("Loading data...")
    df = pd.read_parquet("../data/processed/sample_albums_embedded.parquet")
    
    # Ensure embeddings are list of floats
    if isinstance(df["embedding"].iloc[0], np.ndarray):
        df["embedding"] = df["embedding"].apply(lambda x: x.tolist())

    print(f"Inserting {len(df)} items...")
    collection.insert([
        df["album_name"].tolist(),
        df["artist_name"].tolist(),
        df["embedding"].tolist()
    ])
    
    print("Creating index...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    collection.load()
    
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
