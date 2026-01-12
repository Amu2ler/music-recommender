from pymilvus import connections, utility, Collection
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"

print(f"Connecting to {MILVUS_HOST}:{MILVUS_PORT}...")
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ Connection successful")
    
    if utility.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' exists")
        col = Collection(COLLECTION_NAME)
        print(f"✅ Collection loaded. Entities: {col.num_entities}")
    else:
        print(f"❌ Collection '{COLLECTION_NAME}' DOES NOT EXIST")
        
except Exception as e:
    print(f"❌ Connection FAILED: {e}")
