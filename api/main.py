from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import os
import sys

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(
    title="Music Recommender API",
    description="API for Guts of Darkness music recommendation system",
    version="1.0.0"
)

# Global variables
collection = None
model = None

class AlbumResult(BaseModel):
    id: int
    title: str
    artist: str
    distance: float
    extra: Optional[Dict[str, Any]] = {}

class SearchResponse(BaseModel):
    results: List[AlbumResult]

@app.on_event("startup")
async def startup_event():
    global collection, model
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            collection.load()
            print(f"✅ Connected to Milvus collection '{COLLECTION_NAME}'.")
        else:
            print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Please run the load script.")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")

    print(f"Loading model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/health")
def health():
    milvus_status = "KO"
    try:
        if connections.has_connection("default"):
            milvus_status = "OK"
    except:
        pass
        
    return {
        "status": "OK",
        "milvus": milvus_status,
        "model": "OK" if model else "KO"
    }

@app.post("/search", response_model=SearchResponse)
def search(query: str = Query(..., description="Text query"), top_k: int = 10):
    if not collection or not model:
        raise HTTPException(status_code=503, detail="Service not initialized (Milvus or Model missing)")

    try:
        # 1. Encode query
        query_embedding = model.encode([query])
        
        # 2. Search in Milvus
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["album_name", "artist_name"]
        )
        
        # 3. Format results
        output = []
        for hits in results:
            for hit in hits:
                output.append(AlbumResult(
                    id=hit.id,
                    title=hit.entity.get("album_name"),
                    artist=hit.entity.get("artist_name"),
                    distance=hit.distance,
                    extra={} # Placeholder for extra fields if we add them later
                ))
        
        return SearchResponse(results=output)

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
