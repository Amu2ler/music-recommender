from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import sys

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "music_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "data/processed/sample_albums_embedded.parquet"

app = FastAPI(
    title="Music Recommender API",
    description="API for Guts of Darkness music recommendation system with advanced filters",
    version="2.0.0"
)

# Global variables
collection = None
model = None
metadata_df = None

class AlbumResult(BaseModel):
    id: int
    title: str
    artist: str
    distance: float
    note: Optional[float] = None
    styles: Optional[str] = None
    chronique_excerpt: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[AlbumResult]
    total_found: int
    filters_applied: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global collection, model, metadata_df
    
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
    
    print(f"Loading metadata from {DATA_PATH}...")
    try:
        metadata_df = pd.read_parquet(DATA_PATH)
        print(f"✅ Metadata loaded ({len(metadata_df)} albums).")
    except Exception as e:
        print(f"⚠️ Failed to load metadata: {e}")

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
        "model": "OK" if model else "KO",
        "metadata": "OK" if metadata_df is not None else "KO"
    }

@app.post("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="Text query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    min_note: Optional[float] = Query(None, ge=0, le=6, description="Minimum average note"),
    styles: Optional[str] = Query(None, description="Comma-separated styles to filter"),
    sort_by: str = Query("score", regex="^(score|note|alphabetical)$", description="Sort results by")
):
    if not collection or not model:
        raise HTTPException(status_code=503, detail="Service not initialized (Milvus or Model missing)")

    try:
        # 1. Encode query
        query_embedding = model.encode([query])
        
        # 2. Search in Milvus (get more results for filtering)
        search_limit = top_k * 3 if (min_note or styles) else top_k
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=search_limit,
            output_fields=["album_name", "artist_name"]
        )
        
        # 3. Enrich with metadata and apply filters
        output = []
        for hits in results:
            for hit in hits:
                # Get metadata
                if metadata_df is not None:
                    meta_row = metadata_df[
                        (metadata_df["album_name"] == hit.entity.get("album_name")) & 
                        (metadata_df["artist_name"] == hit.entity.get("artist_name"))
                    ]
                    
                    if not meta_row.empty:
                        meta = meta_row.iloc[0]
                        album_note = meta.get("note_moyenne")
                        album_styles = meta.get("styles", "")
                        chronique = meta.get("chronique", "")
                        
                        # Apply filters
                        if min_note and (pd.isna(album_note) or album_note < min_note):
                            continue
                        
                        if styles:
                            required_styles = [s.strip().lower() for s in styles.split(",")]
                            album_styles_list = [s.strip().lower() for s in album_styles.split(";") if s.strip()]
                            if not any(req in album_styles_list for req in required_styles):
                                continue
                        
                        # Create result
                        excerpt = chronique[:200] + "..." if len(chronique) > 200 else chronique
                        output.append(AlbumResult(
                            id=hit.id,
                            title=hit.entity.get("album_name"),
                            artist=hit.entity.get("artist_name"),
                            distance=hit.distance,
                            note=album_note if not pd.isna(album_note) else None,
                            styles=album_styles,
                            chronique_excerpt=excerpt
                        ))
                else:
                    # No metadata available
                    output.append(AlbumResult(
                        id=hit.id,
                        title=hit.entity.get("album_name"),
                        artist=hit.entity.get("artist_name"),
                        distance=hit.distance
                    ))
        
        # 4. Sort results
        if sort_by == "note":
            output.sort(key=lambda x: x.note if x.note else 0, reverse=True)
        elif sort_by == "alphabetical":
            output.sort(key=lambda x: f"{x.artist} {x.title}")
        # "score" is already sorted by distance (default)
        
        # 5. Limit to top_k
        output = output[:top_k]
        
        return SearchResponse(
            results=output,
            total_found=len(output),
            filters_applied={
                "min_note": min_note,
                "styles": styles,
                "sort_by": sort_by
            }
        )

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
