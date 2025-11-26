import pandas as pd
import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

class Recommender:
    def __init__(self, milvus_host="127.0.0.1", milvus_port="19530", data_path="../data/processed/sample_albums_embedded.parquet"):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.data_path = data_path
        self.collection_name = "music_embeddings"
        self.model_name = "all-MiniLM-L6-v2"
        
        self.collection = None
        self.model = None
        self.df = None
        
    def connect(self):
        """Connects to Milvus and loads data."""
        print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print("✅ Connected to Milvus.")
        except Exception as e:
            print(f"⚠️ Failed to connect to Milvus: {e}")
            
        print(f"Loading metadata from {self.data_path}...")
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"✅ Loaded {len(self.df)} albums.")
        except Exception as e:
            print(f"⚠️ Failed to load metadata: {e}")

        print("Loading embedding model...")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model loaded.")

    def search_by_text(self, query: str, top_k: int = 10):
        """Search albums by text query."""
        if not self.collection or not self.model:
            print("Error: Recommender not initialized. Call connect() first.")
            return []
            
        query_embedding = self.model.encode([query])
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "artist"]
        )
        
        output = []
        for hits in results:
            for hit in hits:
                # Enrich with metadata from DataFrame
                meta = self._get_metadata(hit.entity.get("title"), hit.entity.get("artist"))
                output.append({
                    "id": hit.id,
                    "title": hit.entity.get("title"),
                    "artist": hit.entity.get("artist"),
                    "distance": hit.distance,
                    "styles": meta.get("styles", "Unknown"),
                    "note": meta.get("note_moyenne", None)
                })
        return output

    def search_similar_albums(self, album_title: str, top_k: int = 10):
        """Find similar albums given a title."""
        if self.df is None:
            return []
            
        # Find embedding in DataFrame
        row = self.df[self.df["album_name"] == album_title]
        if row.empty:
            print(f"Album '{album_title}' not found in metadata.")
            return []
            
        embedding = row.iloc[0]["embedding"]
        if isinstance(embedding, np.ndarray):
            embedding = [embedding.tolist()]
        else:
            embedding = [embedding]

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k + 1, # +1 because the album itself will be found
            output_fields=["title", "artist"]
        )
        
        output = []
        for hits in results:
            for hit in hits:
                if hit.entity.get("title") == album_title:
                    continue # Skip self
                    
                meta = self._get_metadata(hit.entity.get("title"), hit.entity.get("artist"))
                output.append({
                    "id": hit.id,
                    "title": hit.entity.get("title"),
                    "artist": hit.entity.get("artist"),
                    "distance": hit.distance,
                    "styles": meta.get("styles", "Unknown"),
                    "note": meta.get("note_moyenne", None)
                })
        return output[:top_k]

    def _get_metadata(self, title, artist):
        if self.df is None:
            return {}
        row = self.df[(self.df["album_name"] == title) & (self.df["artist_name"] == artist)]
        if not row.empty:
            return row.iloc[0].to_dict()
        return {}
