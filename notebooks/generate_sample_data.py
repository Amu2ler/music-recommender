import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_sample_data():
    print("Generating sample data...")
    
    # Create directory if not exists
    os.makedirs("../data/processed", exist_ok=True)
    
    # Dummy data
    data = [
        {
            "album_name": "Dark Side of the Moon",
            "artist_name": "Pink Floyd",
            "chronique": "A masterpiece of progressive rock.",
            "styles": "Progressive Rock; Psychedelic",
            "note_moyenne": 5.0,
            "source_url": "http://example.com/1"
        },
        {
            "album_name": "Unknown Pleasures",
            "artist_name": "Joy Division",
            "chronique": "Dark, atmospheric post-punk classic.",
            "styles": "Post-Punk; Gothic",
            "note_moyenne": 4.8,
            "source_url": "http://example.com/2"
        },
        {
            "album_name": "Selected Ambient Works 85-92",
            "artist_name": "Aphex Twin",
            "chronique": "Ambient techno masterpiece.",
            "styles": "Ambient; Techno",
            "note_moyenne": 4.9,
            "source_url": "http://example.com/3"
        }
    ]
    
    df = pd.DataFrame(data)
    
    # Generate embeddings
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding...")
    texts = (df["album_name"] + " " + df["artist_name"] + " " + df["styles"] + " " + df["chronique"]).tolist()
    embeddings = model.encode(texts)
    
    df["embedding"] = [emb.tolist() for emb in embeddings]
    
    output_path = "../data/processed/sample_albums_embedded.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Sample data saved to {output_path}")

if __name__ == "__main__":
    generate_sample_data()
