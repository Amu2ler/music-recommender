"""
Module de réduction dimensionnelle (UMAP 2D)
"""
import pandas as pd
import numpy as np
from umap import UMAP
from pathlib import Path
import json

def get_dominant_style(styles_str):
    if pd.isna(styles_str) or not styles_str:
        return "unknown"
    styles = [s.strip() for s in styles_str.split(";") if s.strip()]
    return styles[0] if styles else "unknown"

def generate_2d_map(input_parquet: str, output_parquet: str, styles_output: str):
    """Génère les coordonnées 2D pour la visualisation"""
    print(f"Chargement de {input_parquet}...")
    if not Path(input_parquet).exists():
        print(f"❌ Fichier non trouvé: {input_parquet}")
        return None
        
    df = pd.read_parquet(input_parquet)
    
    if "embedding" not in df.columns:
        print("❌ Colonne 'embedding' manquante")
        return None

    # Extraction des embeddings
    print("Mise en forme des données...")
    # Convertir en numpy array si nécessaire
    embeddings_list = df["embedding"].tolist()
    # Gérer le cas où c'est des chaînes (par sécurité)
    if isinstance(embeddings_list[0], str):
        embeddings_list = [eval(e) for e in embeddings_list]
        
    embeddings = np.array(embeddings_list)
    
    # UMAP
    print("Application de UMAP (384D -> 2D)...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    embeddings_2d = umap_model.fit_transform(embeddings)
    
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]
    
    # Styles
    df["dominant_style"] = df["styles"].apply(get_dominant_style)
    style_counts = df["dominant_style"].value_counts()
    top_styles = style_counts.head(20).index.tolist()
    
    def get_style_category(style):
        return style if style in top_styles else "other"
        
    df["style_category"] = df["dominant_style"].apply(get_style_category)
    
    # Sauvegarde Parquet réduit
    cols = ["album_name", "artist_name", "styles", "note_moyenne", "x", "y", "dominant_style", "style_category"]
    # Ajouter d'autres colonnes si besoin pour l'UI
    
    viz_df = df[cols]
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    viz_df.to_parquet(output_parquet, index=False)
    print(f"✅ Carte 2D sauvegardée dans {output_parquet}")
    
    # Sauvegarde styles JSON
    if styles_output:
        style_categories = sorted(viz_df["style_category"].unique().tolist())
        Path(styles_output).parent.mkdir(parents=True, exist_ok=True)
        with open(styles_output, "w") as f:
            json.dump(style_categories, f, indent=2)
            
    return viz_df
