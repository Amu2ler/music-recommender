"""
Generate 2D embeddings from 384D vectors using UMAP
For interactive visualization in the Explore page
"""
import pandas as pd
import numpy as np
from umap import UMAP
from pathlib import Path
import json

# Configuration
INPUT_FILE = Path("data/processed/sample_albums_embedded.parquet")
OUTPUT_FILE = Path("data/processed/albums_2d.parquet")
STYLES_FILE = Path("data/processed/styles.json")

print("=" * 60)
print("🗺️  GENERATING 2D EMBEDDINGS FOR VISUALIZATION")
print("=" * 60)

# Load data
print(f"\n📥 Loading data from {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded {len(df)} albums")

# Extract embeddings
print("\n🔄 Extracting embeddings...")
embeddings = np.array(df["embedding"].tolist())
print(f"✅ Embeddings shape: {embeddings.shape}")

# Apply UMAP
print("\n🧮 Applying UMAP dimensionality reduction (384D → 2D)...")
print("   This may take a few minutes...")
umap_model = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    verbose=True
)
embeddings_2d = umap_model.fit_transform(embeddings)
print(f"✅ 2D embeddings generated: {embeddings_2d.shape}")

# Add 2D coordinates to dataframe
df["x"] = embeddings_2d[:, 0]
df["y"] = embeddings_2d[:, 1]

# Determine dominant style for each album
print("\n🎸 Determining dominant styles...")
def get_dominant_style(styles_str):
    if pd.isna(styles_str) or not styles_str:
        return "unknown"
    styles = [s.strip() for s in styles_str.split(";") if s.strip()]
    return styles[0] if styles else "unknown"

df["dominant_style"] = df["styles"].apply(get_dominant_style)

# Get top styles for color mapping
style_counts = df["dominant_style"].value_counts()
top_styles = style_counts.head(20).index.tolist()
print(f"✅ Top 20 styles identified: {', '.join(top_styles[:5])}...")

# Assign color category
def get_style_category(style):
    if style in top_styles:
        return style
    return "other"

df["style_category"] = df["dominant_style"].apply(get_style_category)

# Save result
print(f"\n💾 Saving to {OUTPUT_FILE}...")
# Keep only necessary columns for visualization
viz_df = df[[
    "album_name", "artist_name", "styles", "note_moyenne", 
    "x", "y", "dominant_style", "style_category"
]]
viz_df.to_parquet(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(viz_df)} albums with 2D coordinates")

# Save style categories for UI
style_categories = sorted(viz_df["style_category"].unique().tolist())
with open(STYLES_FILE.parent / "style_categories.json", "w") as f:
    json.dump(style_categories, f, indent=2)
print(f"✅ Saved {len(style_categories)} style categories")

print("\n" + "=" * 60)
print("🎉 2D EMBEDDINGS GENERATION COMPLETE!")
print("=" * 60)
print(f"📊 Statistics:")
print(f"   - Total albums: {len(viz_df)}")
print(f"   - X range: [{viz_df['x'].min():.2f}, {viz_df['x'].max():.2f}]")
print(f"   - Y range: [{viz_df['y'].min():.2f}, {viz_df['y'].max():.2f}]")
print(f"   - Style categories: {len(style_categories)}")
print("\n✅ Ready for visualization!")
