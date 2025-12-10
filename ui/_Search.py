import streamlit as st
import requests
import os
import json

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
STYLES_FILE = "data/processed/styles.json"

st.set_page_config(page_title="Music Recommender", layout="wide", page_icon="🎵")

# Load available styles
try:
    with open(STYLES_FILE, 'r') as f:
        available_styles = json.load(f)
except:
    available_styles = []

# Custom CSS
st.markdown("""
<style>
    .album-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .style-tag {
        display: inline-block;
        background-color: #e1e4e8;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title(" Guts of Darkness Recommender")
st.markdown("*Découvrez des albums grâce à la recherche sémantique*")

# Sidebar
with st.sidebar:
    st.header(" Filtres")
    
    # Health check
    if st.button(" Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                st.success(f"API: {data.get('status')}")
                st.info(f"Milvus: {data.get('milvus')}")
                st.info(f"Model: {data.get('model')}")
                st.info(f"Metadata: {data.get('metadata')}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
    
    st.divider()
    
    # Note filter
    st.subheader(" Note minimale")
    min_note = st.slider(
        "Filtrer par note",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.5,
        help="Albums avec une note moyenne >= à cette valeur"
    )
    use_note_filter = st.checkbox("Activer le filtre de note", value=False)
    
    st.divider()
    
    # Styles filter
    st.subheader(" Styles musicaux")
    if available_styles:
        selected_styles = st.multiselect(
            "Sélectionner des styles",
            options=available_styles[:50],  # Limit to first 50 for performance
            help="Albums contenant au moins un de ces styles"
        )
    else:
        selected_styles = []
        st.info("Styles non disponibles")
    
    st.divider()
    
    # Sort options
    st.subheader(" Tri des résultats")
    sort_by = st.selectbox(
        "Trier par",
        options=["score", "note", "alphabetical"],
        format_func=lambda x: {
            "score": "Score de similarité",
            "note": "Note moyenne",
            "alphabetical": "Ordre alphabétique"
        }[x]
    )

# Main search
st.header(" Recherche")
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Décrivez ce que vous voulez écouter",
        placeholder="Ex: dark ambient industrial, atmospheric black metal...",
        label_visibility="collapsed"
    )
with col2:
    top_k = st.number_input("Résultats", min_value=1, max_value=50, value=10)

if st.button(" Rechercher", type="primary", use_container_width=True):
    if query:
        with st.spinner("Recherche en cours..."):
            try:
                # Build params
                params = {
                    "query": query,
                    "top_k": top_k,
                    "sort_by": sort_by
                }
                
                if use_note_filter and min_note > 0:
                    params["min_note"] = min_note
                
                if selected_styles:
                    params["styles"] = ",".join(selected_styles)
                
                # API call
                response = requests.post(f"{API_URL}/search", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    total = data.get("total_found", 0)
                    filters = data.get("filters_applied", {})
                    
                    # Display filters applied
                    if filters.get("min_note") or filters.get("styles"):
                        st.info(f" Filtres appliqués: " + 
                               (f"Note ≥ {filters.get('min_note')}" if filters.get("min_note") else "") +
                               (f" | Styles: {filters.get('styles')}" if filters.get("styles") else ""))
                    
                    st.success(f"✅ {total} résultat(s) trouvé(s)")
                    
                    if results:
                        for i, album in enumerate(results, 1):
                            # Convert score (0-1) to 1-6 scale for better readability
                            raw_score = album['distance']
                            score_6 = 1 + (raw_score * 5)  # Convert 0-1 to 1-6
                            
                            # Score color and label
                            if raw_score >= 0.6:
                                score_class = "score-excellent"
                                score_label = "⭐⭐⭐⭐⭐"
                            elif raw_score >= 0.5:
                                score_class = "score-good"
                                score_label = "⭐⭐⭐⭐"
                            elif raw_score >= 0.4:
                                score_class = "score-medium"
                                score_label = "⭐⭐⭐"
                            else:
                                score_class = "score-low"
                                score_label = "⭐⭐"
                            
                            # Display card
                            with st.container():
                                st.markdown(f"""
                                <div class="album-card">
                                    <h3 style="color: #1f1f1f; margin-bottom: 0.5rem;">{i}. {album['title']}</h3>
                                    <p style="color: #4a4a4a; font-size: 1.1rem; margin-bottom: 0.5rem;"><strong> Artiste:</strong> {album['artist']}</p>
                                    <p style="margin-bottom: 0.5rem;"><span class="{score_class}"> Score de similarité: {score_6:.1f}/6 {score_label}</span></p>
                                """, unsafe_allow_html=True)
                                
                                # Note
                                if album.get('note'):
                                    note_stars = "⭐" * int(album['note'])
                                    st.markdown(f"<p style='color: #2c3e50;'><strong> Note moyenne:</strong> {album['note']:.1f}/6 {note_stars}</p>", unsafe_allow_html=True)
                                
                                # Styles
                                if album.get('styles'):
                                    styles_list = [s.strip() for s in album['styles'].split(';') if s.strip()]
                                    if styles_list:
                                        styles_html = " ".join([f'<span class="style-tag">{s}</span>' for s in styles_list[:5]])
                                        st.markdown(f"<p style='color: #2c3e50;'><strong>🎸 Styles:</strong> {styles_html}</p>", unsafe_allow_html=True)
                                
                                # Chronique excerpt
                                if album.get('chronique_excerpt'):
                                    with st.expander(" Extrait de la chronique"):
                                        st.write(album['chronique_excerpt'])
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Aucun résultat ne correspond aux critères.")
                else:
                    st.error(f"Erreur API: {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion: {e}")
    else:
        st.warning("Veuillez entrer une requête.")
