import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Explore - Music Recommender", layout="wide", page_icon="🗺️")

st.title("🗺️ Explore - Carte de Similarité")
st.markdown("*Visualisez les albums dans un espace 2D basé sur leur similarité*")

# Sidebar filters
with st.sidebar:
    st.header("⚙️ Filtres")
    
    # Fetch data first to get style categories
    try:
        response = requests.get(f"{API_URL}/explore")
        if response.status_code == 200:
            data = response.json()
            style_categories = data.get("style_categories", [])
            
            # Style filter
            st.subheader("🎸 Style")
            selected_style = st.selectbox(
                "Filtrer par style",
                options=["all"] + style_categories,
                format_func=lambda x: "Tous les styles" if x == "all" else x
            )
            
            # Note filter
            st.subheader("📊 Note minimale")
            min_note = st.slider(
                "Note",
                min_value=0.0,
                max_value=6.0,
                value=0.0,
                step=0.5
            )
            use_note_filter = st.checkbox("Activer le filtre de note", value=False)
            
            # Apply filters button
            if st.button("🔄 Appliquer les filtres", type="primary"):
                st.rerun()
        else:
            st.error("Impossible de charger les données")
            style_categories = []
            selected_style = "all"
            min_note = 0.0
            use_note_filter = False
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        style_categories = []
        selected_style = "all"
        min_note = 0.0
        use_note_filter = False

# Main content
try:
    # Build params
    params = {}
    if selected_style != "all":
        params["style_filter"] = selected_style
    if use_note_filter and min_note > 0:
        params["min_note"] = min_note
    
    # Fetch data
    response = requests.get(f"{API_URL}/explore", params=params)
    
    if response.status_code == 200:
        data = response.json()
        albums = data.get("albums", [])
        total = data.get("total", 0)
        
        if albums:
            # Convert to DataFrame
            df = pd.DataFrame(albums)
            
            # Info
            st.info(f"📊 {total} albums affichés")
            
            # Create interactive scatter plot
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="style_category",
                hover_data={
                    "title": True,
                    "artist": True,
                    "note": ":.1f",
                    "x": False,
                    "y": False,
                    "style_category": False
                },
                labels={
                    "style_category": "Style",
                    "title": "Album",
                    "artist": "Artiste",
                    "note": "Note"
                },
                title="Carte de Similarité des Albums",
                width=1200,
                height=700
            )
            
            # Customize layout
            fig.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')),
                hovertemplate="<b>%{customdata[0]}</b><br>" +
                             "Artiste: %{customdata[1]}<br>" +
                             "Note: %{customdata[2]:.1f}/6<br>" +
                             "<extra></extra>"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(240,240,240,0.5)',
                xaxis=dict(showgrid=True, gridcolor='white', title=""),
                yaxis=dict(showgrid=True, gridcolor='white', title=""),
                legend=dict(
                    title="Styles",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='closest'
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Instructions
            with st.expander("ℹ️ Comment utiliser cette carte ?"):
                st.markdown("""
                **Navigation :**
                - 🖱️ **Zoom** : Molette de la souris ou pincement
                - 👆 **Déplacer** : Cliquer-glisser
                - 🔍 **Hover** : Survolez un point pour voir les détails
                - 🎨 **Légende** : Cliquez sur un style pour le masquer/afficher
                
                **Interprétation :**
                - Les albums **proches** dans l'espace sont **similaires** musicalement
                - Les **clusters** (groupes) représentent des styles similaires
                - Les **couleurs** indiquent le style dominant de chaque album
                """)
            
            # Sample albums
            with st.expander("📋 Quelques albums affichés"):
                sample = df.head(10)[["title", "artist", "style_category", "note"]]
                st.dataframe(sample, use_container_width=True)
        else:
            st.warning("Aucun album ne correspond aux critères sélectionnés.")
    else:
        st.error(f"Erreur API: {response.text}")

except Exception as e:
    st.error(f"Erreur: {e}")
