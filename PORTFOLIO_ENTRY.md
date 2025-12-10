# 🎵 Semantic Music Recommender

## 📝 Description
Un moteur de recommandation musical intelligent capable de comprendre le langage naturel. Contrairement à une recherche classique par mots-clés, ce système utilise l'IA (embeddings vectoriels) pour trouver des albums basés sur leur "vibe", leur description textuelle et leur similarité musicale. Le projet indexe plus de 14 000 albums de la scène underground.

## 🛠️ Technologies
*   **Langage** : Python 3.10
*   **Backend** : FastAPI, Pydantic
*   **IA & Data** : Sentence-Transformers (BERT), UMAP, Pandas, NumPy
*   **Base de Données** : Milvus (Vector DB), Docker
*   **Frontend** : Streamlit, Plotly (Visualisation interactive)
*   **DevOps** : Docker Compose, Git

## ✨ Fonctionnalités Principales
*   **Recherche Sémantique** : Permet des requêtes complexes comme *"dark ambient industrial with melancholic atmosphere"* et trouve les albums correspondants même sans correspondance de mots-clés exacte.
*   **Carte d'Exploration 2D** : Visualisation interactive de 14 000 albums projetés en 2D, regroupés par clusters de styles musicaux.
*   **Filtres Avancés** : Combinaison de recherche vectorielle avec des filtres de métadonnées (note minimale, styles, tri).
*   **Architecture Microservices** : Séparation propre entre l'API, la base de données vectorielle et l'interface utilisateur.

## 🔗 Liens
*   **Code Source** : [GitHub Repository](https://github.com/Amu2ler/music-recommender)
*   **Démo** : *(Ajoute ici un lien si tu héberges le projet, sinon supprime cette ligne)*

## 💡 Défis & Solutions
*   **Visualisation de données complexes** : Le défi était de représenter 14 000 vecteurs de 384 dimensions sur un écran 2D.
    *   *Solution* : Implémentation de l'algorithme **UMAP** pour la réduction dimensionnelle, permettant de conserver la structure locale (les albums similaires restent proches) tout en rendant la visualisation lisible.
*   **Performance de recherche** : Assurer une réponse instantanée sur un gros volume de données.
    *   *Solution* : Utilisation de **Milvus** avec un index IVF_FLAT pour une recherche de similarité ultra-rapide (<50ms).

## 🚀 Évolutions Possibles
*   Génération automatique de playlists basées sur un chemin dans le graphe 2D.
*   Intégration avec l'API Spotify pour écouter directement les recommandations.
