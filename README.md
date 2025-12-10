# 🎵 Music Recommender - Système de Recommandation Musicale IA

## 📋 Description

Système de recommandation musicale intelligent développé en équipe, capable de suggérer des albums similaires en combinant plusieurs approches d'analyse :

- **Similarité sémantique** via embeddings textuels des critiques et métadonnées
- **Analyse des métadonnées** (artiste, genre, tags, année)
- **Interactions utilisateurs** (notes, commentaires)
- **Caractéristiques audio** (MFCC, tempo, spectre) pour enrichir les recommandations

Le système utilise une base de données vectorielle pour effectuer des recherches par similarité plutôt que par mots-clés exacts, offrant des recommandations plus pertinentes et contextuelles.

## 🛠️ Stack Technique

### Backend & API

- **FastAPI** - Framework web moderne et performant pour l'API REST
- **Python 3.11** - Langage principal du projet

### Intelligence Artificielle & Machine Learning

- **Sentence Transformers** - Génération d'embeddings textuels pour la similarité sémantique
- **Librosa** - Extraction de caractéristiques audio (MFCC, tempo, spectrogrammes)
- **Mutagen** - Manipulation des métadonnées audio

### Base de Données & Stockage

- **Milvus** - Base de données vectorielle pour recherche par similarité (cosine distance)
- **etcd** - Coordination distribuée pour Milvus
- **MinIO** - Stockage d'objets compatible S3

### Data Processing & Scraping

- **Pandas** - Manipulation et analyse de données
- **BeautifulSoup4** - Web scraping des critiques musicales
- **Requests** - Collecte de données depuis sites spécialisés

### Interface & Visualisation

- **Streamlit** - Interface utilisateur interactive pour démonstration

### DevOps & Infrastructure

- **Docker Compose** - Orchestration des services (Milvus, etcd, MinIO)
- **Loguru** - Logging avancé
- **Pytest** - Tests unitaires et d'intégration

## 🏗️ Architecture

```
SCRAPING (BeautifulSoup)
    ↓
DATAFRAME (Pandas - métadonnées normalisées)
    ↓
EMBEDDING (Sentence Transformers)
    ↓
MILVUS (Index vectoriel + recherche par similarité)
    ↓
FASTAPI (API REST)
    ↓
STREAMLIT (Interface utilisateur)
```

## 🎯 Fonctionnalités Clés

1. **Collecte de données** - Scraping automatisé de critiques musicales et métadonnées
2. **Vectorisation intelligente** - Transformation des textes en embeddings via modèles transformers
3. **Recherche par similarité** - Algorithmes de distance cosinus pour trouver des albums similaires
4. **API REST** - Endpoints pour requêtes de recommandation en temps réel
5. **Filtrage avancé** - Combinaison de critères (notes, genres, année) avec similarité sémantique

## 👥 Équipe

Projet collaboratif développé par :

- Arthur Muller
- Abdoulaye Diallo
- Semih Taskin

## 🔑 Points Techniques Notables

- Utilisation de **Vector Store (Milvus)** pour des performances optimales sur grandes quantités de données
- Architecture **microservices** avec Docker Compose
- Pipeline complet de **ML Engineering** : collecte → nettoyage → vectorisation → indexation → API
- Approche **hybride** combinant NLP et analyse audio
