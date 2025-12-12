# 🎵 Music Recommender – Guts of Darkness

## 📌 Présentation du projet

Ce projet vise à concevoir un **système de recommandation musicale sémantique** à partir de données issues du site [Guts of Darkness](https://www.gutsofdarkness.com/) (site web non inclus dans le projet, uniquement une source de données).

L’objectif est de permettre à un utilisateur de découvrir des albums similaires à ses goûts en s’appuyant sur des techniques modernes de **Traitement du Langage Naturel (NLP)**, de **recherche vectorielle** et de **visualisation interactive**.

Le système repose sur un pipeline complet allant du _scraping_ des données jusqu’à leur exploitation via une API et une interface graphique.

## 🧠 Principe général

Le fonctionnement global du projet suit le pipeline suivant, allant de la donnée brute à la recommandation finale :

### 1. Collecte des données

- _Scraping_ des pages albums (artiste, styles, chroniques, notes, tags).
- Stockage intermédiaire sous forme de fichiers CSV.

### 2. Prétraitement et vectorisation

- Nettoyage des champs textuels (normalisation, suppression du bruit).
- Construction d’un **texte sémantique global** par album.
- Génération d’**embeddings** (représentations vectorielles) à l’aide d’un modèle de type _Sentence Transformers_.

### 3. Réduction dimensionnelle & visualisation

- Application d’**UMAP** pour projeter les embeddings en 2D.
- Préparation des données pour une **visualisation interactive** des similarités (carte des albums).

### 4. Stockage vectoriel

- Insertion des embeddings dans une **base de données vectorielle Milvus**.
- Indexation pour permettre des recherches rapides par similarité (recherche du plus proche voisin).
- Recherche sémantique à partir d’une requête textuelle (similarité vectorielle).

### 5. Recherche et recommandation

- Recherche sémantique à partir d’une requête textuelle.
- Filtrage possible par styles musicaux et notes.
- Classement des résultats selon différents critères.

### 6. Exposition via API et interface utilisateur

- **API REST** pour interroger le moteur de recommandation.
- **Interface graphique** pour la recherche et l’exploration visuelle des albums.

## ⚙️ Technologies utilisées

| Catégorie                            | Outils                                   | Description                                                                                          |
| :----------------------------------- | :--------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Langages & Frameworks**            | Python, FastAPI, Streamlit, Pandas/NumPy | Langage principal, API REST performante, Interface utilisateur interactive, Manipulation de données. |
| **Intelligence Artificielle & Data** | SentenceTransformers, UMAP               | Génération d'embeddings sémantiques, Réduction de dimension pour la visualisation.                   |
| **Recherche par Similarité**         | Milvus                                   | Base de données vectorielle spécialisée.                                                             |
| **Base de données & Infrastructure** | Docker & Docker Compose, MinIO, Etcd     | Orchestration des services, Stockage objet pour Milvus.                                              |
| **Visualisation**                    | Plotly                                   | Graphiques interactifs (carte 2D des albums).                                                        |

## 🗂️ Architecture du projet (vue logique)

```
.
├── api/                # API FastAPI (endpoints de recherche)
├── data/               # Données brutes et traitées
├── notebooks/          # Notebooks d'exploration et scripts d'ingestion
├── src/                # Code source (ingestion, vectorization, recommendation)
├── ui/                 # Interface Streamlit
├── run_pipeline.py     # Orchestrateur du pipeline
├── docker-compose.yml  # Services nécessaires (Milvus, MinIO...)
└── requirements.txt
```

## 🚀 Lancement du projet

Suivez les étapes ci-dessous pour démarrer et exploiter le système de recommandation.

### 1. Démarrer les services (Milvus, MinIO, Etcd)

```bash
docker-compose up -d
```

### 2\. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 3\. Lancer le pipeline complet (Scraping, Vectorisation, Indexation)

> **Note :** Cette étape peut être longue lors de la première exécution.

```bash
python run_pipeline.py
```

### 4\. Démarrer l’API

L'API sera accessible sur `http://127.0.0.1:8000`.

```bash
uvicorn api.main:app --reload
```

### 5\. Lancer l’interface utilisateur

```bash
streamlit run ui/_Search.py
```

## 🎯 Objectif pédagogique

Ce projet met en œuvre un pipeline complet de recommandation basé sur des techniques modernes de traitement du texte, de recherche vectorielle et de visualisation interactive, illustrant une application concrète de l’IA et de la data science à un cas réel.
