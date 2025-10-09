# 🧠 Codex — Music Recommender AI

## 🎯 Objectif global

Développer un **système de recommandation musicale** capable de proposer des **albums similaires** en combinant :

- les **métadonnées** (artiste, album, genre, tags, année) ;
  - les **commentaires et critiques textuelles** provenant de sites spécialisés ;
  - les **notes et interactions utilisateurs** ;
  - la **similarité sémantique** entre descriptions via des embeddings ;
  - et, à terme, des **caractéristiques audio** (MFCC, tempo, spectre) pour enrichir les résultats.
    Le moteur repose sur un **Vector Store (Milvus)** qui permet des recherches par similarité (cosine distance) plutôt que par mots-clés exacts.

## ⚙️ Stack technique

| Domaine             | Outils / Librairies principales              |
| ------------------- | -------------------------------------------- |
| Backend API         | **FastAPI**                                  |
| Vector store        | **Milvus**                                   |
| Embeddings texte    | **Sentence-Transformers**                    |
| Audio features      | **Librosa**, **Mutagen**                     |
| Scraping & collecte | **Requests**, **BeautifulSoup4**, **pandas** |
| Interface (demo)    | **Streamlit**                                |
| Logging & tests     | **Loguru**, **Pytest**                       |
| Documentation       | **agent.md**, **codex.md**                   |

## 🏗️ Structure cible du dépôt

```
  music-recommender/
  │
  ├── README.md
  ├── agent.md
  ├── codex.md
  ├── requirements.txt
  ├── .gitignore
  │
  ├── src/
  │ ├── ingestion/
  │ ├── vectorization/
  │ ├── recommendation/
  │ └── api/
  │
  ├── data/
  │ ├── raw/
  │ └── processed/
  │
  ├── notebooks/
  ├── tests/
  └── docs/
- └── codex.md (présent fichier)
```

## 🪜 Étapes principales du projet

### 1. Collecte des données

- Source prioritaire : [Guts of Darkness](https://www.gutsofdarkness.com/god/).
  - Champs à récupérer pour chaque album :
- - `artist_name`
- - `album_name`
- - `review_text` (commentaire / critique)
- - `tags` (styles, genres)
- - `user_name` (auteur de la critique)
- - `user_rating` (note)
    - Format de sortie : `data/processed/albums.csv`.
    - **Une ligne = un album unique.**

### 2. Script de scraping

- Implémentation Python (`requests`, `BeautifulSoup4`, `pandas`).
  - Fonction principale envisagée : `collect_albums(url_list: list[str]) -> pd.DataFrame`.
  - Bonnes pratiques :
- - délai entre les requêtes ;
- - limitation des accès ;
- - tests sur échantillon réduit avant collecte complète.

### 3. Structure des données

Chaque album devient une entrée unique enrichie d’un vecteur d’embedding :
| id | artist_name | album_name | review_text | tags | user_rating | embedding_vector |
| -- | ----------- | ---------- | ----------- | ---- | ----------- | ---------------- |

### 4. Embeddings

- Concaténer les champs textuels (`artist_name`, `album_name`, `review_text`, `tags`).
  - Encoder via `SentenceTransformer('all-MiniLM-L6-v2')` (baseline).
  - Stocker le vecteur obtenu pour chaque album.

### 5. Stockage vectoriel (Milvus)

- Indexer les paires `album_id` `embedding` ainsi que les métadonnées utiles (`artist_name`, `user_rating`, `tags`).
  - Utiliser des métriques de similarité adaptées (cosine / inner product).
  - Milvus remplace la base SQL classique pour les requêtes de similarité.

### 6. Requêtes et recommandations

1. Transformer la requête utilisateur en vecteur (`SentenceTransformer.encode`).
2. Rechercher les voisins les plus proches dans Milvus.
3. Filtrer selon les contraintes métier (ex. `user_rating > 8`, genre ciblé).
4. Retourner la liste d’albums recommandés avec leurs métadonnées.

## ⚙️ Architecture technique globale

```
  SCRAPING (HTML)
- ↓
  DATAFRAME (métadonnées normalisées)
- ↓
  EMBEDDING (SentenceTransformer)
- ↓
  MILVUS (index vectoriel)
- ↓
  FASTAPI (API de requête)
- ↓
  INTERFACE (Streamlit / app utilisateur)
```

## 👥 Rôles et responsabilités

| Membre            | Rôle | Missions principales |
| ----------------- | ---- | -------------------- |
| **Arthur Muller** |
| **Semih Taskin**  |
| **Abdoulaye**     |

> ℹ️ La branche `main` est protégée. Chaque membre travaille sur une branche `feature/...` avec revue de code et PR.

## 🧱 Prochaines étapes (Sprint 1)

| Étape | Description                                                     | Responsable |
| ----- | --------------------------------------------------------------- | ----------- |
| 1️⃣    | Implémenter le scraper `guts_scraper.py` (HTML → CSV)           | Semih       |
| 2️⃣    | Nettoyer et structurer les données (`data_cleaner.py`)          | Semih       |
| 3️⃣    | Générer les embeddings texte (`vectorization/text_embedder.py`) | Arthur      |
| 4️⃣    | Créer la collection Milvus insertion des vecteurs               | Arthur      |
| 5️⃣    | Tester la première requête de similarité                        | Abdoulaye   |

## 📘 Agents

Chaque dossier de code contient (ou contiendra) un fichier `agent.md` qui :

- décrit la **mission du module** ;
  - précise ses **entrées/sorties** ;
  - liste les **librairies utilisées** ;
  - rappelle les **points d’attention techniques**.
    Ces fichiers servent de **documentation vivante**. À chaque création ou modification d’un fichier Python, mettre à jour l’`agent.md` correspondant pour conserver une traçabilité claire des choix techniques.
