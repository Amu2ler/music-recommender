# 🎵 Music Recommender - Start-up Guide

## 1. Démarrer la stack Milvus
Assurez-vous que Docker Desktop est lancé.
```powershell
docker compose up -d
```
Vérifiez que les conteneurs tournent :
```powershell
docker compose ps
```

## 2. Charger les embeddings dans Milvus
Ce script va créer la collection et charger les données depuis le fichier Parquet.
```powershell
python scripts/load_to_milvus.py
```

## 3. Lancer l’API
L'API sera accessible sur `http://localhost:8000`.
```powershell
uvicorn api.main:app --reload
```
Documentation interactive (Swagger) : `http://localhost:8000/docs`

## 4. Lancer l’UI
L'interface sera accessible sur `http://localhost:8501`.
```powershell
streamlit run ui/app.py
```

## Configuration
Les variables d'environnement suivantes peuvent être définies (valeurs par défaut indiquées) :
- `MILVUS_HOST`: `localhost`
- `MILVUS_PORT`: `19530`
- `API_URL`: `http://localhost:8000`
