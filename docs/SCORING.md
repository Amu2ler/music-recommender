# Système de Score - Music Recommender

## Métrique actuelle : L2 (Distance Euclidienne)

### Comment lire le score L2 ?
- **0.0** = Match parfait (identique)
- **0.0 - 0.5** = Très similaire ⭐⭐⭐⭐⭐
- **0.5 - 1.0** = Similaire ⭐⭐⭐⭐
- **1.0 - 2.0** = Moyennement similaire ⭐⭐⭐
- **> 2.0** = Peu similaire ⭐⭐

**Plus le score est BAS, meilleur est le match.**

## Alternative : Similarité Cosinus (0-1)

Si tu préfères un score où **1 = très similaire** et **0 = pas similaire**, tu peux changer la métrique.

### Pour changer vers Cosinus :

1. **Modifier `scripts/load_to_milvus.py` et `scripts/process_full_dataset.py`** :
```python
# Remplacer
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",  # ← Changer ici
    "params": {"nlist": 1024}
}

# Par
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # Inner Product (similaire à cosinus)
    "params": {"nlist": 1024}
}
```

2. **Modifier `api/main.py`** :
```python
# Remplacer
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# Par
search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
```

3. **Recharger les données** :
```powershell
python scripts/load_to_milvus.py
```

### Avec IP (Inner Product) :
- **1.0** = Match parfait
- **0.8 - 1.0** = Très similaire ⭐⭐⭐⭐⭐
- **0.6 - 0.8** = Similaire ⭐⭐⭐⭐
- **0.4 - 0.6** = Moyennement similaire ⭐⭐⭐
- **< 0.4** = Peu similaire ⭐⭐

**Plus le score est HAUT, meilleur est le match.**

## Quelle métrique choisir ?

- **L2** : Bonne pour la distance absolue entre vecteurs
- **IP/Cosinus** : Meilleure pour la similarité directionnelle (recommandé pour du texte)

Pour ton cas (recommandation musicale basée sur du texte), **IP est généralement préférable** car elle mesure l'orientation des vecteurs plutôt que leur distance absolue.
