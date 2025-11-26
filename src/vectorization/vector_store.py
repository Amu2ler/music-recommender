"""
Gestion du Vector Store Milvus
Création de collections, insertion et recherche de similarité
"""

import sys
from typing import List, Dict, Optional

import pandas as pd
from loguru import logger
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)


class MilvusStore:
    """Wrapper pour les opérations Milvus"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "music_embeddings",
        embedding_dim: int = 384
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None
        
    def connect(self):
        """Connexion à Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.success(f"✅ Connecté à Milvus ({self.host}:{self.port})")
        except Exception as e:
            logger.error(f"Erreur connexion Milvus : {e}")
            raise
    
    def create_collection(self, drop_existing: bool = False):
        """
        Crée une collection Milvus pour les embeddings musicaux
        
        Args:
            drop_existing: Si True, supprime la collection existante
        """
        # Supprimer l'ancienne collection si demandé
        if drop_existing and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"Collection '{self.collection_name}' supprimée")
        
        # Si la collection existe déjà, la charger
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' chargée ({self.collection.num_entities} entités)")
            return self.collection
        
        # Créer le schéma
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="album_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="artist_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="styles", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="note_chronique", dtype=DataType.FLOAT),
            FieldSchema(name="note_moyenne", dtype=DataType.FLOAT),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        
        schema = CollectionSchema(fields, description="Embeddings musicaux Guts of Darkness")
        
        # Créer la collection
        self.collection = Collection(self.collection_name, schema=schema)
        logger.success(f"✅ Collection '{self.collection_name}' créée")
        
        return self.collection
    
    def create_index(self, metric_type: str = "COSINE"):
        """
        Crée un index pour les recherches de similarité
        
        Args:
            metric_type: Type de métrique (COSINE, L2, IP)
        """
        if self.collection is None:
            raise ValueError("Collection non initialisée. Appelez create_collection() d'abord.")
        
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.success(f"✅ Index créé (métrique: {metric_type})")
    
    def insert_from_dataframe(self, df: pd.DataFrame):
        """
        Insère des données depuis un DataFrame
        
        Args:
            df: DataFrame avec colonnes : album_name, artist_name, embedding, etc.
        """
        if self.collection is None:
            raise ValueError("Collection non initialisée")
        
        # Préparer les données
        data = []
        for col in ["album_name", "artist_name", "styles", "note_chronique", 
                    "note_moyenne", "source_url", "embedding"]:
            if col in df.columns:
                if col == "embedding":
                    # S'assurer que les embeddings sont des listes
                    data.append(df[col].tolist())
                else:
                    # Remplir les NaN
                    if col in ["note_chronique", "note_moyenne"]:
                        data.append(df[col].fillna(0.0).tolist())
                    else:
                        data.append(df[col].fillna("").astype(str).tolist())
            else:
                logger.warning(f"Colonne '{col}' manquante, utilisation de valeurs par défaut")
                if col == "embedding":
                    raise ValueError("Colonne 'embedding' requise")
                elif col in ["note_chronique", "note_moyenne"]:
                    data.append([0.0] * len(df))
                else:
                    data.append([""] * len(df))
        
        # Insérer
        self.collection.insert(data)
        self.collection.flush()
        
        logger.success(f"✅ {len(df)} embeddings insérés dans Milvus")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Recherche les albums les plus similaires
        
        Args:
            query_embedding: Vecteur de requête
            top_k: Nombre de résultats
            output_fields: Champs à retourner
        
        Returns:
            Liste de résultats avec métadonnées
        """
        if self.collection is None:
            raise ValueError("Collection non initialisée")
        
        if output_fields is None:
            output_fields = ["album_name", "artist_name", "styles", "note_chronique", "source_url"]
        
        # Charger la collection en mémoire
        self.collection.load()
        
        # Paramètres de recherche
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        # Effectuer la recherche
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        # Formater les résultats
        formatted = []
        for hits in results:
            for hit in hits:
                result = {
                    "distance": hit.distance,
                    **{field: hit.entity.get(field) for field in output_fields}
                }
                formatted.append(result)
        
        return formatted
    
    def disconnect(self):
        """Déconnexion de Milvus"""
        connections.disconnect("default")
        logger.info("Déconnecté de Milvus")


if __name__ == "__main__":
    logger.add(sys.stderr, level="INFO")
    
    # Test de connexion
    store = MilvusStore()
    store.connect()
    store.create_collection()
    
    print("\n✅ Milvus configuré avec succès")

