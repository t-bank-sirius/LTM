from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, 
    Filter, 
    FieldCondition, MatchValue
)
import uuid
import os
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "ltm_memories")
        self.client = None
        
    async def initialize(self, vector_size: int):
        try:
            logger.info(f"Подключаюсь к Qdrant: {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                logger.info(f"Создаю коллекцию: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Коллекция создана успешно")
            else:
                logger.info(f"Коллекция {self.collection_name} уже существует")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Qdrant: {e}")
            raise
    
    async def store_vector(
        self, 
        user_id: str,
        content: str, 
        vector: np.ndarray,
        context: Optional[str] = None
    ) -> str:
        if self.client is None:
            raise RuntimeError("Клиент Qdrant не инициализирован")
        
        try:
            point_id = str(uuid.uuid4())
            
            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "user_id": user_id,
                "content": content,
                "time": current_time
            }
            
            if context:
                payload["context"] = context
            
            point = PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Вектор сохранен с ID: {point_id} для пользователя: {user_id}")
            return point_id
        except Exception as e:
            logger.error(f"Ошибка при сохранении вектора: {e}")
            raise

    async def search_similar(
        self, 
        user_id: str,
        query_vector: np.ndarray, 
        limit: int = 5, 
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Клиент Qdrant не инициализирован")
        
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score
            )
            
            results = []
            for scored_point in search_result:
                result = {
                    "score": scored_point.score,
                    "user_id": scored_point.payload.get("user_id", ""),
                    "content": scored_point.payload.get("content", ""),
                    "context": scored_point.payload.get("context"),
                    "time": scored_point.payload.get("time")
                }
                results.append(result)
            
            logger.info(f"Найдено {len(results)} результатов для пользователя {user_id}")
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise
