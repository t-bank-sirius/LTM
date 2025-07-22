from typing import List, Dict, Optional
import logging
from collections import Counter
import math

from .embedding_service import EmbeddingService
from .qdrant_service import QdrantService
from ..models.schemas import MemoryResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ltm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryService:
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return
            
        try:
            logger.info("Инициализация сервиса памяти...")
            await self.embedding_service.initialize()
            await self.qdrant_service.initialize(1024)
            self.initialized = True
            logger.info("Сервис памяти успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации сервиса памяти: {e}")
            raise
    
    async def store_memory(
        self,
        user_id: str,
        content: str,
        context: Optional[str] = None
    ) -> Dict[str, str]:
        if not self.initialized:
            raise RuntimeError("Сервис не инициализирован")
        
        try:
            logger.info(f"Сохраняю в память для пользователя {user_id}: {content[:100]}...")
            embedding = self.embedding_service.encode_text_with_context(content, context)
            memory_id = await self.qdrant_service.store_vector(
                user_id=user_id,
                content=content,
                vector=embedding,
                context=context
            )
            logger.info(f"Память сохранена с ID: {memory_id}")
            return {"id": memory_id}
        except Exception as e:
            logger.error(f"Ошибка при сохранении памяти: {e}")
            raise
    
    async def search_memory(
        self,
        user_id: str,
        query: str,
        limit: int = 5, 
        min_score: float = 0.3
    ) -> List[MemoryResponse]:
        if not self.initialized:
            raise RuntimeError("Сервис не инициализирован")
        
        try:
            logger.info(f"Поиск в памяти пользователя {user_id}: {query}")
            query_embedding = self.embedding_service.encode_query(query)
            
            search_results = await self.qdrant_service.search_similar(
                user_id=user_id,
                query_vector=query_embedding,
                limit=limit,
                min_score=min_score
            )
            
            memories = []
            for result in search_results:
                try:
                    memory = MemoryResponse(
                        user_id=result["user_id"],
                        content=result["content"],
                        score=result["score"],
                        context=result.get("context"),
                        time=result.get("time")
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.error(f"Ошибка при обработке результата {result.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Найдено {len(memories)} воспоминаний для пользователя {user_id}")
            return memories
        except Exception as e:
            logger.error(f"Ошибка при поиске в памяти: {e}")
            raise

class FaceService:
    def __init__(self):
        self.faces = {}
        self.counter = 0

    async def add_face(self, user_id: str, name: str, image: str):
        self.counter += 1
        face_id = str(self.counter)
        if user_id not in self.faces:
            self.faces[user_id] = []
        self.faces[user_id].append({
            "id": face_id,
            "name": name,
            "image": image
        })
        return face_id

    @staticmethod
    def cosine_similarity(a: str, b: str) -> float:
        a_words = Counter(a.lower().split())
        b_words = Counter(b.lower().split())
        all_words = set(a_words) | set(b_words)
        v1 = [a_words.get(w, 0) for w in all_words]
        v2 = [b_words.get(w, 0) for w in all_words]
        dot = sum(x*y for x, y in zip(v1, v2))
        norm1 = math.sqrt(sum(x*x for x in v1))
        norm2 = math.sqrt(sum(x*x for x in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def find_face(self, user_id: str, query: str, min_score: float = 0.7):
        faces = self.faces.get(user_id, [])
        best = None
        best_score = 0.0
        for face in faces:
            score = self.cosine_similarity(face["name"], query)
            if score > best_score:
                best_score = score
                best = face
        if best and best_score >= min_score:
            return {"image": best["image"], "score": best_score}
        return None
