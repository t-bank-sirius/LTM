from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.model = None
        self.vector_size = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def initialize(self):
        try:
            logger.info(f"Загружаю модель эмбеддингов: {self.model_name}")
            logger.info(f"Устройство: {self.device}")
            if self.device == "cuda":
                logger.info(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            test_embedding = self.model.encode("test")
            self.vector_size = len(test_embedding)
            logger.info(f"Модель загружена успешно. Размер вектора: {self.vector_size}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def encode_text_with_context(self, content: str, context: str = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Вызовите initialize() сначала.")
        
        try:
            if context:
                combined_text = f"passage: {content} context: {context}"
                logger.info(f"Кодирую документ с контекстом: {content[:50]}...")
                return self.encode_text(combined_text)
            else:
                prefixed_content = f"passage: {content}"
                return self.encode_text(prefixed_content)
        except Exception as e:
            logger.error(f"Ошибка при создании составного эмбеддинга: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Вызовите initialize() сначала.")
        
        try:
            prefixed_query = f"query: {query}"
            logger.info(f"Кодирую запрос: {query}")
            return self.encode_text(prefixed_query)
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса: {e}")
            raise
 
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Вызовите initialize() сначала.")
        
        try:
            embeddings = self.model.encode(text, convert_to_tensor=False)
            
            if isinstance(embeddings, np.ndarray):
                return embeddings
            else:
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            raise
    
    
    def similarity(self, text1: str, text2: str) -> float:
        try:
            embeddings = self.encode_text([text1, text2])
            
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            cosine_sim = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
            return float((cosine_sim + 1) / 2)
        except Exception as e:
            logger.error(f"Ошибка при вычислении сходства: {e}")
            return 0.0 