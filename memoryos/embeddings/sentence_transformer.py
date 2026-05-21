"""Sentence Transformer embedding implementation."""
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.embed([text1])[0]
        embedding2 = self.embed([text2])[0]
        return self.cosine_similarity(embedding1, embedding2)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    