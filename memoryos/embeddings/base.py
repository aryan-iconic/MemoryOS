"""Base embedding classes."""
import numpy as np
import logging
import base64
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into their corresponding embeddings."""
        pass

    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts based on their embeddings."""
        pass

    def encode_embedding(self, embedding: np.ndarray) -> str:
        """Encode a numpy array embedding into a base64 string."""
        return base64.b64encode(embedding.tobytes()).decode('utf-8')

    def decode_embedding(self, encoded: str, dtype=np.float32) -> np.ndarray:
        """Decode a base64 string back into a numpy array embedding."""
        byte_data = base64.b64decode(encoded)
        return np.frombuffer(byte_data, dtype=dtype)