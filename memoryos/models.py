from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import time
import uuid


FactType = Literal["identity", "preference", "goal", "decision", "context"]
FactSource = Literal["conversation", "manual", "system"]
MemorySource = Literal["working", "episodic", "semantic"]


@dataclass
class Fact:
    content: str
    type: FactType
    confidence: float
    session_id: str

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: FactSource = "conversation"
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Fact":
        return Fact(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            type=data["type"],
            confidence=data["confidence"],
            session_id=data["session_id"],
            source=data.get("source", "conversation"),
            timestamp=data.get("timestamp", time.time()),
            access_count=data.get("access_count", 0),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Turn:
    user_message: str
    ai_response: str
    session_id: str

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        return f"User: {self.user_message}\nAI: {self.ai_response}"


@dataclass
class MemorySearchResult:
    content: str
    source: MemorySource
    score: float

    id: Optional[str] = None
    type: Optional[FactType] = None
    confidence: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryOSConfig:
    working_memory_size: int = 8

    min_fact_confidence: float = 0.65
    duplicate_similarity_threshold: float = 0.90

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    semantic_top_k: int = 5
    episodic_top_k: int = 3

    max_context_tokens: int = 6000
    response_buffer_tokens: int = 1000

    working_memory_max_ratio: float = 0.60
    facts_max_ratio: float = 0.50
    summaries_max_ratio: float = 0.30

    db_path: str = "memoryos.db"
    faiss_index_path: str = "memoryos.faiss"