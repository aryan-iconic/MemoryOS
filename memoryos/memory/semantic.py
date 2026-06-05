"""Semantic long-term memory backed by persisted facts and embeddings."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from memoryos.embeddings.sentence_transformer import SentenceTransformerEmbedding
from memoryos.models import Fact, MemorySearchResult
from memoryos.storage.sqlite_store import SQLiteStore


class SemanticMemory:
    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        embedder: Optional[SentenceTransformerEmbedding] = None,
        similarity_threshold: float = 0.35,
    ):
        self.store = store
        self.embedder = embedder or SentenceTransformerEmbedding()
        self.similarity_threshold = similarity_threshold

    def add_fact(self, fact: Fact) -> Fact:
        store = self._require_store()
        if not getattr(fact, "content", None):
            raise ValueError("Cannot add Fact without content.")
        fact.embedding = self._embed_text(fact.content)
        saved_fact = store.save_fact(fact)
        return saved_fact or fact

    def add_facts(self, facts: List[Fact]) -> List[Fact]:
        return [self.add_fact(fact) for fact in facts]

    def search(
        self,
        query: str,
        top_k: int = 5,
        fact_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        query_embedding = self._embed_text(query)
        facts = self._load_facts(fact_type=fact_type, session_id=session_id)
        scored_results: List[tuple[Fact, float]] = []
        threshold = self.similarity_threshold if min_score is None else min_score

        for fact in facts:
            if not getattr(fact, "content", None):
                continue  # pragma: no cover
            if self._is_missing_embedding(getattr(fact, "embedding", None)):
                fact.embedding = self._embed_text(fact.content)
                self._save_existing_fact(fact)
            fact_embedding = self._to_float_list(fact.embedding)
            score = self._cosine_similarity(query_embedding, fact_embedding)
            if score >= threshold:
                scored_results.append((fact, score))

        scored_results.sort(key=lambda item: item[1], reverse=True)
        return [
            MemorySearchResult(
                content=fact.content,
                source="semantic",
                score=score,
                id=fact.id,
                timestamp=fact.timestamp,
                metadata={
                    "session_id": fact.session_id,
                    "fact_type": fact.type,
                    "match_type": "semantic",
                    "original_confidence": fact.confidence,
                    "similarity_score": score,
                },
            )
            for fact, score in scored_results[:top_k]
        ]

    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        return self._require_store().get_all_facts(limit=limit)

    def rebuild_embeddings(self, batch_size: int = 100) -> None:
        store = self._require_store()
        facts = store.get_all_facts()
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            contents = [fact.content for fact in batch]
            embeddings = self.embedder.embed(contents)
            for fact, embedding in zip(batch, embeddings):
                fact.embedding = self._to_float_list(embedding)
                self._save_existing_fact(fact)

    def _load_facts(
        self,
        fact_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Fact]:
        store = self._require_store()
        if session_id is not None:
            facts = store.get_facts_by_session(session_id)
        elif fact_type is not None:
            facts = store.get_facts_by_type(fact_type)  # pragma: no cover
        else:
            facts = store.get_all_facts()
        if fact_type is not None and session_id is not None:
            facts = [fact for fact in facts if fact.type == fact_type]  # pragma: no cover
        return facts

    def _embed_text(self, text: str) -> List[float]:
        embedding = self.embedder.embed([text])[0]
        return self._to_float_list(embedding)

    def _to_float_list(self, embedding: Any) -> List[float]:
        if embedding is None:
            return []  # pragma: no cover
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        return [float(value) for value in embedding]

    def _is_missing_embedding(self, embedding: Any) -> bool:
        if embedding is None:
            return True  # pragma: no cover
        try:
            return len(embedding) == 0
        except TypeError:  # pragma: no cover
            return True  # pragma: no cover

    def _cosine_similarity(self, vec1: Sequence[float], vec2: Sequence[float]) -> float:
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _save_existing_fact(self, fact: Fact) -> Fact:
        saved_fact = self._require_store().save_fact(fact)
        return saved_fact or fact

    def _require_store(self) -> SQLiteStore:
        if self.store is None:
            raise ValueError("SemanticMemory requires a SQLiteStore instance.")
        return self.store
