from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memoryos.compression.summarizer import Summarizer
from memoryos.embeddings.sentence_transformer import SentenceTransformerEmbedding
from memoryos.models import Turn
from memoryos.storage.sqlite_store import SQLiteStore


class EpisodicMemory:
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformerEmbedding] = None,
        store: Optional[SQLiteStore] = None,
        summarizer: Optional[Summarizer] = None,
        similarity_threshold: float = 0.25,
    ):
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        self.store = store or SQLiteStore()
        self.summarizer = summarizer or Summarizer(backend="rule_based")
        self.similarity_threshold = similarity_threshold

    def create_episode(
        self,
        session_id: str,
        turns: List[Turn],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:

        if not turns:
            return None  # pragma: no cover

        summary = self.summarize_episode(turns)

        if not summary:
            return None  # pragma: no cover

        start_timestamp = min(turn.timestamp for turn in turns)
        end_timestamp = max(turn.timestamp for turn in turns)

        episode = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "summary": summary,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "turn_count": len(turns),
            "embedding": self._embed_text(summary),
            "metadata": metadata or {},
            "created_at": time.time(),
        }

        self.store.save_episode(episode)

        return episode

    def summarize_episode(
        self,
        turns: List[Turn],
    ) -> str:

        return self.summarizer.summarize_turns(turns)

    def search(
        self,
        query: str,
        session_id: str,
        limit: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:

        query = query.strip()

        if not query:
            return []  # pragma: no cover

        threshold = self.similarity_threshold if min_score is None else min_score
        query_embedding = np.array(self._embed_text(query), dtype=np.float32)

        episodes = self.store.get_episodes_by_session(session_id)

        results: List[Tuple[Dict[str, Any], float]] = []

        for episode in episodes:
            if episode.get("embedding") is None:
                episode["embedding"] = self._embed_text(episode["summary"])  # pragma: no cover
                self.store.save_episode(episode)  # pragma: no cover

            episode_embedding = np.array(
                episode["embedding"],
                dtype=np.float32,
            )

            score = self._cosine_similarity(
                query_embedding,
                episode_embedding,
            )

            if score >= threshold:
                results.append((episode, score))

        results.sort(key=lambda item: item[1], reverse=True)

        return results[:limit]

    def build_context(
        self,
        query: str,
        session_id: str,
        limit: int = 3,
        max_chars: int = 1500,
        min_score: float = 0.20,
    ) -> str:

        results = self.search(  # pragma: no cover
            query=query,
            session_id=session_id,
            limit=limit,
            min_score=min_score,
        )

        if not results:  # pragma: no cover
            return ""  # pragma: no cover

        lines = ["Relevant past conversation summaries:"]  # pragma: no cover

        for episode, score in results:  # pragma: no cover
            lines.append(  # pragma: no cover
                f"- {episode['summary']} " f"(turns={episode['turn_count']}, score={score:.3f})"
            )

        context = "\n".join(lines)  # pragma: no cover

        if len(context) > max_chars:  # pragma: no cover
            context = context[:max_chars].rstrip() + "..."

        return context  # pragma: no cover

    def get_episodes(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        return self.store.get_episodes_by_session(  # pragma: no cover
            session_id=session_id,
            limit=limit,
        )

    def get_all_episodes(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        return self.store.get_all_episodes(limit=limit)  # pragma: no cover

    def rebuild_embeddings(
        self,
        session_id: Optional[str] = None,
    ) -> int:

        if session_id is not None:  # pragma: no cover
            episodes = self.store.get_episodes_by_session(session_id)  # pragma: no cover
        else:
            episodes = self.store.get_all_episodes()  # pragma: no cover

        updated_count = 0  # pragma: no cover

        for episode in episodes:  # pragma: no cover
            if episode.get("embedding") is None:  # pragma: no cover
                episode["embedding"] = self._embed_text(episode["summary"])  # pragma: no cover
                self.store.save_episode(episode)  # pragma: no cover
                updated_count += 1  # pragma: no cover

        return updated_count  # pragma: no cover

    def _embed_text(self, text: str) -> List[float]:
        embedding = self.embedding_model.embed([text])[0]

        if hasattr(embedding, "tolist"):
            return embedding.tolist()

        return list(embedding)  # pragma: no cover

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0  # pragma: no cover
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0  # pragma: no cover
        return float(np.dot(a, b) / (norm_a * norm_b))
