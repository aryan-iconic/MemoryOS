from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from memoryos.compression import CompressionConfig, Compressor, TokenBudgetManager
from memoryos.compression.summarizer import (
    CallableSummarizer,
    LocalHTTPSummarizer,
    RuleBasedSummarizer,
    Summarizer,
    SummaryConfig,
)
from memoryos.config import MemoryOSConfig
from memoryos.exceptions import (
    ConfigError,
    DependencyNotInstalledError,
    IndexBackendError,
    MemoryOSError,
)
from memoryos.interfaces import (
    BaseEmbeddingProvider,
    RankerInterface,
    StorageBackend,
    SummarizerInterface,
)
from memoryos.memory.semantic import SemanticMemory
from memoryos.memory.working import WorkingMemory
from memoryos.models import Fact, MemorySearchResult, Turn
from memoryos.retrieval.retriever import MemoryRetriever
from memoryos.retrieval.scorer import RetrievalScorer
from memoryos.storage.faiss_index import FAISSVectorIndex
from memoryos.storage.index import InMemoryVectorIndex, VectorRecord, cosine_similarity
from memoryos.storage.migrations import migrate_database
from memoryos.storage.sqlite_store import SQLiteStore


class TinyEmbedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "dark" in lowered else 0.1,
                    1.0 if "memoryos" in lowered else 0.1,
                    1.0 if "legal" in lowered else 0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class FailingTurnSummarizer:
    def summarize_turns(self, turns: Sequence[Any]) -> str:
        raise RuntimeError("turn path failed")

    def summarize_texts(self, texts: Sequence[str]) -> str:
        return " ".join(texts)


def test_compressor_covers_text_turn_fact_paths() -> None:
    compressor = Compressor(
        summarizer=FailingTurnSummarizer(),
        config=CompressionConfig(
            max_input_tokens=8,
            max_output_words=12,
            chunk_size_turns=2,
            overlap_turns=1,
            min_turns_to_compress=2,
            preserve_recent_turns=1,
            include_timestamps=True,
            include_metadata=True,
            empty_summary="EMPTY",
        ),
    )
    turns = [
        Turn(
            "My name is Aryan",
            "Saved",
            "s",
            timestamp=1.0,
            metadata={"topic": "identity"},
        ),
        Turn("I prefer dark UI", "Saved", "s", timestamp=2.0),
        Turn("I want to build MemoryOS", "Good", "s", timestamp=3.0),
    ]

    assert compressor.should_compress(turns)
    assert not compressor.should_compress([], min_turns=1)
    assert len(compressor.chunk_turns(turns, chunk_size=2, overlap=1)) == 2
    old, recent = compressor.split_for_compression(turns)
    assert len(old) == 2 and len(recent) == 1

    text = compressor.turn_to_text(
        {
            "user_message": " hi ",
            "ai_response": " there ",
            "timestamp": 1.0,
            "metadata": {"project": "x"},
        }
    )
    assert "Timestamp" in text and "project=x" in text

    turn_result = compressor.compress_turns(turns, session_id="s", metadata={"source": "test"})
    assert turn_result.summary
    assert turn_result.source_type == "turns"
    assert turn_result.metadata["session_id"] == "s"
    assert not turn_result.is_empty()
    assert turn_result.to_dict()["summary"] == turn_result.summary

    text_result = compressor.compress_texts(["", "one two three", "four five six seven eight nine"])
    assert text_result.source_type == "texts"
    assert text_result.compressed_count == 1

    fact_result = compressor.compress_facts(
        [
            Fact("User prefers dark UI", "preference", 0.9, "s"),
            {"content": "User builds MemoryOS", "type": "goal", "confidence": 0.8},
        ]
    )
    assert fact_result.source_type == "facts"

    empty = compressor.compress_texts([])
    assert empty.summary == "EMPTY"
    assert empty.source_type == "texts"
    assert compressor.clean_text(" a   b ") == "a b"
    assert compressor.summary_id("same") == compressor.summary_id("same")
    assert compressor._compression_ratio(0, 10) == 0.0


def test_summarizers_and_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = TokenBudgetManager(max_tokens=3)
    assert budget.can_add(2)
    assert budget.add_tokens(2)
    assert not budget.add_tokens(2)
    budget.reset()
    assert budget.current_tokens == 0

    rule = RuleBasedSummarizer(SummaryConfig(max_words=8, max_input_words=30))
    assert "Aryan" in rule.summarize_texts(["My name is Aryan. Temporary hello."])
    assert rule.summarize_texts([]) == ""
    assert rule.summarize_turns([Turn("I prefer dark UI", "Saved", "s")])
    assert rule._split_sentences("Hi? Yes!") == ["Hi.", "Yes."]

    called: List[str] = []

    def fn(prompt: str) -> str:
        called.append(prompt)
        return "custom summary"

    callable_summarizer = Summarizer(backend="callable", callable_fn=fn)
    assert callable_summarizer.summarize_texts(["hello"]) == "custom summary"
    assert callable_summarizer.summarize_turns([Turn("hello", "world", "s")]) == "custom summary"
    assert called

    failing_callable = CallableSummarizer(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert failing_callable.summarize_texts(["I want MemoryOS"])  # fallback
    assert failing_callable.summarize_turns([Turn("I want MemoryOS", "ok", "s")])

    memory_summarizer = Summarizer(TokenBudgetManager(max_tokens=2))
    assert memory_summarizer.add_to_summary("one")
    assert not memory_summarizer.add_to_summary("two three")
    assert memory_summarizer.get_summary() == "one"
    memory_summarizer.reset()
    assert memory_summarizer.get_summary() == ""
    assert memory_summarizer.generate_summary(["I decided MemoryOS stays model agnostic"])

    with pytest.raises(ValueError):
        Summarizer(backend="unsupported")
    with pytest.raises(ValueError):
        Summarizer(backend="callable")
    with pytest.raises(ValueError):
        LocalHTTPSummarizer(SummaryConfig(backend="local_http"))

    local = LocalHTTPSummarizer(SummaryConfig(backend="local_http", model="m", endpoint="http://unused"))
    monkeypatch.setattr(local, "_call_local_http", lambda prompt: "remote summary")
    assert local.summarize_texts(["hello"]) == "remote summary"
    assert local.summarize_turns([Turn("hello", "world", "s")]) == "remote summary"


def test_config_exceptions_and_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MemoryOSConfig.from_dict({"db_path": str(tmp_path / "x.db"), "unknown": "ignored"})
    assert config.db_path.endswith("x.db")
    config.ensure_paths()
    assert Path(config.db_path).parent.exists()

    monkeypatch.setenv("MEMORYOS_WORKING_MEMORY_SIZE", "4")
    monkeypatch.setenv("MEMORYOS_ENABLE_FAISS", "yes")
    assert MemoryOSConfig.from_env().working_memory_size == 4
    assert MemoryOSConfig.from_env().enable_faiss is True

    with pytest.raises(ConfigError):
        MemoryOSConfig(working_memory_size=0).validate()
    with pytest.raises(ConfigError):
        MemoryOSConfig(min_fact_confidence=2.0).validate()
    with pytest.raises(ConfigError):
        MemoryOSConfig(working_memory_max_ratio=2.0).validate()
    with pytest.raises(ConfigError):
        MemoryOSConfig(db_path="").validate()
    with pytest.raises(ConfigError):
        MemoryOSConfig(storage_backend="bad").validate()
    with pytest.raises(ConfigError):
        MemoryOSConfig(vector_backend="bad").validate()
    monkeypatch.setenv("BAD_ENABLE_FAISS", "maybe")
    with pytest.raises(ConfigError):
        MemoryOSConfig.from_env(prefix="BAD_")

    err = MemoryOSError("bad", details={"x": 1})
    assert "details" in str(err)
    assert "bad" in str(MemoryOSError("bad"))

    score = RetrievalScorer().score_fact("I prefer reliable systems", 0.8)
    assert 0.0 <= score <= 1.0


def test_working_semantic_retriever_and_index_paths(tmp_path: Path) -> None:
    working = WorkingMemory(MemoryOSConfig(working_memory_size=2))
    assert working.get_context_text() == ""
    assert working.build_context() == ""
    t1 = Turn("hello dark UI", "ok", "s")
    t2 = Turn("MemoryOS project", "ok", "s")
    working.add_turn(t1)
    working.add_turn(t2)
    assert len(working) == 2
    assert working.build_context(max_chars=20).startswith("...")
    assert "Recent conversation" in working.build_context(max_chars=500)
    assert working.search("dark")
    assert working.search("") == []

    store = SQLiteStore(str(tmp_path / "semantic.db"))
    semantic = SemanticMemory(store=store, embedder=TinyEmbedder(), similarity_threshold=0.0)
    with pytest.raises(ValueError):
        SemanticMemory(store=None, embedder=TinyEmbedder()).get_all_facts()
    with pytest.raises(ValueError):
        semantic.add_fact(Fact("", "context", 0.8, "s"))
    fact = semantic.add_fact(Fact("User prefers dark UI", "preference", 0.9, "s"))
    fact.embedding = []
    store.save_fact(fact)
    results = semantic.search("dark UI", min_score=0.0)
    assert results and results[0].source == "semantic"
    semantic.rebuild_embeddings(batch_size=1)
    assert semantic.get_all_facts()
    assert semantic.search("") == []
    assert semantic._cosine_similarity([], [1.0]) == 0.0
    assert semantic._cosine_similarity([0.0], [1.0]) == 0.0

    retriever = MemoryRetriever(working_memory=working, semantic_memory=semantic, episodic_memory=None)
    found = retriever.retrieve("dark", session_id="s", min_score=0.0, include_episodic=False)
    assert found
    assert retriever.retrieve("   ") == []
    assert retriever._normalize_result({"content": "x", "score": 0.3}, source="semantic").content == "x"
    with pytest.raises(Exception):
        retriever._normalize_result(object(), source="semantic")

    index = InMemoryVectorIndex(dim=2, persist_path=str(tmp_path / "idx.json"))
    index.add_many([VectorRecord("a", [1, 0]), VectorRecord("b", [0, 1])])
    assert index.search([1, 0], top_k=0) == []
    assert index.search([1, 0], min_score=0.1)[0].id == "a"
    index.delete("missing")
    index.save()
    loaded = InMemoryVectorIndex()
    loaded.load(str(tmp_path / "idx.json"))
    assert len(loaded) == 2
    assert cosine_similarity([1, 0], [0, 0]) == 0.0
    with pytest.raises(IndexBackendError):
        index.add("bad", [])
    with pytest.raises(IndexBackendError):
        index.add("bad", [1, 2, 3])
    with pytest.raises(IndexBackendError):
        InMemoryVectorIndex().load(str(tmp_path / "missing.json"))


class FakeFaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = np.empty((0, dim), dtype=np.float32)

    def add(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def search(self, query: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        if self.matrix.size == 0:
            return np.asarray([[]], dtype=np.float32), np.asarray([[]], dtype=np.int64)
        scores = self.matrix @ query[0]
        order = np.argsort(-scores)[:limit]
        return scores[order].reshape(1, -1), order.reshape(1, -1)


class FakeFaiss:
    @staticmethod
    def IndexFlatIP(dim: int) -> FakeFaissIndex:
        return FakeFaissIndex(dim)

    @staticmethod
    def write_index(index: FakeFaissIndex, path: str) -> None:
        Path(path).write_text("fake", encoding="utf-8")

    @staticmethod
    def read_index(path: str) -> FakeFaissIndex:
        return FakeFaissIndex(2)


def test_faiss_index_fallback_and_fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FAISSVectorIndex, "_import_faiss", staticmethod(lambda: FakeFaiss))
    index = FAISSVectorIndex(dim=2, persist_path=str(tmp_path / "faiss.index"))
    assert index.search([1, 0]) == []
    index.add("a", [1, 0], {"label": "a"})
    index.add("b", [0, 1])
    assert index.search([1, 0], top_k=1)[0].id == "a"
    index.delete("missing")
    index.delete("b")
    assert len(index) == 1
    index.save()
    loaded = FAISSVectorIndex(dim=2, persist_path=str(tmp_path / "faiss.index"))
    loaded.load()
    assert loaded.ids == ["a"]
    index.clear()
    assert len(index) == 0
    with pytest.raises(IndexBackendError):
        index.add("bad", [1, 2, 3])
    with pytest.raises(IndexBackendError):
        index.add("zero", [0, 0])
    with pytest.raises(IndexBackendError):
        loaded.load(str(tmp_path / "missing.index"))

    monkeypatch.setattr(
        FAISSVectorIndex,
        "_import_faiss",
        staticmethod(lambda: (_ for _ in ()).throw(DependencyNotInstalledError("missing"))),
    )
    with pytest.raises(DependencyNotInstalledError):
        FAISSVectorIndex(dim=2)


def test_migrations_and_interfaces(tmp_path: Path) -> None:
    old = SQLiteStore(str(tmp_path / "old.db"))
    old.save_fact(Fact("User prefers dark UI", "preference", 0.9, "s"))
    old.save_turn(Turn("I prefer dark UI", "Saved", "s"))
    old.save_episode(
        {
            "id": "ep1",
            "session_id": "s",
            "summary": "User discussed dark UI.",
            "start_timestamp": 1.0,
            "end_timestamp": 2.0,
            "turn_count": 1,
            "embedding": [1.0, 0.0],
            "metadata": {"x": 1},
        }
    )
    old.close()
    counts = migrate_database(str(tmp_path / "old.db"), str(tmp_path / "new.db"))
    assert counts == {"facts": 1, "turns": 1, "episodes": 1}
    new = SQLiteStore(str(tmp_path / "new.db"))
    assert new.get_all_facts()
    assert new.get_turns_by_session("s")
    assert new.get_all_episodes()
    new.close()

    class Emb(BaseEmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> List[List[float]]:
            return [[float(len(text))] for text in texts]

        def similarity(self, text1: str, text2: str) -> float:
            return 1.0 if text1 == text2 else 0.0

    assert Emb().embed_one("abc") == [3.0]
    assert Emb().similarity("a", "a") == 1.0

    class Ranker(RankerInterface):
        def score(self, query: str, result: MemorySearchResult) -> float:
            return result.score + (1.0 if query in result.content else 0.0)

    ranked = Ranker().rank(
        "x",
        [
            MemorySearchResult("x", "semantic", 0.1),
            MemorySearchResult("y", "semantic", 0.9),
        ],
    )
    assert ranked[0].content == "x"
    assert len(Ranker().rank("x", ranked, limit=1)) == 1

    class Summ(SummarizerInterface):
        def summarize_texts(self, texts: Sequence[str]) -> str:
            return " ".join(texts)

        def summarize_turns(self, turns: Sequence[Any]) -> str:
            return " ".join(turn.user_message for turn in turns)

    assert Summ().summarize_texts(["a", "b"]) == "a b"
    assert Summ().summarize_turns([Turn("a", "b", "s")]) == "a"

    class Store(StorageBackend):
        def __init__(self) -> None:
            self.facts: Dict[str, Fact] = {}
            self.turns: List[Dict[str, Any]] = []
            self.episodes: List[Dict[str, Any]] = []

        def save_fact(self, fact: Any) -> None:
            f = fact if isinstance(fact, Fact) else Fact.from_dict(fact)
            self.facts[f.id] = f

        def get_fact(self, fact_id: str) -> Optional[Fact]:
            return self.facts.get(fact_id)

        def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
            return list(self.facts.values())[:limit]

        def get_facts_by_session(self, session_id: str) -> List[Fact]:
            return [fact for fact in self.facts.values() if fact.session_id == session_id]

        def get_facts_by_type(self, fact_type: str) -> List[Fact]:
            return [fact for fact in self.facts.values() if fact.type == fact_type]

        def search_facts_keyword(self, keyword: str) -> List[Fact]:
            return [fact for fact in self.facts.values() if keyword in fact.content]

        def update_fact_access_count(self, fact_id: str) -> None:
            if fact_id in self.facts:
                self.facts[fact_id].access_count += 1

        def delete_fact(self, fact_id: str) -> None:
            self.facts.pop(fact_id, None)

        def save_turn(self, turn: Any) -> None:
            self.turns.append(turn if isinstance(turn, dict) else {"session_id": turn.session_id})

        def get_turns_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
            return [turn for turn in self.turns if turn.get("session_id") == session_id][:limit]

        def save_episode(self, episode: Dict[str, Any]) -> None:
            self.episodes.append(episode)

        def get_episodes_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
            return [ep for ep in self.episodes if ep.get("session_id") == session_id][:limit]

        def get_all_episodes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
            return self.episodes[:limit]

        def clear_session(self, session_id: str) -> None:
            self.facts = {k: v for k, v in self.facts.items() if v.session_id != session_id}

        def clear_all(self) -> None:
            self.facts.clear()
            self.turns.clear()
            self.episodes.clear()

    store = Store()
    fact = Fact("User prefers tests", "preference", 0.9, "s")
    store.save_facts([fact])
    assert store.get_fact(fact.id) is fact
    store.update_fact_access_count(fact.id)
    assert fact.access_count == 1
    store.delete_fact(fact.id)
    assert store.get_fact(fact.id) is None

    assert importlib.import_module("memoryos.interfaces").__all__
    assert importlib.import_module("memoryos.log").logger.name == "memoryos"
