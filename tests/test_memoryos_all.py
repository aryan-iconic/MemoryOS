"""End-to-end MemoryOS test covering the complete local flow."""

from __future__ import annotations

from pathlib import Path

from memoryos import MemoryOS, MemoryOSConfig
from memoryos.extraction import ConfidenceScorer, Deduplicator, Extractor
from memoryos.models import Fact, MemorySearchResult, Turn
from memoryos.retrieval.builder import PromptContextBuilder
from memoryos.retrieval.ranker import MemoryRanker
from memoryos.storage.index import InMemoryVectorIndex
from memoryos.storage.sqlite_store import SQLiteStore


def test_memoryos_end_to_end(tmp_path: Path) -> None:
    config = MemoryOSConfig(
        db_path=str(tmp_path / "memoryos_test.db"),
        working_memory_size=8,
        semantic_top_k=6,
        episodic_top_k=3,
        min_episode_turns=2,
        episode_turn_window=4,
        min_fact_confidence=0.60,
    )

    memory = MemoryOS(config=config, session_id="test_session")
    memory.clear_all()

    result_1 = memory.process_turn(
        user_message="My name is Aryan. I prefer dark UI. I want to build MemoryOS.",
        ai_response="Nice, MemoryOS sounds powerful.",
    )
    result_2 = memory.process_turn(
        user_message="I am working on an AI memory system. I decided to keep it model agnostic.",
        ai_response="Got it.",
    )

    assert result_1["turn"].session_id == "test_session"
    assert len(result_1["new_facts"]) >= 3
    assert len(result_2["new_facts"]) >= 2

    all_facts = memory.get_all_facts()
    fact_text = "\n".join(fact.content for fact in all_facts)
    assert "User's name is Aryan" in fact_text
    assert "User prefers dark UI" in fact_text
    assert "User wants to build MemoryOS" in fact_text
    assert all(fact.embedding for fact in all_facts)

    duplicate = memory.process_turn(
        user_message="I prefer dark UI.",
        ai_response="Already remembered.",
    )
    assert duplicate["new_facts"] == []

    manual_fact = memory.add_memory(
        "User wants MemoryOS to support pluggable storage and retrieval.",
        fact_type="goal",
        confidence=0.95,
    )
    assert manual_fact.source == "manual"

    ui_results = memory.search_memory("What UI theme does the user prefer?", top_k=5, min_score=0.0)
    assert ui_results
    assert any("dark UI" in result.content for result in ui_results)
    assert all(isinstance(result, MemorySearchResult) for result in ui_results)

    project_results = memory.search_memory("What is the user building?", top_k=5, min_score=0.0)
    assert any("MemoryOS" in result.content or "AI memory system" in result.content for result in project_results)

    episode = memory.maybe_create_episode()
    assert episode is not None
    assert episode["summary"]
    assert episode["embedding"]
    assert len(memory.get_episodes()) >= 1

    context = memory.build_prompt_context(
        "What should I know about the user's project and preferences?",
        memory_limit=8,
        turn_limit=4,
        max_chars=4000,
    )
    assert "Relevant user facts" in context
    assert "Recent conversation" in context
    assert "dark UI" in context
    assert "MemoryOS" in context

    turns = memory.get_turns()
    assert len(turns) == 3

    session_facts = memory.get_session_facts()
    assert len(session_facts) == len(memory.get_all_facts())

    memory.clear_session()
    assert memory.get_turns() == []
    assert memory.get_session_facts() == []
    assert memory.get_episodes() == []
    memory.close()


def test_storage_extraction_retrieval_helpers(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "store.db"))
    turn = Turn(
        user_message="My name is Aryan. I like clean documentation. My goal is shipping MemoryOS.",
        ai_response="Understood.",
        session_id="helper_session",
    )
    store.save_turn(turn)
    assert len(store.get_turns_by_session("helper_session")) == 1

    extractor = Extractor(min_confidence=0.60)
    facts = extractor.extract(turn)
    assert len(facts) >= 3

    deduped = Deduplicator().deduplicate_facts(facts + facts)
    assert len(deduped) == len(facts)

    score = ConfidenceScorer().calculate("I prefer reliable systems", 0.80)
    assert 0.0 <= score <= 1.0

    saved = store.save_facts(facts)
    assert len(saved) == len(facts)
    assert store.search_facts_keyword("documentation")

    index = InMemoryVectorIndex(dim=3)
    index.add("a", [1.0, 0.0, 0.0], metadata={"label": "first"})
    index.add("b", [0.0, 1.0, 0.0], metadata={"label": "second"})
    index_results = index.search([1.0, 0.0, 0.0], top_k=1)
    assert index_results[0].id == "a"

    ranker = MemoryRanker()
    ranked = ranker.rank(
        [
            MemorySearchResult(content="low", source="semantic", score=0.1),
            MemorySearchResult(content="high", source="semantic", score=0.9),
        ]
    )
    assert ranked[0].content == "high"

    builder = PromptContextBuilder(max_chars=1000)
    built = builder.build(
        query="test",
        results=[MemorySearchResult(content="User prefers reliable systems.", source="semantic", score=0.8)],
        recent_turns=[turn],
    )
    assert "Relevant user facts" in built
    assert "Recent conversation" in built

    store.clear_all()
    assert store.get_all_facts() == []
    store.close()
