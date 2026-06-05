# MemoryOS

> Persistent, local-first memory infrastructure for AI applications.

MemoryOS is a Python library that gives AI applications long-term memory across conversations, sessions, users, and projects. It stores recent dialogue, extracts reusable facts, summarizes past sessions, retrieves relevant memories, and builds a compact context block that can be passed to any LLM.

MemoryOS is designed to be **model-agnostic**, **local-first**, and **developer-controlled**. It does not force a specific LLM provider, vector database, or cloud service.

---

## Why MemoryOS?

Most LLM applications are stateless by default. They can respond intelligently inside one conversation, but they often forget:

- who the user is,
- what the user prefers,
- what decisions were already made,
- what project context matters,
- what happened in earlier sessions.

MemoryOS solves this by adding a reusable memory layer between your application and the LLM.

Instead of sending full chat history every time, MemoryOS stores structured memory and retrieves only what is relevant for the current query.

---

## Core Features

- **Working memory** for recent conversation turns.
- **Semantic memory** for long-term facts, preferences, goals, decisions, and context.
- **Episodic memory** for compressed summaries of previous sessions.
- **Fact extraction** from natural language messages.
- **Confidence scoring** to filter low-quality facts.
- **Deduplication** to avoid repeated memory entries.
- **SQLite persistence** for local-first storage.
- **Embedding-based retrieval** with deterministic fallback embeddings.
- **Optional sentence-transformers support** for higher-quality semantic embeddings.
- **Optional FAISS index support** for vector search experiments.
- **Context builder** that prepares clean memory context for LLM prompts.
- **Pluggable interfaces** for custom storage, embeddings, summarizers, and rankers.
- **Quality gates** with ruff, mypy, pytest, build checks, and full test coverage.

---

## Architecture

MemoryOS uses a three-layer memory architecture.

```text
┌─────────────────────────────────────────────────────────────┐
│                         Your App                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        MemoryOS API                         │
│              process_turn · search · build_context          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Manager                         │
│        coordinates extraction, storage, retrieval, context   │
└───────────────┬────────────────┬────────────────┬───────────┘
                │                │                │
                ▼                ▼                ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Working Memory  │  │ Semantic Memory  │  │ Episodic Memory  │
│  recent turns    │  │ durable facts    │  │ session summaries │
└──────────────────┘  └──────────────────┘  └──────────────────┘
                │                │                │
                └────────────────┴────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Retriever + Ranker + Builder               │
│       fetch relevant memories and build final prompt context │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Layers

### 1. Working Memory

Working memory stores the most recent conversation turns. It helps the assistant maintain immediate context.

Typical use:

- recent user messages,
- recent assistant replies,
- short-term context needed in the current session.

### 2. Semantic Memory

Semantic memory stores durable facts extracted from conversations or manually added by the developer.

Examples:

- `User prefers dark UI.`
- `User is building an AI memory system.`
- `User decided to keep the project model-agnostic.`

Semantic memory is searched using embeddings and ranked by relevance, confidence, recency, and metadata.

### 3. Episodic Memory

Episodic memory stores compressed summaries of previous conversation windows. It helps preserve long-term continuity without sending full chat logs to the LLM.

Examples:

- summary of a planning session,
- summary of a debugging session,
- summary of project decisions made over time.

---

## Installation

### Install from source

```bash
git clone https://github.com/<your-username>/memoryos.git
cd memoryos
python -m venv .venv
```

Activate the environment.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install MemoryOS:

```bash
pip install -e .
```

### Optional dependencies

For sentence-transformers embeddings:

```bash
pip install -e ".[embeddings]"
```

For FAISS support:

```bash
pip install -e ".[faiss]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from memoryos import MemoryOS

memory = MemoryOS(
    db_path="memoryos.db",
    session_id="user_1",
)

memory.process_turn(
    user_message="My name is Aryan. I prefer dark UI. I am building MemoryOS.",
    ai_response="Got it. I will remember that.",
)

context = memory.build_context("What should I remember about this user?")
print(context)

memory.close()
```

Example output:

```text
Relevant user facts:
- User's name is Aryan.
- User prefers dark UI.
- User wants to build MemoryOS.

Recent conversation:
User: My name is Aryan. I prefer dark UI. I am building MemoryOS.
AI: Got it. I will remember that.
```

---

## Manual Memory

You can manually add durable memory without relying on extraction.

```python
from memoryos import MemoryOS

memory = MemoryOS(db_path="memoryos.db", session_id="user_1")

memory.add_memory(
    "User wants MemoryOS to be a reusable memory layer for AI applications.",
    fact_type="goal",
    confidence=0.95,
)

results = memory.search_memory("What is the user's project goal?", top_k=3)

for result in results:
    print(result.content, result.score)

memory.close()
```

---

## Using MemoryOS With Any LLM

MemoryOS does not call an LLM by default. It builds the memory context and lets your application pass that context to any model.

```python
user_message = "Can you continue from where we stopped?"

memory_context = memory.build_context(user_message)

prompt = f"""
Use the following memory context when helpful.

{memory_context}

User message:
{user_message}
"""

# Send `prompt` to your LLM provider of choice.
```

This makes MemoryOS usable with OpenAI, Anthropic, Gemini, local models, custom inference servers, or your own AI application.

---

## Configuration

```python
from memoryos import MemoryOS
from memoryos.config import MemoryOSConfig

config = MemoryOSConfig(
    db_path="data/memoryos.db",
    working_memory_size=8,
    semantic_top_k=5,
    episodic_top_k=3,
    min_fact_confidence=0.65,
    duplicate_similarity_threshold=0.90,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    auto_create_episodes=False,
)

memory = MemoryOS(config=config, session_id="user_1")
```

### Environment variables

MemoryOS can also read configuration from environment variables.

```bash
MEMORYOS_DB_PATH=./data/memoryos.db
MEMORYOS_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MEMORYOS_WORKING_MEMORY_SIZE=8
MEMORYOS_SEMANTIC_TOP_K=5
MEMORYOS_EPISODIC_TOP_K=3
MEMORYOS_MIN_FACT_CONFIDENCE=0.65
MEMORYOS_ENABLE_FAISS=false
```

```python
from memoryos import MemoryOS

memory = MemoryOS.from_env(session_id="user_1")
```

---

## Public API

| Method | Purpose |
|---|---|
| `process_turn(user_message, ai_response="")` | Save a conversation turn and extract durable facts. |
| `add_memory(content, fact_type="context")` | Manually add a long-term fact. |
| `search_memory(query, top_k=5)` | Search working, semantic, and episodic memory. |
| `search(query, top_k=5)` | Alias for `search_memory`. |
| `build_context(query)` | Build final memory context for an LLM prompt. |
| `build_prompt_context(query)` | Backward-compatible context builder. |
| `maybe_create_episode()` | Create an episodic summary from recent turns when enough exist. |
| `get_all_facts()` | Return stored facts. |
| `get_session_facts(session_id)` | Return facts for a session. |
| `get_turns(session_id)` | Return stored conversation turns. |
| `get_episodes(session_id)` | Return episodic summaries. |
| `clear_session(session_id)` | Delete memory for one session. |
| `clear_all()` | Delete all stored memory. |
| `close()` | Close storage resources. |

---

## Fact Model

MemoryOS stores durable memory as structured facts.

```python
{
    "id": "uuid",
    "content": "User prefers dark UI.",
    "type": "preference",
    "confidence": 0.95,
    "session_id": "user_1",
    "source": "conversation",
    "timestamp": 1714639200.0,
    "access_count": 0,
    "embedding": [0.01, 0.02, ...],
    "metadata": {},
}
```

Supported fact types:

- `identity`
- `preference`
- `goal`
- `decision`
- `context`

Supported fact sources:

- `conversation`
- `manual`
- `system`

---

## Extraction Philosophy

MemoryOS is intentionally conservative.

It should not store every sentence. It should store reusable information that is likely to matter later.

Examples of useful memory:

- user preferences,
- stable identity details,
- long-term goals,
- project decisions,
- important context.

Examples of memory to avoid:

- temporary small talk,
- one-off questions,
- unsupported assumptions,
- noisy or low-confidence facts.

Core principle:

> Extract less, but extract meaningful information.

---

## Retrieval Flow

When you call `build_context(query)`, MemoryOS performs the following steps:

```text
query
  → retrieve recent working memory
  → search semantic facts
  → search episodic summaries
  → rank by relevance, confidence, recency, and source
  → deduplicate results
  → build compact prompt context
```

The final output is a clean text block that can be inserted into an LLM prompt.

---

## Project Structure

```text
memoryos/
├── compression/        # summarization, compression, token budgeting
├── embeddings/         # embedding providers and deterministic fallback
├── extraction/         # fact extraction, confidence scoring, deduplication
├── interfaces/         # pluggable backend/provider contracts
├── memory/             # working, semantic, episodic memory and manager
├── retrieval/          # retriever, ranker, scorer, context builder
├── storage/            # SQLite storage, vector index, FAISS support, migrations
├── utils/              # utility helpers
├── config.py           # runtime configuration
├── core.py             # public MemoryOS API
├── exceptions.py       # custom exceptions
└── models.py           # core dataclasses
```

---

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q
```

Run coverage:

```bash
python -m pytest --cov=memoryos --cov-report=term-missing
```

Run linting:

```bash
python -m ruff check memoryos tests
```

Run type checking:

```bash
python -m mypy memoryos
```

Build package:

```bash
python -m build
```

Full quality gate:

```bash
python -m ruff check memoryos tests
python -m mypy memoryos
python -m pytest -q
python -m pytest --cov=memoryos --cov-report=term-missing
python -m build
python -c "from memoryos import MemoryOS; print('ok')"
```

Current local quality status for v0.1.0:

```text
ruff: passed
mypy: passed
pytest: passed
coverage: 100%
build: passed
import check: passed
```

---

## Debug Demo

Run the included debug script:

```bash
python debug.py
```

The script demonstrates:

- processing turns,
- extracting facts,
- saving memory,
- creating an episode,
- searching memory,
- building context.

---

## Design Principles

- **Local-first**: Memory should work without mandatory cloud infrastructure.
- **Model-agnostic**: MemoryOS should work with any LLM or chatbot.
- **Storage-flexible**: SQLite by default, but designed for future backends.
- **Context-efficient**: Retrieve only the memory needed for the current task.
- **Transparent**: Developers should be able to inspect, debug, and control memory.
- **Pluggable**: Embeddings, summarizers, rankers, and storage can evolve independently.

---

## Limitations

MemoryOS is currently an early `v0.1.0` library. The core system is functional, tested, and installable, but some production extensions are still future work.

Current limitations:

- SQLite is the default storage backend.
- Built-in extraction is rule/heuristic based.
- Advanced LLM-based extraction is not bundled by default.
- FAISS is optional and experimental.
- Distributed/team memory, auth, encryption, and hosted APIs are not included yet.
- Quality of semantic retrieval depends on the embedding provider used.

---

## Roadmap

Planned improvements:

- Postgres and pgvector backend.
- Import/export and backup tools.
- More advanced fact extraction.
- LLM-powered optional summarization and extraction adapters.
- Stronger benchmarking suite.
- More integration examples.
- Hosted API wrapper.
- Better documentation site.
- CI/CD workflow for GitHub Actions.

---

## Example Use Cases

MemoryOS can be used in:

- AI assistants,
- AI coding copilots,
- legal AI workspaces,
- customer support agents,
- personal productivity assistants,
- research assistants,
- education tutors,
- local-first AI tools,
- multi-session chatbot systems.

---

## Contributing

Contributions are welcome. Good first areas include:

- improving extraction patterns,
- adding more tests,
- building integrations,
- improving documentation,
- adding new storage backends,
- improving retrieval and ranking.

Before opening a pull request, run:

```bash
python -m ruff check memoryos tests
python -m mypy memoryos
python -m pytest --cov=memoryos --cov-report=term-missing
```

---

## License

Add your chosen license here, for example MIT, Apache-2.0, or another license depending on your release plan.

---

## Summary

MemoryOS is a persistent memory layer for AI applications.

It combines working memory, semantic memory, and episodic memory into one local-first Python package. It helps AI systems remember useful context across sessions while keeping memory retrieval compact, transparent, and developer-controlled.
