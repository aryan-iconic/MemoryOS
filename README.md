<div align="center">

# MemoryOS

**Persistent long-term memory for any AI application.**

[![PyPI version](https://badge.fury.io/py/memoryos.svg)](https://badge.fury.io/py/memoryos)
[![Python](https://img.shields.io/pypi/pyversions/memoryos)](https://pypi.org/project/memoryos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/aryan-iconic/MemoryOS?style=social)](https://github.com/aryan-iconic/MemoryOS)

MemoryOS gives any LLM or AI assistant persistent, structured memory across sessions.  
Works with ChatGPT, Claude, Gemini, local models, or any LLM-based system.  
No cloud required. No vendor lock-in. Fully local by default.

[Installation](#installation) • [Quickstart](#quickstart) • [Architecture](#architecture) • [Examples](#examples) • [Configuration](#configuration) • [API Reference](#api-reference)

</div>

---

## Why MemoryOS

LLMs are stateless. Every conversation starts from zero — users repeat themselves, context is lost, and AI assistants feel shallow.

MemoryOS adds a structured memory layer that:

- Remembers user preferences, goals, identity, and decisions across sessions
- Retrieves only the most relevant memories for each query (not everything at once)
- Stays fully local with SQLite and in-memory vector search out of the box
- Works with any LLM — just inject the context string into your prompt

```python
from memoryos import MemoryOS

memory = MemoryOS(session_id="user_123")

# Turn 1
memory.process_turn("My name is Aryan. I prefer dark mode UI.", "Got it!")

# Turn 2 — days later, new session
context = memory.build_context("What does this user prefer?")
# → "Relevant user facts:\n- User's name is Aryan (type=identity)\n- User prefers dark mode UI (type=preference)"
```

---

## Installation

```bash
pip install memoryos
```

With semantic embeddings (recommended for production):

```bash
pip install "memoryos[embeddings]"
```

With FAISS vector index (for large memory stores):

```bash
pip install "memoryos[embeddings,faiss]"
```

**Requirements:** Python 3.9+, numpy (auto-installed)

---

## Quickstart

### Basic usage

```python
from memoryos import MemoryOS

# Initialize — SQLite database created automatically
memory = MemoryOS(db_path="memory.db", session_id="user_1")

# Process conversation turns
memory.process_turn("My name is Aryan. I'm building an AI memory library.", "That's awesome!")
memory.process_turn("I prefer dark mode and minimal UI.", "Noted!")
memory.process_turn("My goal is to land a remote backend engineering role.", "Great goal!")

# Build context for your next LLM prompt
context = memory.build_context("Tell me about this user.")
print(context)

# Search memory directly
results = memory.search_memory("UI preferences", top_k=3)
for result in results:
    print(f"{result.content}  [score={result.score:.3f}]")

memory.close()
```

### Using context in an LLM prompt

```python
from memoryos import MemoryOS

memory = MemoryOS(db_path="memory.db", session_id="user_1")

def chat(user_message: str) -> str:
    # Get relevant memory context
    context = memory.build_context(user_message)

    # Inject into your LLM system prompt
    system_prompt = f"""You are a helpful assistant.

{context}

Use the above memory context to personalize your response."""

    # Call your LLM here (OpenAI, Anthropic, local model, etc.)
    ai_response = your_llm_call(system_prompt, user_message)

    # Save the turn so memory grows over time
    memory.process_turn(user_message, ai_response)

    return ai_response
```

---

## Architecture

MemoryOS uses a three-tier memory architecture. Each layer serves a different purpose.

```
┌─────────────────────────────────────────────────────────┐
│                        MemoryOS                         │
│                                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Working Memory │  │   Semantic   │  │  Episodic │  │
│  │   (RAM)         │  │   Memory     │  │  Memory   │  │
│  │                 │  │  (SQLite +   │  │  (SQLite) │  │
│  │ Last 6–10 turns │  │   Vectors)   │  │           │  │
│  │ Always injected │  │              │  │ Compressed│  │
│  │ Verbatim        │  │ Long-term    │  │ episode   │  │
│  │                 │  │ facts        │  │ summaries │  │
│  └─────────────────┘  └──────────────┘  └───────────┘  │
│                                                         │
│              MemoryRetriever + MemoryRanker              │
│         (similarity · confidence · recency · type)      │
│                                                         │
│                   PromptContextBuilder                   │
│              (token-budget-aware formatting)             │
└─────────────────────────────────────────────────────────┘
```

### Working Memory
- Stores the last 6–10 conversation turns verbatim
- Always injected into context — no retrieval needed
- Provides immediate short-term continuity

### Semantic Memory
- Extracts and stores long-term facts from conversations
- Fact types: `identity`, `preference`, `goal`, `decision`, `context`
- Retrieved by vector similarity to the current query
- Deduplicated automatically — no repeated facts
- Confidence scoring filters out low-quality extractions

### Episodic Memory
- Compressed summaries of past conversation sessions
- Retrieved by semantic similarity
- Reduces token usage for long conversation histories

### Fact Extraction
MemoryOS automatically detects and extracts facts from user messages:

| Pattern | Example Input | Extracted Fact |
|---------|--------------|----------------|
| Identity | "My name is Aryan" | `User's name is Aryan` |
| Preference | "I prefer dark mode" | `User prefers dark mode` |
| Goal | "My goal is to learn Rust" | `User's goal is to learn Rust` |
| Decision | "I decided to use Postgres" | `User decided to use Postgres` |
| Context | "I am working on MemoryOS" | `User is working on MemoryOS` |

---

## Examples

### ChatGPT integration

```python
from openai import OpenAI
from memoryos import MemoryOS

client = OpenAI()
memory = MemoryOS(db_path="chat.db", session_id="user_1")

def chat(user_message: str) -> str:
    context = memory.build_context(user_message)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"},
            {"role": "user", "content": user_message},
        ],
    )

    ai_response = response.choices[0].message.content
    memory.process_turn(user_message, ai_response)
    return ai_response

print(chat("My name is Aryan and I prefer concise answers."))
print(chat("What's my name?"))  # Memory recalls it
```

### Claude integration

```python
import anthropic
from memoryos import MemoryOS

client = anthropic.Anthropic()
memory = MemoryOS(db_path="chat.db", session_id="user_1")

def chat(user_message: str) -> str:
    context = memory.build_context(user_message)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=f"You are a helpful assistant.\n\n{context}",
        messages=[{"role": "user", "content": user_message}],
    )

    ai_response = response.content[0].text
    memory.process_turn(user_message, ai_response)
    return ai_response
```

### Gemini integration

```python
import google.generativeai as genai
from memoryos import MemoryOS

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
memory = MemoryOS(db_path="chat.db", session_id="user_1")

def chat(user_message: str) -> str:
    context = memory.build_context(user_message)
    prompt = f"{context}\n\nUser: {user_message}"

    response = model.generate_content(prompt)
    ai_response = response.text

    memory.process_turn(user_message, ai_response)
    return ai_response
```

### Custom embedding provider

```python
from memoryos import MemoryOS
from memoryos.interfaces.embedding_provider import EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
    model_name = "my-custom-model"
    dimension = 768

    def embed(self, texts):
        # Call your embedding API or local model here
        return my_embedding_api(texts)

    def similarity(self, text1: str, text2: str) -> float:
        e1 = self.embed_one(text1)
        e2 = self.embed_one(text2)
        return cosine_similarity(e1, e2)

memory = MemoryOS(db_path="chat.db", session_id="user_1")
# Inject custom embedding at the semantic memory level
from memoryos.memory.semantic import SemanticMemory
# See advanced docs for full custom injection
```

### Adding memory manually

```python
from memoryos import MemoryOS

memory = MemoryOS(db_path="chat.db", session_id="user_1")

# Manually add a fact without extraction
memory.add_memory("User is a backend engineer at Tata Motors", fact_type="context", confidence=0.99)
memory.add_memory("User's preferred language is Python", fact_type="preference", confidence=0.95)

# Retrieve it
context = memory.build_context("What tech does this user use?")
print(context)
```

### Context manager

```python
from memoryos import MemoryOS

with MemoryOS(db_path="chat.db", session_id="user_1") as memory:
    memory.process_turn("I'm working on a RAG pipeline.", "Nice!")
    context = memory.build_context("What is the user building?")
    print(context)
# Connection closed automatically
```

---

## Configuration

All settings can be passed to `MemoryOSConfig` or as keyword arguments.

```python
from memoryos import MemoryOS, MemoryOSConfig

config = MemoryOSConfig(
    db_path="my_memory.db",           # SQLite database path
    working_memory_size=10,           # Turns to keep in working memory
    semantic_top_k=5,                 # Top facts to retrieve per query
    episodic_top_k=3,                 # Top episode summaries to retrieve
    min_fact_confidence=0.65,         # Minimum confidence to store a fact
    duplicate_similarity_threshold=0.90,  # Threshold to skip duplicate facts
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    max_context_tokens=6000,          # Max characters in context output
    auto_create_episodes=False,       # Auto-create episode summaries
)

memory = MemoryOS(config=config, session_id="user_1")
```

### Environment variable configuration

```bash
export MEMORYOS_DB_PATH=./data/memory.db
export MEMORYOS_WORKING_MEMORY_SIZE=12
export MEMORYOS_MIN_FACT_CONFIDENCE=0.70
export MEMORYOS_ENABLE_FAISS=true
```

```python
memory = MemoryOS.from_env(session_id="user_1")
```

### Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `memoryos.db` | SQLite database file path |
| `working_memory_size` | `8` | Number of recent turns to keep |
| `semantic_top_k` | `5` | Top semantic facts retrieved per query |
| `episodic_top_k` | `3` | Top episode summaries retrieved per query |
| `min_fact_confidence` | `0.65` | Minimum confidence score to store a fact |
| `duplicate_similarity_threshold` | `0.90` | Cosine threshold for deduplication |
| `embedding_model_name` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `embedding_dim` | `384` | Embedding vector dimensions |
| `max_context_tokens` | `6000` | Max characters in built context |
| `enable_faiss` | `False` | Use FAISS for vector index |
| `auto_create_episodes` | `False` | Auto-create episode summaries |
| `min_episode_turns` | `4` | Minimum turns before creating an episode |

---

## API Reference

### `MemoryOS`

#### Initialization

```python
MemoryOS(
    db_path: str = None,
    session_id: str = "default_session",
    config: MemoryOSConfig = None,
)
```

#### Core methods

| Method | Description |
|--------|-------------|
| `process_turn(user_message, ai_response)` | Save a turn and extract facts |
| `build_context(query)` | Build context string for LLM prompt |
| `search_memory(query, top_k, min_score)` | Search all memory layers |
| `add_memory(content, fact_type, confidence)` | Manually add a fact |

#### Data access

| Method | Description |
|--------|-------------|
| `get_all_facts(limit)` | Return all stored facts |
| `get_session_facts(session_id, limit)` | Return facts for a session |
| `get_turns(session_id, limit)` | Return stored conversation turns |
| `get_episodes(session_id, limit)` | Return episode summaries |

#### Session management

| Method | Description |
|--------|-------------|
| `clear_session(session_id)` | Clear all data for a session |
| `clear_all()` | Clear all data in the database |
| `close()` | Close database connection |
| `maybe_create_episode(session_id)` | Manually trigger episode creation |

### `Fact` object

```python
@dataclass
class Fact:
    content: str          # "User's name is Aryan"
    type: str             # identity | preference | goal | decision | context
    confidence: float     # 0.0 – 1.0
    session_id: str
    id: str               # UUID
    source: str           # conversation | manual | system
    timestamp: float      # Unix timestamp
    access_count: int
    embedding: List[float]
    metadata: Dict
```

### `MemorySearchResult` object

```python
@dataclass
class MemorySearchResult:
    content: str          # Memory content
    source: str           # working | semantic | episodic
    score: float          # Ranked relevance score 0.0 – 1.0
    type: str             # Fact type if semantic
    confidence: float     # Original extraction confidence
    timestamp: float
    metadata: Dict
```

---

## Ranking System

MemoryOS ranks retrieved memories using four signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Similarity | 70% | Cosine similarity to query embedding |
| Confidence | 15% | Extraction confidence score |
| Recency | 10% | Exponential decay (30-day half-life by default) |
| Source | 5% | semantic > episodic > working |

Fact types also apply multipliers: `identity` (1.05×) → `decision` (1.03×) → `goal` (1.02×) → `preference` (1.01×) → `context` (1.00×).

---

## Design Philosophy

- **Local-first** — no required cloud services, no hidden API calls
- **Model-agnostic** — works with any LLM, any embedding model
- **No forced dependencies** — sentence-transformers and FAISS are optional
- **Transparent** — confidence scores, source labels, and ranked scores are always visible
- **Extensible** — replace any component: storage, embeddings, ranking, summarization
- **Opinionated defaults** — works out of the box, customizable for production

---

## Roadmap

- [x] Three-tier memory architecture (working, semantic, episodic)
- [x] Fact extraction with confidence scoring and deduplication
- [x] SQLite storage backend
- [x] In-memory and FAISS vector index
- [x] Token-budget-aware context builder
- [x] Recency-weighted ranking
- [x] `from_env()` configuration
- [ ] PostgreSQL + pgvector storage backend
- [ ] Async API (`aprocess_turn`, `abuild_context`)
- [ ] OpenAI-compatible embedding provider
- [ ] Memory export / import (JSON)
- [ ] REST API server mode
- [ ] LangChain and LlamaIndex integrations

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large PR.

```bash
git clone https://github.com/aryan-iconic/MemoryOS
cd MemoryOS
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Aryan Gupta](https://github.com/aryan-iconic) · [GitHub](https://github.com/aryan-iconic/MemoryOS) · [PyPI](https://pypi.org/project/memoryos/)

</div>
