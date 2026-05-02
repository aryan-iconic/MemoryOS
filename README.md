# MemoryOS

## Universal AI Memory System for Any LLM or Chatbot

**MemoryOS** is a Python library that gives any AI assistant persistent, intelligent memory. It works with ChatGPT, Claude, Gemini, local models, and any LLM-based system.

It is a **local-first, model-agnostic memory engine** that provides the complex layer of memory handling while letting users control infrastructure.

---

## Core Vision

LLMs are stateless by default. MemoryOS adds structured, persistent memory so AI systems can:

* Remember past conversations
* Store user preferences
* Track long-term goals
* Maintain continuity across sessions

The goal is to build a **foundational memory layer** for AI systems.

---

## Design Philosophy

* Local-first (no required cloud)
* No forced APIs or paid dependencies
* Model-agnostic
* User-controlled infrastructure
* Opinionated defaults + deep customization
* Transparent and debuggable

---

## Three-Tier Memory Architecture

### 1. Working Memory (RAM)

* Last 6–10 turns
* Stored verbatim
* Always injected
* Maintains immediate context

### 2. Episodic Memory (SQLite)

* Summarized past conversations
* Stored persistently
* Retrieved by recency
* Reduces token usage

### 3. Semantic Memory (FAISS)

* Extracted facts
* Stored as embeddings
* Retrieved by similarity
* High-value long-term memory

---

## Fact Schema (Core Data Structure)

```python
fact = {
    "id": "uuid",
    "content": "User prefers dark mode UI",
    "embedding": [...],
    "source": "conversation",
    "timestamp": 1714639200,
    "confidence": 0.85,
    "type": "preference",
    "access_count": 0
}
```

### Fact Types

* identity
* preference
* goal
* decision
* context

### Design Rules

* Store only **stable, reusable information**
* Avoid noise (questions, generic chat)
* Deduplicate similar facts

---

## Fact Extraction Strategy

### Approach: Hybrid (Rule + Heuristic)

### Trigger Conditions

* Pattern match OR
* Strong semantic signal

### Example Patterns

* "my name is"
* "i prefer"
* "i like"
* "i am working on"
* "my goal is"

### Extraction Flow

1. Detect pattern
2. Convert to structured fact
3. Assign type
4. Compute confidence
5. Filter noise
6. Store if valid

### Key Principle

> Extract less, but extract meaningful information

---

## Confidence Scoring System

### Base Scores

* Direct pattern match → 0.9
* Strong inference → 0.75
* Weak inference → 0.6

### Modifiers

+0.05 → repeated or explicit statement
-0.1 → uncertain words ("maybe", "I think", "probably")

### Final Calculation

```python
confidence = max(0.0, min(1.0, base + modifiers))
```

### Usage

* Discard facts if confidence < 0.65
* Use in ranking
* Use in deduplication

---

## Summarization Strategy

### Default (Lightweight)

* Extractive summarization
* No heavy model
* Select top 1–2 important sentences

### Customization

Users can replace summarizer with:

* LLM-based summarizer
* Local transformer

---

## Token Budget Strategy

### Goal

Fit all memory within model context limits.

### Priority Order

1. Working memory
2. Semantic facts
3. Episodic summaries

### Adaptive Allocation (No Fixed Ratios)

```python
WORKING_MIN = 0.25
WORKING_MAX = 0.60
FACTS_MIN = 0.20
FACTS_MAX = 0.50
SUMMARY_MAX = 0.30
```

### Context Assembly

1. Add working memory fully
2. Add ranked semantic facts
3. Fill remaining space with summaries

### Overflow Handling

* Remove oldest summaries
* Remove lowest-ranked facts
* Trim working memory (last resort)

---

## Context Building Flow

1. Retrieve working memory
2. Retrieve semantic facts (FAISS)
3. Retrieve episodic summaries (SQLite)
4. Rank memory
5. Merge context
6. Enforce token budget

---

## Core Pipeline (Fixed Structure)

1. Receive input
2. Retrieve memory
3. Rank memory
4. Build context
5. Return context
6. Save turn
7. Extract facts
8. Store memory

### Rule

Users can modify **how each step works**, but not remove steps.

---

## Customization System

Users can replace:

* summarizer_fn
* embedding_fn
* rank_fn
* extract_fn
* storage backend (advanced)

Example:

```python
memory = MemoryOS(
    summarizer_fn=my_summarizer,
    embedding_fn=my_embeddings,
    rank_fn=my_ranker
)
```

---

## Technology Stack

* FAISS → semantic search
* sentence-transformers → embeddings
* SQLite → episodic storage
* tiktoken → token control

---

## Public API

### Initialize

```python
memory = MemoryOS(session_id="user_1")
```

### Get Context

```python
context = memory.get_context(user_message)
```

### Save Turn

```python
memory.save_turn(user_msg, ai_response)
```

### Search

```python
memory.search(query)
```

---

## Default vs Advanced Usage

### Default Users

* Works out of the box
* Local embeddings
* SQLite + FAISS

### Advanced Users

* Replace models
* Customize ranking
* Modify pipeline logic

---

## Key Strengths

* No cloud dependency
* No hidden cost
* Fully local capable
* Works with any LLM
* Highly modular
* Transparent system

---

## Build Roadmap

### Phase 1

* MemoryManager
* Fact schema
* Extraction
* Context builder

### Phase 2

* Summarization improvements
* Compression tuning

### Phase 3

* Testing + benchmarks

### Phase 4

* Packaging (pip)
* Documentation

---

## Summary

MemoryOS is a **local-first, model-agnostic memory engine** for AI systems.

It combines:

* working memory (short-term)
* episodic memory (compressed history)
* semantic memory (facts)

With structured facts, adaptive context building, and confidence-aware filtering, MemoryOS provides a clean, extensible foundation for long-term AI memory.
