"""Central configuration helpers for MemoryOS.

The project already exposes ``MemoryOSConfig`` from ``memoryos.models``. This
module keeps that public shape compatible while adding validation and convenient
factory methods for future backends.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

from .exceptions import ConfigError
from .models import MemoryOSConfig as _BaseMemoryOSConfig


@dataclass
class MemoryOSConfig(_BaseMemoryOSConfig):
    """Runtime settings for MemoryOS.

    This class extends the dataclass currently defined in ``models.py`` so older
    imports keep working, while new code can import from ``memoryos.config``.
    """

    storage_backend: str = "sqlite"
    vector_backend: str = "memory"
    enable_faiss: bool = False
    auto_create_episodes: bool = False
    episode_turn_window: int = 12
    min_episode_turns: int = 4
    episode_similarity_threshold: float = 0.25
    retrieval_recency_half_life_days: float = 30.0

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None) -> "MemoryOSConfig":
        """Create a config from a mapping, ignoring unknown keys safely."""
        if data is None:
            return cls()

        allowed = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in dict(data).items() if key in allowed}
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_env(cls, prefix: str = "MEMORYOS_") -> "MemoryOSConfig":
        """Create config from environment variables.

        Example: ``MEMORYOS_DB_PATH=./memoryos.db``.
        """
        config = cls()
        env_map = {
            "DB_PATH": ("db_path", str),
            "EMBEDDING_MODEL_NAME": ("embedding_model_name", str),
            "EMBEDDING_DIM": ("embedding_dim", int),
            "WORKING_MEMORY_SIZE": ("working_memory_size", int),
            "SEMANTIC_TOP_K": ("semantic_top_k", int),
            "EPISODIC_TOP_K": ("episodic_top_k", int),
            "MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
            "DUPLICATE_SIMILARITY_THRESHOLD": ("duplicate_similarity_threshold", float),
            "MIN_FACT_CONFIDENCE": ("min_fact_confidence", float),
            "ENABLE_FAISS": ("enable_faiss", _to_bool),
            "FAISS_INDEX_PATH": ("faiss_index_path", str),
            "AUTO_CREATE_EPISODES": ("auto_create_episodes", _to_bool),
        }

        values: dict[str, Any] = {}
        for suffix, (attr, caster) in env_map.items():
            raw_value = os.getenv(f"{prefix}{suffix}")
            if raw_value is not None:
                try:
                    values[attr] = caster(raw_value)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ConfigError(
                        f"Invalid environment value for {prefix}{suffix}",
                        details={"value": raw_value, "target": attr},
                    ) from exc

        merged = asdict(config)
        merged.update(values)
        return cls.from_dict(merged)

    def validate(self) -> None:
        """Validate config values early so runtime errors are easier to debug."""
        positive_ints = {
            "working_memory_size": self.working_memory_size,
            "embedding_dim": self.embedding_dim,
            "semantic_top_k": self.semantic_top_k,
            "episodic_top_k": self.episodic_top_k,
            "max_context_tokens": self.max_context_tokens,
            "response_buffer_tokens": self.response_buffer_tokens,
            "episode_turn_window": self.episode_turn_window,
            "min_episode_turns": self.min_episode_turns,
        }

        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer.", details={name: value})

        for name in ("min_fact_confidence", "duplicate_similarity_threshold"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1.", details={name: value})

        ratios = {
            "working_memory_max_ratio": self.working_memory_max_ratio,
            "facts_max_ratio": self.facts_max_ratio,
            "summaries_max_ratio": self.summaries_max_ratio,
        }
        for name, value in ratios.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1.", details={name: value})

        if not self.db_path:
            raise ConfigError("db_path cannot be empty.")

        if self.storage_backend not in {"sqlite", "custom"}:
            raise ConfigError("Unsupported storage backend.", details={"storage_backend": self.storage_backend})

        if self.vector_backend not in {"memory", "faiss", "custom"}:
            raise ConfigError("Unsupported vector backend.", details={"vector_backend": self.vector_backend})

    def ensure_paths(self) -> None:
        """Create parent directories for local storage paths."""
        for path_value in (self.db_path, self.faiss_index_path):
            path = Path(path_value)
            if path.parent and str(path.parent) != ".":
                path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Config = MemoryOSConfig


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")
