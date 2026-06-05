"""Custom exceptions used across MemoryOS.

Keeping project-specific exceptions in one place makes the public API easier to
handle and prevents low-level errors from leaking out of storage, retrieval,
indexing, compression, and extraction components.
"""

from __future__ import annotations

from typing import Any, Optional


class MemoryOSError(Exception):
    """Base class for all MemoryOS-specific exceptions."""

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message} | details={self.details}"


class ConfigError(MemoryOSError):
    """Raised when MemoryOS configuration is invalid."""


class ValidationError(MemoryOSError):
    """Raised when a user-provided value cannot be accepted."""


class StorageError(MemoryOSError):
    """Raised for storage backend failures."""


class DatabaseError(StorageError):
    """Raised for SQLite/Postgres connection, schema, or query failures."""


class SerializationError(StorageError):
    """Raised when MemoryOS cannot serialize or deserialize stored data."""


class EmbeddingError(MemoryOSError):
    """Raised when embedding generation fails."""


class ExtractionError(MemoryOSError):
    """Raised when fact extraction fails."""


class CompressionError(MemoryOSError):
    """Raised when conversation compression or summarization fails."""


class RetrievalError(MemoryOSError):
    """Raised when memory retrieval fails."""


class RankingError(RetrievalError):
    """Raised when ranking memory results fails."""


class IndexBackendError(MemoryOSError):
    """Raised when a vector index backend fails."""


class DependencyNotInstalledError(MemoryOSError):
    """Raised when an optional dependency such as FAISS is missing."""


class MemoryNotFoundError(MemoryOSError):
    """Raised when a requested memory record does not exist."""


# Backward-compatible aliases that are convenient in imports.
ConfigurationError = ConfigError
MemoryValidationError = ValidationError
VectorIndexError = IndexBackendError
