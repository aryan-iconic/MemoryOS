"""Summarizer contracts for MemoryOS.

A summarizer turns raw conversation text or turns into compact durable memory.
This interface intentionally does not depend on any model provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class SummarizerInterface(ABC):
    """Abstract base class for summary providers."""

    @abstractmethod
    def summarize_texts(self, texts: Sequence[str]) -> str:
        """Summarize a sequence of raw text chunks."""
        raise NotImplementedError

    @abstractmethod
    def summarize_turns(self, turns: Sequence[Any]) -> str:
        """Summarize conversation turns.

        A turn can be a MemoryOS Turn object or any object/dict with equivalent
        user/assistant message fields, depending on the concrete implementation.
        """
        raise NotImplementedError


# Shorter alias used by some implementations.
BaseSummarizer = SummarizerInterface


__all__ = ["SummarizerInterface", "BaseSummarizer"]
