"""Fact extraction utilities for MemoryOS."""

from memoryos.extraction.confidence import ConfidenceScorer
from memoryos.extraction.deduplicator import Deduplicator
from memoryos.extraction.extractor import Extractor
from memoryos.extraction.patterns import (
    DEFAULT_PATTERNS,
    ExtractionPattern,
    compile_pattern,
)
from memoryos.models import Fact, Turn

__all__ = [
    "Extractor",
    "DEFAULT_PATTERNS",
    "ExtractionPattern",
    "compile_pattern",
    "Deduplicator",
    "ConfidenceScorer",
    "Fact",
    "Turn",
]
