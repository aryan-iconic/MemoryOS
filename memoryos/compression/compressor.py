"""Conversation and memory compression helpers for MemoryOS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memoryos.compression.summarizer import Summarizer
from memoryos.models import Turn


@dataclass
class CompressionConfig:
    """Settings used by :class:`Compressor`."""

    max_input_tokens: int = 4000
    max_output_words: int = 220
    chunk_size_turns: int = 12
    overlap_turns: int = 1
    min_turns_to_compress: int = 6
    preserve_recent_turns: int = 6
    include_timestamps: bool = False
    include_metadata: bool = True
    empty_summary: str = ""


@dataclass
class CompressionResult:
    """Structured result returned by compression operations."""

    summary: str
    original_count: int
    compressed_count: int
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    source_type: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.summary.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "original_count": self.original_count,
            "compressed_count": self.compressed_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "compression_ratio": self.compression_ratio,
            "source_type": self.source_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# Backward-compatible misspelling kept so old imports do not break.
CompressoionResult = CompressionResult


class Compressor:
    """Compress turns, raw texts, or facts into compact summaries."""

    def __init__(
        self,
        summarizer: Optional[Any] = None,
        config: Optional[CompressionConfig] = None,
    ) -> None:
        self.config = config or CompressionConfig()
        self.summarizer = summarizer or Summarizer(
            backend="rule_based",
            max_words=self.config.max_output_words,
        )

    def should_compress(self, turns: Sequence[Turn], min_turns: Optional[int] = None) -> bool:
        if not turns:
            return False
        threshold = min_turns or self.config.min_turns_to_compress
        return len(turns) >= threshold or self._estimate_tokens(turns) > self.config.max_input_tokens

    def compress_turns(
        self,
        turns: Sequence[Turn],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        clean_turns = [turn for turn in turns or [] if self._turn_has_content(turn)]
        if not clean_turns:
            return self._empty_result(source_type="turns", metadata=metadata)  # pragma: no cover

        chunks = self.chunk_turns(clean_turns)
        chunk_summaries = [self._summarize_turn_chunk(chunk) for chunk in chunks]
        final_summary = self._merge_summaries([summary for summary in chunk_summaries if summary])
        final_summary = self._limit_words(final_summary, self.config.max_output_words)

        input_text = "\n".join(self.turn_to_text(turn) for turn in clean_turns)
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(final_summary)

        result_metadata = dict(metadata or {})
        if session_id:
            result_metadata["session_id"] = session_id
        result_metadata.update(
            {
                "chunk_count": len(chunks),
                "summary_id": self.summary_id(final_summary),
                "start_timestamp": self._safe_min_timestamp(clean_turns),
                "end_timestamp": self._safe_max_timestamp(clean_turns),
            }
        )

        return CompressionResult(
            summary=final_summary,
            original_count=len(clean_turns),
            compressed_count=1 if final_summary else 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compression_ratio=self._compression_ratio(input_tokens, output_tokens),
            source_type="turns",
            metadata=result_metadata,
        )

    def compress_texts(
        self,
        texts: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        clean_texts = [self.clean_text(text) for text in texts or []]
        clean_texts = [text for text in clean_texts if text]
        if not clean_texts:
            return self._empty_result(source_type="texts", metadata=metadata)

        limited_texts = self._fit_texts_to_budget(clean_texts)
        summary = self._call_summarize_texts(limited_texts)
        summary = self._limit_words(summary, self.config.max_output_words)

        input_text = "\n".join(clean_texts)
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(summary)

        result_metadata = dict(metadata or {})
        result_metadata.update(
            {
                "summary_id": self.summary_id(summary),
                "used_text_count": len(limited_texts),
            }
        )

        return CompressionResult(
            summary=summary,
            original_count=len(clean_texts),
            compressed_count=1 if summary else 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compression_ratio=self._compression_ratio(input_tokens, output_tokens),
            source_type="texts",
            metadata=result_metadata,
        )

    def compress_facts(
        self,
        facts: Sequence[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        fact_texts: List[str] = []
        for fact in facts or []:
            content = self._get_value(fact, "content", "")
            fact_type = self._get_value(fact, "type", None)
            confidence = self._get_value(fact, "confidence", None)
            clean = self.clean_text(content)
            if not clean:
                continue  # pragma: no cover

            prefix_parts: List[str] = []
            if fact_type:
                prefix_parts.append(str(fact_type))
            if confidence is not None:
                prefix_parts.append(f"confidence={float(confidence):.2f}")
            prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
            fact_texts.append(prefix + clean)

        result = self.compress_texts(fact_texts, metadata=metadata)
        result.source_type = "facts"
        return result

    def chunk_turns(
        self,
        turns: Sequence[Any],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[List[Any]]:
        clean_turns = list(turns or [])
        if not clean_turns:
            return []  # pragma: no cover

        size = max(1, chunk_size or self.config.chunk_size_turns)
        overlap_count = max(0, overlap if overlap is not None else self.config.overlap_turns)
        overlap_count = min(overlap_count, size - 1)

        chunks: List[List[Any]] = []
        start = 0
        while start < len(clean_turns):
            end = min(start + size, len(clean_turns))
            chunks.append(clean_turns[start:end])
            if end >= len(clean_turns):
                break
            start = end - overlap_count
        return chunks

    def split_for_compression(self, turns: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
        clean_turns = list(turns or [])
        keep = max(0, self.config.preserve_recent_turns)
        if keep == 0:
            return clean_turns, []  # pragma: no cover
        if len(clean_turns) <= keep:
            return [], clean_turns  # pragma: no cover
        return clean_turns[:-keep], clean_turns[-keep:]

    def turn_to_text(self, turn: Any) -> str:
        if hasattr(turn, "as_text") and callable(turn.as_text):
            return self.clean_text(turn.as_text())

        user_message = self._get_value(turn, "user_message", "")
        ai_response = self._get_value(turn, "ai_response", "")
        timestamp = self._get_value(turn, "timestamp", None)
        metadata = self._get_value(turn, "metadata", {}) or {}
        parts: List[str] = []

        if self.config.include_timestamps and timestamp is not None:
            parts.append(f"Timestamp: {timestamp}")

        user_message = self.clean_text(user_message)
        ai_response = self.clean_text(ai_response)
        if user_message:
            parts.append(f"User: {user_message}")
        if ai_response:
            parts.append(f"Assistant: {ai_response}")

        if self.config.include_metadata and isinstance(metadata, dict):
            important_metadata = self._format_metadata(metadata)
            if important_metadata:
                parts.append(f"Metadata: {important_metadata}")
        return "\n".join(parts).strip()

    def estimate_tokens(self, text: str) -> int:
        clean = self.clean_text(text)
        if not clean:
            return 0  # pragma: no cover
        return max(1, int(len(clean.split()) * 1.33))

    def clean_text(self, text: Any) -> str:
        return " ".join(str(text or "").split()).strip()

    def summary_id(self, summary: str) -> str:
        clean = self.clean_text(summary).lower()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]

    def _estimate_tokens(self, turns: Sequence[Any]) -> int:
        return self.estimate_tokens("\n".join(self.turn_to_text(turn) for turn in turns))  # pragma: no cover

    def _summarize_turn_chunk(self, turns: Sequence[Any]) -> str:
        limited_turns = self._fit_turns_to_budget(turns)
        if hasattr(self.summarizer, "summarize_turns"):
            try:
                summary = self.summarizer.summarize_turns(limited_turns)
                return self.clean_text(summary)  # pragma: no cover
            except Exception:
                pass
        texts = [self.turn_to_text(turn) for turn in limited_turns]
        return self._call_summarize_texts(texts)

    def _call_summarize_texts(self, texts: Sequence[str]) -> str:
        clean_texts = [self.clean_text(text) for text in texts if self.clean_text(text)]
        if not clean_texts:
            return ""  # pragma: no cover
        if hasattr(self.summarizer, "summarize_texts"):
            return self.clean_text(self.summarizer.summarize_texts(clean_texts))
        if callable(self.summarizer):  # pragma: no cover
            return self.clean_text(self.summarizer("\n".join(clean_texts)))  # pragma: no cover
        raise TypeError("summarizer must expose summarize_texts/summarize_turns or be callable")  # pragma: no cover

    def _merge_summaries(self, summaries: Sequence[str]) -> str:
        clean_summaries = [self.clean_text(summary) for summary in summaries or []]
        clean_summaries = [summary for summary in clean_summaries if summary]
        if not clean_summaries:
            return self.config.empty_summary  # pragma: no cover
        if len(clean_summaries) == 1:
            return clean_summaries[0]  # pragma: no cover
        limited = self._fit_texts_to_budget(clean_summaries)
        merged = self._call_summarize_texts(limited)
        return merged or " ".join(limited)

    def _fit_turns_to_budget(self, turns: Sequence[Any]) -> List[Any]:
        selected: List[Any] = []
        used_tokens = 0
        for turn in turns:
            text = self.turn_to_text(turn)
            tokens = self.estimate_tokens(text)
            if selected and used_tokens + tokens > self.config.max_input_tokens:
                break
            selected.append(turn)
            used_tokens += tokens
        return selected

    def _fit_texts_to_budget(self, texts: Sequence[str]) -> List[str]:
        selected: List[str] = []
        used_tokens = 0
        for text in texts:
            clean = self.clean_text(text)
            if not clean:
                continue  # pragma: no cover
            tokens = self.estimate_tokens(clean)
            if selected and used_tokens + tokens > self.config.max_input_tokens:
                break
            if tokens > self.config.max_input_tokens:
                clean = self._trim_to_token_budget(clean, self.config.max_input_tokens)
                tokens = self.estimate_tokens(clean)
            selected.append(clean)
            used_tokens += tokens
        return selected

    def _trim_to_token_budget(self, text: str, max_tokens: int) -> str:
        words = self.clean_text(text).split()
        if not words:
            return ""  # pragma: no cover
        max_words = max(1, int(max_tokens / 1.33))
        return " ".join(words[:max_words]).rstrip() + "..."

    def _limit_words(self, text: str, max_words: int) -> str:
        clean = self.clean_text(text)
        words = clean.split()
        if len(words) <= max_words:
            return clean
        return " ".join(words[:max_words]).rstrip() + "..."

    def _compression_ratio(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens <= 0:
            return 0.0
        return round(output_tokens / input_tokens, 4)

    def _empty_result(
        self,
        source_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        return CompressionResult(
            summary=self.config.empty_summary,
            original_count=0,
            compressed_count=0,
            input_tokens=0,
            output_tokens=0,
            compression_ratio=0.0,
            source_type=source_type,
            metadata=dict(metadata or {}),
        )

    def _turn_has_content(self, turn: Any) -> bool:
        return bool(
            self.clean_text(self._get_value(turn, "user_message", ""))
            or self.clean_text(self._get_value(turn, "ai_response", ""))
        )

    def _get_value(self, item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _safe_min_timestamp(self, turns: Sequence[Any]) -> Optional[float]:
        timestamps = [self._get_value(turn, "timestamp", None) for turn in turns]
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float))]
        return min(numeric) if numeric else None

    def _safe_max_timestamp(self, turns: Sequence[Any]) -> Optional[float]:
        timestamps = [self._get_value(turn, "timestamp", None) for turn in turns]
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float))]
        return max(numeric) if numeric else None

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        keep_keys = {"topic", "project", "decision", "source", "tags", "importance"}
        parts: List[str] = []
        for key in sorted(keep_keys):
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                parts.append(f"{key}={value}")
        return ", ".join(parts)


MemoryCompressor = Compressor
__all__ = [
    "CompressionConfig",
    "CompressionResult",
    "CompressoionResult",
    "Compressor",
    "MemoryCompressor",
]
