
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Sequence, Union

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:

    backend: str = "rule_based"
    max_words: int = 180
    max_input_words: int = 1600

    # Never hard-code model.
    # Required only for model-based backends.
    model: Optional[str] = None

    # Never force endpoint.
    # User can pass endpoint or set environment variable.
    endpoint: Optional[str] = None

    temperature: float = 0.2
    timeout: int = 60
    fallback_to_rule_based: bool = True


class BaseSummarizer(Protocol):
    def summarize_texts(self, texts: Sequence[str]) -> str:
        ...

    def summarize_turns(self, turns: Sequence[Any]) -> str:
        ...


class RuleBasedSummarizer:

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig(backend="rule_based")

    def summarize_texts(self, texts: Sequence[str]) -> str:
        clean_texts = []

        for text in texts:
            clean = self._clean_text(text)
            if clean:
                clean_texts.append(clean)

        if not clean_texts:
            return ""

        combined = " ".join(clean_texts)
        combined = self._limit_words(combined, self.config.max_input_words)

        summary = self._compress_basic(combined)

        return self._limit_words(summary, self.config.max_words)

    def summarize_turns(self, turns: Sequence[Any]) -> str:
        user_points: List[str] = []
        assistant_points: List[str] = []

        for turn in turns:
            user_message = self._clean_text(getattr(turn, "user_message", ""))
            ai_response = self._clean_text(getattr(turn, "ai_response", ""))

            if user_message:
                user_points.append(user_message)

            if ai_response:
                assistant_points.append(ai_response)

        parts: List[str] = []

        if user_points:
            user_text = self._limit_words(
                " ".join(user_points),
                max(40, self.config.max_words // 2),
            )
            parts.append(f"User discussed: {user_text}")

        if assistant_points:
            assistant_text = self._limit_words(
                " ".join(assistant_points),
                max(40, self.config.max_words // 2),
            )
            parts.append(f"Assistant responded about: {assistant_text}")

        return self._limit_words(" ".join(parts), self.config.max_words)

    def _compress_basic(self, text: str) -> str:
        sentences = self._split_sentences(text)

        if not sentences:
            return text

        selected: List[str] = []

        important_markers = [
            "my name is",
            "i prefer",
            "i like",
            "i dislike",
            "i want",
            "my goal",
            "i am working on",
            "remember that",
            "i decided",
            "project",
            "build",
            "memory",
            "preference",
            "goal",
            "decision",
        ]

        for sentence in sentences:
            lowered = sentence.lower()

            if any(marker in lowered for marker in important_markers):
                selected.append(sentence)

        if not selected:
            selected = sentences[:5]

        return " ".join(selected)

    def _split_sentences(self, text: str) -> List[str]:
        text = text.replace("?", ".").replace("!", ".")
        parts = text.split(".")

        sentences = []

        for part in parts:
            clean = part.strip()
            if clean:
                sentences.append(clean + ".")

        return sentences

    def _clean_text(self, text: Any) -> str:
        return " ".join(str(text or "").split())

    def _limit_words(self, text: str, max_words: int) -> str:
        words = text.split()

        if len(words) <= max_words:
            return text.strip()

        return " ".join(words[:max_words]).rstrip() + "..."


class LocalHTTPSummarizer:

    def __init__(
        self,
        config: SummaryConfig,
        fallback: Optional[RuleBasedSummarizer] = None,
    ):
        if not config.model:
            raise ValueError(
                "model is required for local HTTP summarizer. "
                "Example: Summarizer(backend='ollama', model='your-model', endpoint='your-endpoint')"
            )

        self.config = config
        self.fallback = fallback or RuleBasedSummarizer(config=config)

    def summarize_texts(self, texts: Sequence[str]) -> str:
        prompt = self._build_prompt("\n".join(texts))
        return self._generate_or_fallback(prompt, texts=texts)

    def summarize_turns(self, turns: Sequence[Any]) -> str:
        text = "\n".join(self._turn_to_text(turn) for turn in turns)
        prompt = self._build_prompt(text)
        return self._generate_or_fallback(prompt, turns=turns)

    def _generate_or_fallback(
        self,
        prompt: str,
        texts: Optional[Sequence[str]] = None,
        turns: Optional[Sequence[Any]] = None,
    ) -> str:
        try:
            return self._call_local_http(prompt)
        except Exception as exc:
            logger.warning("Local HTTP summarizer failed: %s", exc)

            if not self.config.fallback_to_rule_based:
                raise

            if turns is not None:
                return self.fallback.summarize_turns(turns)

            return self.fallback.summarize_texts(texts or [])

    def _call_local_http(self, prompt: str) -> str:
        endpoint = (
            self.config.endpoint
            or os.getenv("MEMORYOS_SUMMARIZER_ENDPOINT")
            or os.getenv("MEMORYOS_OLLAMA_ENDPOINT")
            or os.getenv("OLLAMA_ENDPOINT")
            or os.getenv("OLLAMA_URL")
        )

        if not endpoint:
            raise ValueError(
                "endpoint is required for local HTTP summarizer. "
                "Pass endpoint='your-endpoint' or set MEMORYOS_SUMMARIZER_ENDPOINT."
            )

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)

        summary = (
            parsed.get("response")
            or parsed.get("summary")
            or parsed.get("text")
            or parsed.get("content")
            or ""
        )

        return " ".join(str(summary).split()).strip()

    def _build_prompt(self, text: str) -> str:
        limited_text = self.fallback._limit_words(
            text,
            self.config.max_input_words,
        )

        return f"""
Summarize the following conversation for long-term AI memory.

Rules:
- Keep only durable memory.
- Preserve identity facts, user preferences, goals, projects, decisions, and recurring constraints.
- Remove temporary small talk.
- Remove duplicate statements.
- Do not invent facts.
- Keep it concise and useful for future context.

Conversation:
{limited_text}

Memory summary:
""".strip()

    def _turn_to_text(self, turn: Any) -> str:
        if hasattr(turn, "as_text"):
            return turn.as_text()

        user_message = getattr(turn, "user_message", "")
        ai_response = getattr(turn, "ai_response", "")

        return f"User: {user_message}\nAI: {ai_response}"


class CallableSummarizer:

    def __init__(
        self,
        callable_fn: Callable[[str], str],
        config: Optional[SummaryConfig] = None,
        fallback: Optional[RuleBasedSummarizer] = None,
    ):
        self.callable_fn = callable_fn
        self.config = config or SummaryConfig(backend="callable")
        self.fallback = fallback or RuleBasedSummarizer(config=self.config)

    def summarize_texts(self, texts: Sequence[str]) -> str:
        prompt = self._build_prompt("\n".join(texts))

        try:
            return self._clean_output(self.callable_fn(prompt))
        except Exception as exc:
            logger.warning("Callable summarizer failed: %s", exc)

            if not self.config.fallback_to_rule_based:
                raise

            return self.fallback.summarize_texts(texts)

    def summarize_turns(self, turns: Sequence[Any]) -> str:
        text = "\n".join(self._turn_to_text(turn) for turn in turns)
        prompt = self._build_prompt(text)

        try:
            return self._clean_output(self.callable_fn(prompt))
        except Exception as exc:
            logger.warning("Callable summarizer failed: %s", exc)

            if not self.config.fallback_to_rule_based:
                raise

            return self.fallback.summarize_turns(turns)

    def _build_prompt(self, text: str) -> str:
        text = self.fallback._limit_words(
            text,
            self.config.max_input_words,
        )

        return f"""
Summarize this conversation for AI memory.

Keep:
- identity facts
- user preferences
- long-term goals
- project context
- decisions
- recurring constraints

Remove:
- small talk
- repeated lines
- temporary details
- irrelevant details

Do not invent facts.

Conversation:
{text}

Memory summary:
""".strip()

    def _turn_to_text(self, turn: Any) -> str:
        if hasattr(turn, "as_text"):
            return turn.as_text()

        user_message = getattr(turn, "user_message", "")
        ai_response = getattr(turn, "ai_response", "")

        return f"User: {user_message}\nAI: {ai_response}"

    def _clean_output(self, output: Any) -> str:
        return " ".join(str(output or "").split()).strip()


class Summarizer:


    def __init__(
        self,
        backend: Union[str, Any] = "rule_based",
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_words: int = 180,
        max_input_words: int = 1600,
        temperature: float = 0.2,
        timeout: int = 60,
        callable_fn: Optional[Callable[[str], str]] = None,
        fallback_to_rule_based: bool = True,
        token_budget_manager: Optional[Any] = None,
    ):


        if not isinstance(backend, str):
            token_budget_manager = backend
            backend = "rule_based"

        self.config = SummaryConfig(
            backend=backend,
            model=model,
            endpoint=endpoint,
            max_words=max_words,
            max_input_words=max_input_words,
            temperature=temperature,
            timeout=timeout,
            fallback_to_rule_based=fallback_to_rule_based,
        )

        self.token_budget_manager = token_budget_manager
        self.summary = ""

        self.backend = self._create_backend(callable_fn=callable_fn)

    def summarize_texts(self, texts: Sequence[str]) -> str:
        return self.backend.summarize_texts(texts)

    def summarize_turns(self, turns: Sequence[Any]) -> str:
        return self.backend.summarize_turns(turns)

    def generate_summary(self, texts: Sequence[str]) -> str:

        return self.summarize_texts(texts)

    def add_to_summary(self, text: str) -> bool:

        clean = " ".join(str(text or "").split())

        if not clean:
            return True

        token_count = len(clean.split())

        if self.token_budget_manager is not None:
            allowed = self.token_budget_manager.add_tokens(token_count)

            if not allowed:
                return False

        else:
            existing_words = len(self.summary.split())
            if existing_words + token_count > self.config.max_words:
                return False

        self.summary += " " + clean
        self.summary = self.summary.strip()

        return True

    def get_summary(self) -> str:

        return self.summary.strip()

    def reset(self) -> None:
        self.summary = ""

        if self.token_budget_manager is not None and hasattr(
            self.token_budget_manager,
            "reset",
        ):
            self.token_budget_manager.reset()

    def _create_backend(
        self,
        callable_fn: Optional[Callable[[str], str]] = None,
    ) -> BaseSummarizer:
        backend = self.config.backend.lower().strip()

        if backend in {"rule_based", "local", "fallback", "default"}:
            return RuleBasedSummarizer(config=self.config)

        if backend in {"ollama", "local_llm", "local_http"}:
            return LocalHTTPSummarizer(config=self.config)

        if backend in {"callable", "cloud", "custom"}:
            if callable_fn is None:
                raise ValueError(
                    "callable_fn is required when backend is 'callable', 'cloud', or 'custom'."
                )

            return CallableSummarizer(
                callable_fn=callable_fn,
                config=self.config,
            )

        raise ValueError(f"Unsupported summarizer backend: {self.config.backend}")