
from typing import List, Dict, Any
from .models import Fact, Turn
from .patterns import DEFAULT_PATTERNS, ExtractionPattern, compile_pattern
from .deduplicator import Deduplicator
from .confidence import ConfidenceScorer

class Extractor:

    def __init__(
        self,
        patterns: Optional[List[ExtractionPattern]] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        deduplicator: Optional[Deduplicator] = None,
        min_confidence: float = 0.65,
    ):
        self.patterns = patterns or DEFAULT_PATTERNS
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.deduplicator = deduplicator or Deduplicator()
        self.min_confidence = min_confidence

    def extract(self, turn: Turn) -> List[Fact]:
        text = turn.user_message.strip()

        if not text:
            return []

        facts: List[Fact] = []

        for pattern in self.patterns:
            compiled = compile_pattern(pattern)

            for match in compiled.finditer(text):
                value = match.group("value").strip()
                value = self._clean_value(value)

                if not self._is_valid_value(value):
                    continue

                confidence = self.confidence_scorer.calculate(
                    text=text,
                    base_score=pattern.base_confidence,
                )

                if confidence < self.min_confidence:
                    continue

                fact = Fact(
                    content=pattern.template.format(value=value),
                    type=pattern.fact_type,
                    confidence=confidence,
                    session_id=turn.session_id,
                    source="conversation",
                    metadata={
                        "pattern_id": pattern.pattern_id,
                        "raw_value": value,
                    },
                )

                facts.append(fact)

        return self.deduplicator.deduplicate_facts(facts)

    def _clean_value(self, value: str) -> str:
        value = value.strip()
        split_markers = [
            " and i ",
            " but i ",
            " because i ",
        ]

        lowered = value.lower()

        for marker in split_markers:
            if marker in lowered:
                index = lowered.index(marker)
                value = value[:index].strip()
                break

        return value.strip(" .,!?:;")

    def _is_valid_value(self, value: str) -> bool:
        if len(value) < 3:
            return False

        if len(value) > 200:
            return False

        junk_values = {
            "ok",
            "okay",
            "yes",
            "no",
            "none",
            "nothing",
            "it",
            "this",
            "that",
        }

        return value.lower() not in junk_values