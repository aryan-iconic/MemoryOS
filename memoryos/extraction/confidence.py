"Confidence scoring for extracted facts."


class ConfidenceScorer:
    UNCERTAIN_WORDS = [
        "maybe",
        "i think",
        "probably",
        "not sure",
        "might",
        "could be",
    ]

    EXPLICIT_WORDS = [
        "remember that",
        "my name is",
        "i prefer",
        "my goal is",
        "i decided",
    ]

    def calculate(
        self,
        text: str,
        base_score: float,
    ) -> float:
        score = base_score
        lowered = text.lower()

        if any(word in lowered for word in self.UNCERTAIN_WORDS):
            score -= 0.10  # pragma: no cover

        if any(word in lowered for word in self.EXPLICIT_WORDS):
            score += 0.05

        return self._clamp(score)

    def _clamp(self, score: float) -> float:
        score = max(0.0, min(1.0, score))
        return round(score, 4)
