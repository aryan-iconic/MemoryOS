from dataclasses import dataclass
from typing import List
import re

from ..models import FactType


@dataclass
class ExtractionPattern:
    pattern_id: str
    description: str
    fact_type: FactType
    regex: str
    template: str
    base_confidence: float


DEFAULT_PATTERNS: List[ExtractionPattern] = [
    ExtractionPattern(
    pattern_id="identity_name",
    description="Extract user's name",
    fact_type="identity",
    regex=r"\bmy name is\s+(?P<value>[A-Za-z][A-Za-z\s'-]{1,50}?)(?=\s*(?:[.!?,;]|$|\band\b|\bbut\b))",
    template="User's name is {value}.",
    base_confidence=0.95,   
    ),
    ExtractionPattern(
        pattern_id="preference_prefer",
        description="Extract explicit user preference",
        fact_type="preference",
        regex=r"\bi prefer\s+(?P<value>[^.?!;]+)",
        template="User prefers {value}.",
        base_confidence=0.90,
    ),
    ExtractionPattern(
        pattern_id="preference_like",
        description="Extract user likes",
        fact_type="preference",
        regex=r"\bi like\s+(?P<value>[^.?!;]+)",
        template="User likes {value}.",
        base_confidence=0.80,
    ),
    ExtractionPattern(
        pattern_id="preference_dislike",
        description="Extract user dislikes",
        fact_type="preference",
        regex=r"\bi dislike\s+(?P<value>[^.?!;]+)",
        template="User dislikes {value}.",
        base_confidence=0.80,
    ),
    ExtractionPattern(
        pattern_id="goal_my_goal",
        description="Extract explicit user goal",
        fact_type="goal",
        regex=r"\bmy goal is\s+(?P<value>[^.?!;]+)",
        template="User's goal is {value}.",
        base_confidence=0.90,
    ),
    ExtractionPattern(
        pattern_id="goal_want_to_build",
        description="Extract project/building goal",
        fact_type="goal",
        regex=r"\bi want to build\s+(?P<value>[^.?!;]+)",
        template="User wants to build {value}.",
        base_confidence=0.85,
    ),
    ExtractionPattern(
        pattern_id="context_working_on",
        description="Extract current work context",
        fact_type="context",
        regex=r"\bi am working on\s+(?P<value>[^.?!;]+)",
        template="User is working on {value}.",
        base_confidence=0.85,
    ),
    ExtractionPattern(
        pattern_id="decision_decided",
        description="Extract user decision",
        fact_type="decision",
        regex=r"\bi decided to\s+(?P<value>[^.?!;]+)",
        template="User decided to {value}.",
        base_confidence=0.88,
    ),
    ExtractionPattern(
        pattern_id="remember_that",
        description="Extract explicit remember instruction",
        fact_type="context",
        regex=r"\bremember that\s+(?P<value>[^.?!;]+)",
        template="User asked to remember that {value}.",
        base_confidence=0.95,
    ),
]


def compile_pattern(pattern: ExtractionPattern) -> re.Pattern:
    return re.compile(pattern.regex, flags=re.IGNORECASE)