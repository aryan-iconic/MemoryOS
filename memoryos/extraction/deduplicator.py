"Deduplication logic for extracted facts."
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from ..models import Fact

class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold

    def deduplicate_facts(self, facts: List[Fact]) -> List[Fact]:

        unique_facts: List[Fact] = []

        for fact in facts:
            duplicate_index = self._find_duplicate_index(fact, unique_facts)

            if duplicate_index is None:
                unique_facts.append(fact)
                continue

            existing_fact = unique_facts[duplicate_index]

            if fact.confidence > existing_fact.confidence:
                unique_facts[duplicate_index] = fact

        return unique_facts

    def filter_new_facts(
        self,
        new_facts: List[Fact],
        existing_facts: List[Fact],
    ) -> List[Fact]:

        filtered_facts: List[Fact] = []

        for fact in new_facts:
            duplicate_index = self._find_duplicate_index(fact, existing_facts)

            if duplicate_index is None:
                filtered_facts.append(fact)

        return self.deduplicate_facts(filtered_facts)

    def is_duplicate(
        self,
        new_fact: Fact,
        existing_facts: List[Fact],
    ) -> bool:
        return self._find_duplicate_index(new_fact, existing_facts) is not None

    def are_facts_similar(
        self,
        fact1: Fact,
        fact2: Fact,
    ) -> bool:

        if fact1.type != fact2.type:
            return False

        text1 = self._normalize_text(fact1.content)
        text2 = self._normalize_text(fact2.content)

        if text1 == text2:
            return True

        similarity = SequenceMatcher(None, text1, text2).ratio()

        return similarity >= self.similarity_threshold

    def delete_duplicate_facts(self, facts: List[Fact]) -> List[Fact]:
        return self.deduplicate_facts(facts)

    def _find_duplicate_index(
        self,
        new_fact: Fact,
        existing_facts: List[Fact],
    ) -> Optional[int]:
        for index, fact in enumerate(existing_facts):
            if self.are_facts_similar(new_fact, fact):
                return index

        return None

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = " ".join(text.split())

        return text