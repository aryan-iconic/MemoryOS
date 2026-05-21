"""Token budget management for compression."""
import logging

logger = logging.getLogger(__name__)

class TokenBudgetManager:
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.current_tokens = 0
    
    def can_add(self, tokens:int) -> bool:
        return self.current_tokens + tokens <= self.max_tokens

    def add_tokens(self, tokens: int) -> bool:
        if self.can_add(tokens):
            self.current_tokens += tokens
            return True
        return False

    def reset(self):
        self.current_tokens = 0
