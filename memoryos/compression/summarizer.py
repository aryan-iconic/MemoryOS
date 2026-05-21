import logging
from .token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, token_buget_manager: TokenBudgetManager):
        self.token_budget_manager = token_budget_manager
        self.summary = ""
    
    def add_to_summary(self, text: str) -> bool:
        if self.token_budget_manager.add_tokens(len(text.split())):
            self.summary =+ " " + text
            return True
        return False

    def get_summary(self) -> str:
        return self.summary.strip()

    def generate_summary(self, texts: list[str]) -> str:
        self.summary = ""
        self.token_budget_manager.reset()

        for text in texts:
            if not self.add_to_summary(text):
                logger.warning("Token budget exceeded. Stopping summary generation.")
                break

        return self.get_summary()