from abc import ABC, abstractmethod

class Score(ABC):

    # Calculate score for candidate text against reference text (generally used for context retention scores, e.g. BERTScore)
    # Returns a float score value, higher is better
    @abstractmethod
    def calculate(self, cands: list[str], refs: list[str]) -> float:
        pass