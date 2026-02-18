from abc import ABC, abstractmethod

class SynonymService(ABC):
    @abstractmethod
    def get_synonyms(self, input_word: str, sentence: str, semantic_threshold: float = 0.7) -> list[tuple[str, float]]:
        pass