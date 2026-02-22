from abc import ABC, abstractmethod

# Provides an interface for getting synonyms for a given input word, taking into account the sentence context and a semantic similarity threshold. Returns a list of tuples containing the synonym and its similarity score to the original word in the given context.
class SynonymService(ABC):
    @abstractmethod
    def get_synonyms(self, input_word: str, sentence: str, semantic_threshold: float = 0.7) -> list[tuple[str, float]]:
        pass