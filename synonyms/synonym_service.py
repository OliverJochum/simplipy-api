import spacy
from nltk.corpus import wordnet as wn

class SynonymService:
    def get_synonyms(self, word: str, sentence: str, semantic_threshold: float) -> list[str]:
        # Get word as token within sentence contex

        # Get all lemmas of token

        # For each lemma, get synonyms, then filter by semantic threshold
        pass