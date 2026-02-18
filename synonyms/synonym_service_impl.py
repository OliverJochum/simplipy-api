from synonyms.synonym_service import SynonymService
import spacy
import wn


de_net = wn.Wordnet("odenet:1.4")

class SynonymServiceImpl(SynonymService):
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp

        # only these POS matter for synonym retrieval
        self.pos_map = {
            "NOUN": "n",
            "VERB": "v",
            "ADJ": "a",
            "ADV": "r",
        }
    
    def get_synonyms(self, input_word: str, sentence: str, semantic_threshold: float = 0.7) -> list[tuple[str, float]]:
        # Get word as token within sentence contex
        doc = self.nlp(sentence)
        token = next((t for t in doc if t.text.lower() == input_word.lower()), None)
        if not token or not token.has_vector:
            return []

        wn_pos = self.pos_map.get(token.pos_, None)
        if not wn_pos:
            return []

        # Get all synonyms of token from ODENet WordNet
        wordnet_synonyms = set()
        for synset in de_net.synsets(token.lemma_, pos=wn_pos): # Get all synsets in German that match the tokens lemma and POS (ex. 'Bank' as NOUN can be financial institution or bench), so both synsets will be retrieved
            # for each lemma in synset, normalize and add to synonym set if different from token
            for lemma in synset.lemmas():
                word = lemma.replace('_', ' ')
                if word.lower() != token.lemma_.lower():
                    wordnet_synonyms.add(word)

        # For each lemma, get synonyms, then filter by semantic threshold using spaCy vectors (similar words have high cosine similarity, i.e. close vectors in vector space)
        filtered_synonyms = []
        for synonym in wordnet_synonyms:
            candidate_token = self.nlp.vocab[synonym]
            if candidate_token and candidate_token.has_vector:
                similarity = token.similarity(candidate_token)
                if similarity >= semantic_threshold:
                    filtered_synonyms.append((synonym, round(similarity, 2)))
        # Sort synonyms by similarity score in descending order
        filtered_synonyms.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_synonyms