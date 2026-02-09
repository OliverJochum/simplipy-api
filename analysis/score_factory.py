from analysis.bertscore import BERTScore
from analysis.score import Score


SCORE_REGISTRY: dict[str, type[Score]] = {
    "bertscore": BERTScore,

}

def create_score(kind: str) -> Score:
    try:
        cls = SCORE_REGISTRY[kind]
        return cls()
    except KeyError:
        raise ValueError(f"Unknown score type: {kind}")