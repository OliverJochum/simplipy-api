from bert_score import score
import torch

from analysis.score import Score

# Implementation of Score for BERTScore, which calculates the BERTScore F1 score for candidate texts against reference texts. Uses the "distilbert-base-multilingual-cased" model for German language processing. Higher F1 score indicates better context retention between candidate and reference texts.
class BERTScore(Score):

    def calculate(self, cands: list[str], refs: list[str]) -> float:
        with torch.no_grad():
            P, R, F1 = score(
                cands,
                refs,
                lang="de",
                batch_size=8, 
                model_type="distilbert-base-multilingual-cased",
                device="cpu",     
                verbose=True
            )
            
        return F1.mean().item()