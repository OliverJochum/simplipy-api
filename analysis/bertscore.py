from bert_score import score
import torch

def calculate(cands: list[str], refs: list[str]):
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

    f1 = F1.mean().item()
    return f"{F1.mean().item():.4f}"