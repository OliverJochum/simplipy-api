from bert_score import score

def calculate(cands: list[str], refs: list[str]):
    P, R, F1 = score(cands, refs, lang='de', verbose=True)
    print(f"Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")   
    return f"{F1.mean().item():.4f}"