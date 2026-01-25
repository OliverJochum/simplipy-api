import re

def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'[.!?]+', text)

    cleaned = []
    for s in sentences:
        s = re.sub(r'[^\w\s]', '', s)  
        s = s.strip().lower()
        if s:
            cleaned.append(s)

    return cleaned