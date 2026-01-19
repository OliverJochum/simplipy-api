To start API locally:

1. Create virtual environment: `python -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install fastapi: `pip install fastapi[standard]`
3.1 General packages:
`pip install python-dotenv`
3.2 Llama installations:
`pip install llama-cpp-python`
`pip install huggingface-hub`
3.3 OpenAI installations:
`pip install -U "langchain[openai]"`
4. Launch API: `fastapi dev main.py`

Run spaCy german mode: `poetry run python -m spacy download de_core_news_md`

Download ODE wordnet: 

```bash
poetry run python - <<EOF
import wn

wn.download("odenet:1.4")
EOF
```

Dependency management via poetry:

Poetry install: `poetry install --extras all`

Run venv via poetry:

`poetry env activate`
`poetry run fastapi dev main.py`