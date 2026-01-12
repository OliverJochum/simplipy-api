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

Dependency management via poetry:

Poetry install: `poetry install --extras all`

Run venv via poetry:

`poetry env activate`
`poetry run fastapi dev main.py`