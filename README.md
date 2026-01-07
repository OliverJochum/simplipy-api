To start API locally:

1. Create virtual environment: `python -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install fastapi: `pip install fastapi[standard]`
4. Llama installations:
`pip install llama-cpp-python`
`pip install huggingface-hub`
4. Launch API: `fastapi dev main.py`