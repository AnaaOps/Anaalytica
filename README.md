# PDF Chatbot (RAG) — Groq Llama3 + Google Gemini Embeddings

This repository provides a Streamlit-based PDF chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline.

Quick start

1. Create and activate a Python 3.8+ venv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your keys:

```bash
cp .env.example .env
# edit .env and add GEMINI_API_KEY and GROQ_API_KEY
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes

- The app expects `GEMINI_API_KEY` to be available in the environment for Gemini embeddings and LLM access (used via `langchain_google_genai`).
- The `app.py` file contains a minimal integration: upload PDFs, process into chunks, build an in-memory Chroma vectorstore, and run a `RetrievalQA` chain.
- This repository provides example scripts (now in the `examples/` folder) demonstrating how to call Gemini/Groq; adapt as needed.

Missing / next steps

- Pin exact package versions in `requirements.txt` once you confirm compatible versions for your environment.
- Add CI and unit tests for the ingestion and QA flow.
- Optionally persist the vector store to disk (Chroma `persist_directory`) for reuse between runs.

If you want, I can:
- pin working package versions in `requirements.txt`, or
- add a small sample PDF and a smoke test to verify the pipeline end-to-end.
