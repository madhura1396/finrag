# Privacy: statement content never leaves this machine

Applies to the whole repo, including `experiments/`, `evals/` and `scripts/`.

- Statement text, transactions and embeddings never go to an external API or service.
- Any option that would send statement text, transactions or embeddings to an external API must say so as its first con.
- Both models are local: Ollama at `http://localhost:11434` (hardcoded in `app/llm.py` and `app/read_config.py`) and all-MiniLM-L6-v2 in-process.
- The API binds 127.0.0.1 and refuses non-loopback clients. The label endpoints (`/changes/*`, `/categories`, `/rules`, `/memory`, `/review`, `/enrichment/retry`, `/embeddings/refresh`) write real data with no authentication. Never expose the API beyond this machine without an authentication decision first.
