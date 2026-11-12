# app/

Production code. File-level rules for the write path, read path and database modules are in `.claude/rules/`.

## Module map and dependency direction

```
app/
  main.py           FastAPI endpoints; loopback-only middleware
  pipeline.py       write path orchestration
  statement.py      StatementData contract + validation (bank-agnostic, no DB)
  profile.py        strict loader for profiles/*.yaml (the only home of bank strings)
  reader.py         PDF -> StatementData by word positions, driven by the profile
  categorization.py label precedence, rules, memory, preview/apply, history, undo (ADR 0005; writes)
  category_store.py current category names, normaliser, description keys, section/reserved names (read only)
  db_setup.py       create_tables(engine, environment), seed(session), category seed data
  db_guard.py       environment-marker guard on every connection
  llm.py            Ollama labelling: temperature 0, schema-constrained, validated
  embedding_text.py embedded text + stale check (no heavy imports)
  embedder.py       sentence-transformers embeddings
  query_router.py   read path orchestration: form -> rows -> numbers -> answer (ADR 0004)
  question_form.py  question -> validated form; dates and merchants resolved by code
  read_queries.py   the read path's SQL queries (thin; date-filtered)
  calculations.py   every read-path number, pure Python over Decimal
  answers.py        exact answers, the "Understood as" line, the checked placeholder rewrite
  read_config.py    read-path thresholds, timeouts, glue-word list, refused-log path, data types
                    (Range, Form, Refusal); no I/O or DB imports
  models.py         SQLAlchemy models, engine (guarded), session factory (new_session, get_session)
  eval_search.py    search eval; cases from ~/.finrag/search_eval.yaml (outside the repo)
  reembed.py        re-embed stale rows
```

- Bank-specific knowledge lives only in `profiles/*.yaml`. Code in `app/` must stay bank-agnostic.
- `app/` must never import from `experiments/` or `tests/`.
- Write path: `main.py` -> `pipeline.py` -> `profile.py`, `reader.py`, `statement.py`, `categorization.py` -> `category_store.py`, `llm.py`, `embedding_text.py`, `embedder.py` -> `models.py`. `statement.py` imports nothing from `app`.
- Setup: `scripts/reset_db.py` -> `db_setup.py` -> `models.py`. `models.py` imports only `db_guard.py`.
- Read path: `main.py` -> `query_router.py` -> `question_form.py`, `read_queries.py`, `calculations.py`, `answers.py`, `embedder.py`, `embedding_text.py`, `category_store.py`, `read_config.py` -> `models.py`. The read path never imports `categorization.py` or `llm.py`.

## Running locally

Prereqs: Postgres with pgvector, Ollama running with `llama3.2` pulled, Python venv at `.venv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
# .env needs DATABASE_URL=postgresql://user:password@localhost:5432/finrag
python -m scripts.reset_db --env production --yes --confirm-name finrag   # fresh database only; see scripts/CLAUDE.md
uvicorn app.main:app --reload                                      # binds 127.0.0.1; the API refuses non-loopback clients
python demo.py                                                      # needs the server running
```
