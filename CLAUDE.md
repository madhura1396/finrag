# FinRAG

Personal finance assistant I use for real. Bank statements in, transactions to Postgres + pgvector, questions in plain English answered by routing to SQL or vector search. Ollama (llama3.2) and all-MiniLM-L6-v2 run locally.

I am also using this project to learn senior engineering, so every design choice must be explained and defensible.

## Scope

- Today: parses Wells Fargo credit card PDF statements only (`app/extractor.py` is Wells Fargo specific).
- Target: statements from any bank and any account type.
- `extractor.py` is being replaced. Do not add more Wells Fargo specific logic. New parsing code must not assume one bank's layout.
- How we support multiple banks is an open architectural decision (see below). Do not pick an approach on your own.

## Hard constraints

- Privacy: statement content never leaves this machine. Any option that would send statement text, transactions or embeddings to an external API must say so as its first con.
- Money is `Decimal` / `Numeric`, never float.
- A parse failure must never silently become a wrong number. Fail loudly or flag the row.

## Architectural decisions: evaluate with me first

An architectural decision is any of:
- a new dependency, model or external service
- a schema change or migration
- a new module, or a change to the boundary between modules
- a contract: API request or response shape, a function signature used across modules, a file or data format
- the concurrency, transaction, error handling or security model
- anything expensive to reverse

Before making one:
1. State the decision in one sentence and the requirement that forces it.
2. Give 2 to 4 real options. Include "keep what we have" when it applies.
3. For each option: how it works, pros, cons, cost (complexity, latency, money, privacy), and what breaks first at scale, on which axis, at roughly what number.
4. Recommend one, and name the condition that would change your recommendation.
5. Stop and wait for my choice. Write no code for it until I choose.
6. After I choose, write `docs/adr/NNNN-short-title.md` with: Status, Date, Context, Options considered, Decision, Consequences.

Not architectural, so just do it inside the agreed task: naming, refactors inside one function, formatting, test cases for behaviour we already agreed.


## Source of truth

- The code is the source of truth. `README.md` and `decisions.md` are out of date in places and contradict the code and each other (for example, four descriptions of the embedding text; `app/embedder.py` is correct).
- `decisions.md` is legacy. New decisions go in `docs/adr/`. Never quote `decisions.md` or the README as fact without checking the code.

## Layout and dependency direction

```
app/            production code
  main.py         FastAPI endpoints
  pipeline.py     write path orchestration
  extractor.py    PDF text + Wells Fargo state machine parser (being replaced)
  llm.py          Ollama enrichment (merchant, category)
  embedder.py     sentence-transformers embeddings
  query_router.py read path: classify, SQL generation, semantic search, formatting
  models.py       SQLAlchemy models, engine, sessions
  eval_search.py  retrieval eval script (runs on import)
  reembed.py      one-off re-embedding script
docs/adr/       architecture decision records
experiments/    model-internals research, not part of the app
tests/          manual print scripts, not pytest yet
```

- `app/` must never import from `experiments/` or `tests/`.
- Write path: `main.py` -> `pipeline.py` -> `extractor.py`, `llm.py`, `embedder.py` -> `models.py`.
- Read path: `main.py` -> `query_router.py` -> `models.py`.

## Running locally

Prereqs: Postgres with pgvector, Ollama running with `llama3.2` pulled, Python venv at `.venv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
# .env needs DATABASE_URL=postgresql://user:password@localhost:5432/finrag
python -c "from app.models import create_tables; create_tables()"   # not called automatically
uvicorn app.main:app --reload
python demo.py                                                      # needs the server running
```

- Ollama URL `http://localhost:11434` and model `llama3.2` are hardcoded in `extractor.py`, `llm.py` and `query_router.py`.
- `experiments/` needs torch, transformers, matplotlib, seaborn and llama_cpp, which are not in `requirements.txt`.

## Tests

- No pytest suite yet. `tests/` holds manual scripts: `python -m tests.test_extractor path/to/statement.pdf`. They need live Ollama and a real PDF.
- New tests must not call Ollama or Postgres unless I ask for an integration test. Mock the HTTP calls.
- Never commit real statements as test fixtures. Use synthetic statements.

## Git

- Never commit to `main`. One branch and one PR per change: `fix/<name>`, `feat/<name>`, `chore/<name>`, `adr/<name>`.
- Commit messages: imperative subject under 72 characters, then a body that says why.
- Never push, force push, rebase or delete branches unless I ask.
- Run `git status` before and after any change and tell me what changed.

## Data safety

- Never read `.env` or anything in `uploads/`. They hold credentials and real bank statements.
- Never run SQL that writes to the database (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER) unless I ask.
- Never paste real transaction data into commit messages, PRs, ADRs or docs.

## Working style

- One task per session. Before starting, restate: goal, files in scope, out of scope, how we know it is done.
- Use a subagent for broad searches ("where is X called") so only the answer comes back.
- When unsure, ask one question rather than guessing.

## Review

After any change to files in app/ or tests/, run the senior-reviewer subagent on the diff before committing. Report its verdict and top issues to me.
