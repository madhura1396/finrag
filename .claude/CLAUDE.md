# FinRAG

Personal finance assistant I use for real. Bank statements in, transactions to Postgres + pgvector, questions in plain English answered by routing to SQL or vector search. Ollama (llama3.2) and all-MiniLM-L6-v2 run locally.

I am also using this project to learn senior engineering, so every design choice must be explained and defensible.

## Scope

- Today: Wells Fargo credit card only (ADR 0002, Decision 1). Target: Wells Fargo, Chase, Discover, Fidelity. PDF statements only.
- Trips removed as out of scope (2026-11-11).
- Extraction design is decided in docs/adr/0002-profile-based-extraction.md. Follow it; do not reopen it without asking me.
- Bank-specific knowledge lives only in profiles/*.yaml. Code in app/ must stay bank-agnostic.
- app/extractor.py is legacy and will be deleted by the plan in docs/plans/extraction-rebuild.md.

## Hard constraints

- Privacy: statement content never leaves this machine. Any option that would send statement text, transactions or embeddings to an external API must say so as its first con.
- Money is `Decimal` / `Numeric`, never float.
- A parse failure must never silently become a wrong number. Fail loudly: reject the statement and save nothing (ADR 0002).

## Code ownership

- Claude writes the code in app/ and tests/. I decide architecture through the decision protocol and review every PR.
- When a change is ready, run senior-reviewer on the diff and fix every blocker, then ask me to confirm the commit.

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
  main.py           FastAPI endpoints; loopback-only middleware
  pipeline.py       write path orchestration
  statement.py      StatementData contract + validation (bank-agnostic, no DB)
  extractor.py      legacy Wells Fargo parser; unused, deleted in PR3
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
  read_queries.py   the read path's five SQL queries (thin; date-filtered)
  calculations.py   every read-path number, pure Python over Decimal
  answers.py        exact answers, the "Understood as" line, the checked placeholder rewrite
  read_config.py    read-path thresholds, timeouts, glue-word list, refused-log path, data types
                    (Range, Form, Refusal); no I/O or DB imports
  models.py         SQLAlchemy models, engine (guarded), session factory (new_session, get_session)
  eval_search.py    search eval; cases from ~/.finrag/search_eval.yaml (outside the repo)
  reembed.py        re-embed stale rows
docs/adr/       architecture decision records
docs/plans/     implementation plans
migrations/     historical SQL; the schema is now rebuilt by scripts/reset_db.py (ADR 0005)
scripts/        reset_db.py (guarded schema reset); check_profiles.py (pass/fail on your real statements)
profiles/       bank layout profiles, YAML
experiments/    model-internals research, not part of the app
evals/          read_path/ form eval (synthetic questions; needs Ollama); search_eval.example.yaml
tests/          unit/ = pytest suite; integration/ = finrag_test suite; top-level test_*.py = manual scripts
```

- `app/` must never import from `experiments/` or `tests/`.
- Write path: `main.py` -> `pipeline.py` -> `profile.py`, `reader.py`, `statement.py`, `categorization.py` -> `category_store.py`, `llm.py`, `embedding_text.py`, `embedder.py` -> `models.py`. `statement.py` imports nothing from `app`. The read path never imports `categorization.py`.
- Setup: `scripts/reset_db.py` -> `db_setup.py` -> `models.py`. `models.py` imports only `db_guard.py`.
- Read path: `main.py` -> `query_router.py` -> `question_form.py`, `read_queries.py`, `calculations.py`, `answers.py`, `embedder.py`, `embedding_text.py`, `category_store.py`, `read_config.py` -> `models.py`. The read path never imports `categorization.py` or `llm.py`.

## Running locally

Prereqs: Postgres with pgvector, Ollama running with `llama3.2` pulled, Python venv at `.venv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
# .env needs DATABASE_URL=postgresql://user:password@localhost:5432/finrag
python -m scripts.reset_db --env production --yes --confirm-name finrag   # fresh database: create, mark production, seed
                                                                   # holds data or marked production: add --allow-production (destroys it)
uvicorn app.main:app --reload                                      # binds 127.0.0.1; the API refuses non-loopback clients
python demo.py                                                      # needs the server running
```

- Ollama URL `http://localhost:11434` and model `llama3.2` are hardcoded in `extractor.py`, `llm.py` and `read_config.py`.
- Form eval: `python -m evals.read_path.run [--rewrite]` (Ollama up). Few-shot examples live in `query_router.TEACH_EXAMPLES`; the eval refuses any overlap with its questions.
- Every database carries an environment marker (`db_environment`): `finrag` is `production`, `finrag_test` is `test`. Connections compare it with `FINRAG_ENV` (default `production`) and refuse on a mismatch or a missing marker.
- The label endpoints (`/changes/*`, `/categories`, `/rules`, `/memory`, `/review`, `/enrichment/retry`, `/embeddings/refresh`) write real data with no authentication. Never expose the API beyond this machine without an authentication decision first.
- `experiments/` needs torch, transformers, matplotlib, seaborn and llama_cpp, which are not in `requirements.txt`.

## Tests

- Unit tests: `pip install -r requirements-dev.txt`, then `python -m pytest` from the repo root. They live in `tests/unit/` and need no Ollama, Postgres or `.env` (`tests/conftest.py` disables `load_dotenv` and sets a dummy `DATABASE_URL`).
- Integration tests (user-approved exception to the no-Postgres rule): `FINRAG_TEST_DATABASE_URL=<finrag_test URL> python -m pytest -m integration`. They live in `tests/integration/`, refuse any database not named `finrag_test`, connect with the environment guard (`FINRAG_ENV=test`), and roll back each test's transaction.
- Manual scripts remain in `tests/` and are excluded from pytest: `python -m tests.test_extractor path/to/statement.pdf`. They need live Ollama and a real PDF.
- New tests must not call Ollama or Postgres unless I ask for an integration test. Mock the HTTP calls.
- Never commit real statements as test fixtures. Use synthetic statements.

## Git

- Never commit to `main`. One branch and one PR per change: `fix/<name>`, `feat/<name>`, `chore/<name>`, `adr/<name>`.
- Commit messages: imperative subject under 72 characters, then a body that says why.
- Before every commit, show me git status, the files to be staged and the commit message, and wait for me to confirm. After I confirm, commit and push the branch with git push -u origin <branch>. Never push to main, never force push, never rebase or delete branches.
- Run `git status` before and after any change and tell me what changed.
- Stage files by name. Never use git add -A or git add .

## Data safety

- Never read `.env` or anything in `uploads/`. They hold credentials and real bank statements.
- Never read the refused-question log (`FINRAG_REFUSED_LOG`, default `~/.finrag/refused_questions.log`) or `~/.finrag/search_eval.yaml`. They hold my questions and real merchant names.
- Never run SQL that writes to the database (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER) unless I ask. Exception: the integration suite may write to `finrag_test` (never `finrag`), and `scripts/reset_db.py` may target `finrag_test` in tests. Running `reset_db` against `finrag` needs my explicit ask.
- Rules, label memory, pins and the change log hold real merchant text and live only in the database. Never copy them into the repo, fixtures or docs.
- Never paste real transaction data into commit messages, PRs, ADRs or docs.
- To check profiles against my real statements, write a script that prints only pass/fail per file, plus on failure the failing check names and section keys (fixed words from code and the profile, never amounts, dates, merchants or other statement text; ADR 0003). I run it; you never read its input files.

## Working style

- One task per session. Before starting, restate: goal, files in scope, out of scope, how we know it is done.
- Use a subagent for broad searches ("where is X called") so only the answer comes back.
- When unsure, ask one question rather than guessing.

## Review

After any change to files in app/ or tests/, run the senior-reviewer subagent on the diff before committing. Report its verdict and top issues to me.
