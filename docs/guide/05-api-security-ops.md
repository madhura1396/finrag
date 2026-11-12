# API security and ops: endpoints, loopback-only, background maintenance, running locally

Status: drafted 2026-11-12. Verified against `app/main.py`, `scripts/check_profiles.py`, `demo.py`, `README.md` and `.claude/CLAUDE.md` on `main` at `2fdd28a`. Where this guide and the code disagree, the code wins.

Related guides: [00-overview](00-overview.md) (whole system), [01-ingestion](01-ingestion.md) (`/upload`), [02-categorization](02-categorization.md) (label endpoints), [03-read-path](03-read-path.md) (`/query`), [04-data-and-storage](04-data-and-storage.md) (schema and reset), [06-testing-and-evals](06-testing-and-evals.md) (test suites).

---

## What this doc covers

- Every endpoint and its status codes
- Loopback-only security: why it's the whole defence, and its limits
- Background maintenance and startup behaviour
- Running locally: prerequisites, commands, `.env`
- `check_profiles` — the real-statement safety check
- Data-safety rules from `.claude/CLAUDE.md`

---

## At a glance

| Piece | File | Responsibility |
|---|---|---|
| FastAPI app + middleware | `app/main.py:55-63` | Loopback-only middleware (403 for non-loopback), lifespan startup checks |
| Endpoints | `app/main.py` | `/upload`, `/query`, `/statements`, `/categories`, `/rules`, `/memory`, `/review`, `/changes`, `/changes/preview`, `/changes/apply`, `/enrichment/retry`, `/embeddings/refresh` |
| Background maintenance | `app/main.py:31` | `run_maintenance`: retry pending labels, refresh stale embeddings |
| Startup | `app/main.py:46-52` | `lifespan`: load single profile, run maintenance in an executor |
| Real-statement check | `scripts/check_profiles.py` | PASS/FAIL per PDF, prints only check names — never statement text |
| Data safety | `.claude/CLAUDE.md` | Never read `.env`, `uploads/`, refused log, or `~/.finrag/*` |

- **No authentication.** The label endpoints write real data. Loopback-only access is the whole defence — never expose the API beyond your own machine.
- **Both models run locally:** llama3.2 via Ollama at `localhost:11434`; all-MiniLM-L6-v2 in-process. Nothing leaves the machine.

---

## Every endpoint

| Method | Path | Purpose | Statuses |
|---|---|---|---|
| POST | `/upload` | Upload a PDF statement | 200 (stored + labelled), 400 (not a PDF), 409 (duplicate period), 422 (reconciliation failure, nothing saved) |
| POST | `/query` | Ask a plain-English question | 200 (answered or refused), 400 (empty question), 503 (model unreachable/timed out) |
| GET | `/statements` | List uploaded statements | 200 |
| GET | `/statements/{id}/transactions` | A statement's transactions | 200, 404 (missing statement, or one with no transactions: issue #4.3) |
| GET | `/categories` | Current categories | 200 |
| GET | `/rules` | Label rules (optionally `include_deleted`) | 200 |
| GET | `/memory` | Remembered labels per description key | 200 |
| GET | `/review` | Open review items grouped by key | 200 |
| GET | `/changes` | Change log, newest first | 200 |
| POST | `/changes/preview` | `{kind, params}` → token, data_version, effect | 200, 422 (invalid change) |
| POST | `/changes/apply` | `{kind, params, token}` → change_id, effect | 200, 409 (stale preview / version changed), 422 (invalid change) |
| POST | `/enrichment/retry` | Label pending rows | 200 |
| POST | `/embeddings/refresh` | Re-embed stale rows | 200 |

`/upload` returns `{statement_id, period_start, period_end, total_transactions, enriched, needs_review, pending}` and waits for labelling (ADR 0005 option A); embeddings follow in the background.

`/query` returns `{type, answer, answer_exact, rewrite, understood_as, interpretation, result, rows, rows_total, notes, query}`. A refusal is a normal 200 with `type: refused` and a reason — never a 500 for a misunderstood question.

---

## Loopback-only security (`app/main.py:58-63`)

```python
LOOPBACK_HOSTS = {"127.0.0.1", "::1"}

@app.middleware("http")
async def loopback_only(request: Request, call_next):
    host = request.client.host if request.client else None
    if host not in LOOPBACK_HOSTS:
        return JSONResponse(status_code=403, content={"detail": "this API only accepts requests from this machine"})
    return await call_next(request)
```

- The app cannot see uvicorn's bind address, but it can see each request's source address — so the check works even if uvicorn is bound to `0.0.0.0`.
- Documented run command binds `127.0.0.1` (uvicorn's default): `uvicorn app.main:app --reload`.
- Binding to localhost does **not** stop requests from a browser page on the same machine. Accepted for a single-user local tool.
- Any UI, container, or remote access needs an authentication decision first (findings recommendation 6).

**The label endpoints write real data with no authentication.** Rules, memory, pins and the change log hold real merchant text and live only in the database. Loopback-only is the whole defence (ADR 0005, decision 4; findings recommendation 6). Fix the `/upload` filename issue (#2.2, ranked first in the findings) before any wider exposure.

---

## Background maintenance (`app/main.py:31-43`)

```python
def run_maintenance() -> None:
    from app.embedder import generate_embedding
    db = new_session()
    try:
        categorization.enrich_pending(db)          # label pending rows
        categorization.refresh_embeddings(db, generate_embedding)  # re-embed stale rows
    except Exception:
        log.exception("background maintenance failed; it will be retried")
    finally:
        db.close()
```

- Safe to run any time: both tasks work from derived state, so a lost run is redone by the next one.
- Runs after `/upload` and after every `/changes/apply` (FastAPI background task).
- Runs at app startup (`lifespan`, `:47-52`) because FastAPI background tasks die on restart.
- Startup also loads the single layout profile (`load_single()`) and refuses to start if it's missing or malformed.

---

## Running locally

**Prerequisites:** Python 3, Postgres with the pgvector extension, Ollama running with `llama3.2` pulled.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create a .env file containing:
# DATABASE_URL=postgresql://user:password@localhost:5432/finrag

# Fresh database: create the tables, mark it production, seed it
python -m scripts.reset_db --env production --yes --confirm-name finrag

uvicorn app.main:app --reload      # binds 127.0.0.1; non-loopback clients get 403
python demo.py                      # needs the server running
```

Every database carries an environment marker (`finrag` = `production`, `finrag_test` = `test`). A connection is refused if the marker is missing or doesn't match `FINRAG_ENV` (default `production`).

### Tests
```bash
pip install -r requirements-dev.txt          # pytest==9.2.0
python -m pytest                              # unit tests: no Ollama, Postgres or .env
FINRAG_TEST_DATABASE_URL=<finrag_test URL> python -m pytest -m integration   # needs a finrag_test Postgres, not Ollama
python -m evals.read_path.run                 # form eval; needs Ollama
```

### Real-statement check (`scripts/check_profiles.py`)
```bash
python -m scripts.check_profiles <pdf or directory> [...] [--numbered]
```
- Prints exactly one line per file: `PASS <name>` or `FAIL <name> <check>[:<section>] ...`
- Every word after the name is a fixed check name from `app.statement.CHECKS`.
- **Never prints amounts, dates, merchants, labels, mismatch values or tracebacks.** An unexpected error prints `FAIL <name> unexpected_error`.
- Reads PDFs only: no database, no model, no writes. Imports only `app.profile`, `app.reader`, `app.statement` — never `app.models` (which would load `.env` and build an engine).
- Exits 1 if any file fails. Run it on each new statement before trusting it (findings lesson: one real statement found two layout facts that hundreds of unit tests could not).

---

## Data-safety rules (`.claude/CLAUDE.md`, Data safety)

These always apply, everywhere in the repo, including `experiments/`:

- Never read `.env` or anything in `uploads/` (credentials and real bank statements).
- Never read the refused-question log (`FINRAG_REFUSED_LOG`, default `~/.finrag/refused_questions.log`) or `~/.finrag/search_eval.yaml` (your questions and real merchant names).
- Never run SQL that writes to the database (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER) unless asked. Exception: the integration suite may write to `finrag_test` (never `finrag`), and `reset_db` may target `finrag_test` in tests. Running `reset_db` against `finrag` needs an explicit ask.
- Rules, label memory, pins and the change log hold real merchant text and live only in the database — never copy them into the repo, fixtures or docs.
- Never paste real transaction data into commit messages, PRs, ADRs or docs.
- The refused-question log holds only refused questions (unsupported or not understood). Data-state refusals (no merchant match, no coverage) are not logged. Claude never reads it.

---

## Known limits and loose ends

- **No authentication; loopback only.** The second user, or any network exposure, needs an auth decision (findings recommendation 6).
- **`/upload` uses the raw client filename** — path traversal and same-name overwrite (findings issue #2.2, ranked first). Keep the basename only, check the path stays in `UPLOAD_DIR`, use a unique stored name.
- **Statement identity is the period alone** — a second card with the same billing dates collides (issue #12; needs an ADR).
- **`reset_db` still lists `trips` in `DATA_TABLES`** — harmless but stale (trips removed as out of scope).
- **Background tasks die with the process** — mitigated by startup retry and the on-demand endpoints, but a crash between upload and maintenance leaves rows pending until the next trigger.
