# FinRAG

A personal finance assistant that runs entirely on one laptop. PDF credit card statements go in, every transaction is stored in Postgres with a category, a merchant name and an embedding, and plain-English questions come back with exact answers.

- **In:** PDF statements. Today only the Wells Fargo credit card layout is supported (ADR 0002).
- **Stored:** signed transactions in Postgres, each with a 384-dimension embedding (pgvector).
- **Out:** answers to questions such as "How much did I spend on travel in March?", "Where did I spend the most last month?" or "Any charges that look like parking?".

Both models run locally: llama3.2 through Ollama, and all-MiniLM-L6-v2 through sentence-transformers. No statement text, transaction or embedding leaves the machine.

The project is also a deliberate exercise in senior engineering. Every architectural choice is recorded in [`docs/adr/`](docs/adr/) with its options, trade-offs and consequences.

---

## Three rules the code enforces

**1. Privacy: statement content never leaves the machine.**
Ollama is reached only at `localhost`, and embeddings are computed in-process. The API refuses every non-loopback client with a 403. Real merchant text lives only in the database or under `~/.finrag/`, never in the repo.

**2. Money is `Decimal`, never float.**
Amounts are stored as `Numeric(10, 2)`, parsed into `Decimal`, and checked against the PDF's own totals by exact equality. Every number on the read path is computed in `app/calculations.py`, and money leaves the API as strings.

**3. Fail loudly: a parse failure never becomes a wrong number.**
If a statement doesn't reconcile with its printed section totals and Account Summary, nothing is saved and the upload gets a 422 that lists every mismatch. When the model can't label a row, the row stays `pending`. When the model misreads a question, the question is refused. If Ollama is down, the API answers 503.

---

## How it works

### Write path: upload a statement

```
PDF ──► reader.py + profiles/*.yaml   word positions (PyMuPDF), driven by a YAML profile
    ──► statement.validate            two independent reconciliation checks; any mismatch → 422, nothing saved
    ──► pipeline.py                   one transaction: insert statement + signed rows
    ──► categorization.py             label by section > pin > rule > memory > model (llama3.2, temp 0, JSON schema)
    ──► embedder.py (background)      all-MiniLM-L6-v2 embeddings into pgvector
```

All bank-specific knowledge (headings, total patterns, column anchors, summary labels) lives in `profiles/*.yaml`, so the code in `app/` stays bank-agnostic.

### Read path: ask a question

```
question ──► question_form.py   llama3.2 fills a schema-constrained form; code resolves dates, merchants, defaults
         ──► read_queries.py    one parameterised, date-filtered SELECT (plus vector search for "looks like")
         ──► calculations.py    every number, in Decimal
         ──► answers.py         code writes every fact; an "Understood as" line shows the interpretation
```

The model never sees the data and never writes SQL. An optional rewrite step can rearrange the fact tokens into a sentence, but a checker rejects any rewrite that adds digits or changes a fact (ADR 0004).

### Label changes

Categories are the user's own list. Every change to a rule, pin, category or remembered label is previewed, token-checked against a `data_version`, applied in one transaction, logged in SCD2 history, and can be undone (ADR 0005).

---

## Design principles

| Principle | What it means | Trade-off |
|---|---|---|
| **Code owns every number** | No number the user sees comes from a model. Write-path numbers must reconcile with the PDF, and read-path numbers are computed in `Decimal`. | Only the 12 built question types are answered. Other questions are refused and logged locally. |
| **Reject rather than guess** | Every ambiguous case refuses visibly: an unreconciled statement, an unknown category, an unmatched merchant, a misread question, a rule tie. | More refusals, but never an invented figure. |
| **Profiles hold bank knowledge** | A new bank means a new YAML profile and a synthetic fixture, not a new parser. | No profile selection yet: exactly one profile is loaded. |
| **The user owns labels** | The model may pick only from the user's categories or answer `none_fit`, and its answers are remembered per description. | Memory makes labels consistent, not necessarily correct, until the user corrects them. |
| **Safety is checked at runtime** | Reconciliation, the "Understood as" line, preview tokens and the database environment marker all run on every request, not only in tests. | |

---

## Tech stack

| Piece | Why |
|---|---|
| **FastAPI** + uvicorn | Typed requests, a thread pool for sync handlers, background tasks, a lifespan hook for startup checks. |
| **Postgres + pgvector** | One store for both relational and vector data, so a similarity search can filter by date, sign, category and merchant in the same SQL. Also row locks and constraints. |
| **SQLAlchemy** | Models, parameterised queries, and the connect hook that holds the environment guard. |
| **Ollama llama3.2** | Local, free, JSON-schema-constrained output at temperature 0. |
| **all-MiniLM-L6-v2** | Small (384 dimensions), fast on a CPU, local. |
| **PyMuPDF** | Deterministic word boxes with coordinates, so the reader can tell columns apart. |
| **YAML profiles** | Declarative, reviewable, versioned. The strict loader refuses unknown keys and YAML's implicit typing. |
| **`Decimal`** | Exact cents and exact equality for reconciliation. |

---

## Project layout

```
app/            production code (write path, read path, labels, models, DB guard)
profiles/       bank layout profiles, YAML
scripts/        reset_db.py (guarded schema reset), check_profiles.py (pass/fail on real statements)
docs/adr/       architecture decision records
docs/plans/     implementation plans
evals/          read-path form eval (synthetic questions), search eval example
tests/          unit/ (pytest, no Ollama or Postgres), integration/ (finrag_test only)
experiments/    model-internals research (attention, FFN, quantisation gap); not part of the app
demo.py         example API client
```

---

## Running locally

**Prerequisites:** Postgres with pgvector, Ollama running with `llama3.2` pulled, Python 3.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create a .env file containing DATABASE_URL=postgresql://user:password@localhost:5432/finrag

# Fresh database: create the tables, mark it as production, and seed it
python -m scripts.reset_db --env production --yes --confirm-name finrag

uvicorn app.main:app --reload     # binds 127.0.0.1; non-loopback clients get 403
python demo.py                    # needs the server running
```

Every database carries an environment marker (`finrag` is `production`, `finrag_test` is `test`). A connection is refused if the marker is missing or doesn't match `FINRAG_ENV`.

### API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload a PDF statement |
| POST | `/query` | Ask a question in plain English |
| GET | `/statements` | List statements |
| GET | `/statements/{id}/transactions` | List a statement's transactions |
| GET | `/categories`, `/rules`, `/memory`, `/review`, `/changes` | Inspect labels, rules, memory, review items and the change log |
| POST | `/changes/preview`, `/changes/apply` | Preview, then apply a label change (including undo) |
| POST | `/enrichment/retry`, `/embeddings/refresh` | Re-run labelling or embedding |

The label endpoints write real data with no authentication. Loopback-only access is the whole defence, so never expose this API beyond your own machine.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest                                              # unit tests: no Ollama, Postgres or .env
FINRAG_TEST_DATABASE_URL=<finrag_test URL> python -m pytest -m integration
python -m evals.read_path.run                                 # form eval; needs Ollama
```

---

## Known limits

- Only one layout has been verified so far: the Wells Fargo credit card.
- llama3.2 often misreads questions: whole-form accuracy is 37.8% on the synthetic eval. The "Understood as" line makes every misread visible, and a misread never produces an invented number.
- A statement's identity is its billing period alone, so a second card with the same billing dates would collide.
- There is no authentication; the API is loopback-only.

See [`docs/findings-and-recommendations.md`](docs/findings-and-recommendations.md) for the full list.
