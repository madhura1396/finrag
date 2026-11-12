# Testing and evals: unit suite, integration suite, form eval, search eval

Status: drafted 2026-11-12. Verified against `pytest.ini`, `tests/conftest.py`, `tests/integration/conftest.py`, `tests/CLAUDE.md`, `evals/CLAUDE.md`, `evals/read_path/run.py`, `tests/fixtures/make_wells_fargo.py` and `docs/findings-and-recommendations.md` on `main` at `2fdd28a`. Where this guide and the code disagree, the code wins.

Related guides: all of them — the tests and evals cover every area.

Decision: test structure follows ADR 0004 ("Testing the numbers") and ADR 0005 (integration suite guard, marker, `CLAUDE.md` carve-out).

---

## What this doc covers

- Unit tests: scope, safety rails, what they cover
- Integration tests: `finrag_test` database, guard, rollback, what they cover
- Manual test script (`test_llm.py`)
- Form eval (read path): how it runs, teach/test split, measured results
- Search eval: format, where cases live, measured results
- Counts and measured numbers from the wrap-up (2026-11-11)

---

## At a glance

| Suite | Command | Needs | Safety rails |
|---|---|---|---|
| Unit | `python -m pytest` | nothing | No Ollama, no Postgres, no `.env` (`tests/conftest.py` neutralizes `load_dotenv` and sets a dummy `DATABASE_URL`) |
| Integration | `FINRAG_TEST_DATABASE_URL=<url> python -m pytest -m integration` | `finrag_test` Postgres with pgvector | URL's database must be exactly `finrag_test`; environment marker checked; each test rolled back |
| Form eval | `python -m evals.read_path.run [--rewrite]` | Ollama with `llama3.2` | Synthetic questions only; refuses overlap with `TEACH_EXAMPLES` |
| Search eval | `python -m app.eval_search` | your own database (`DATABASE_URL`) + cases in `~/.finrag/search_eval.yaml` | Cases live outside the repo; example with invented names in repo |
| Manual | `python -m tests.test_llm` | live Ollama | Excluded from pytest via `collect_ignore` |

- `pytest.ini`: `testpaths = tests/unit tests/integration`; `addopts = -m "not integration"` (integration skipped by default)
- Totals at wrap-up (2026-11-11): **598 unit, 60 integration** tests
- Measured form eval: **31/82 (37.8%)** whole-form correct; baseline 31/79 (39.2%)

---

## Unit tests (`tests/unit/`)

### Safety rails (`tests/conftest.py`)
Runs **before any test module imports `app`**:
- `load_dotenv` is neutralized → tests never read `.env`
- `DATABASE_URL` set to a dummy `postgresql://unit-test:unit-test@127.0.0.1:1/unit-test` (the engine is lazy, never connected)
- `FINRAG_ENV=test` → every database connection checks the marker; a test that reaches a `production`-marked database fails on connect and writes nothing
- Autouse fixture resets the write-path session factory after every test, even one that crashes
- Autouse fixture redirects `FINRAG_REFUSED_LOG` into `tmp_path` → no test appends to the real refused-question log

Unit tests **never touch Ollama, Postgres, `.env` or `uploads/`** (`tests/CLAUDE.md`).

### What they cover
| Test file | What it tests |
|---|---|
| `test_calculations.py` | Every read-path number (pure `Decimal`) |
| `test_answers.py` | Answer rendering, placeholder rewrite checks |
| `test_question_form.py` | Form validation, date/merchant resolution |
| `test_statement.py` | `StatementData`, `validate()`, all 15 check names |
| `test_reader.py` | PDF word-position reading |
| `test_profile.py` | Strict YAML profile loader |
| `test_pipeline.py` | Upload orchestration, validation-before-storage |
| `test_categorization.py` | Precedence, rules, memory, preview/apply, undo, enrichment |
| `test_categories_api.py` | Category/rules/memory/review/changes endpoints |
| `test_query_api.py` | `/query` responses, refusals |
| `test_upload_api.py` | `/upload` outcomes |
| `test_check_profiles.py` | `check_profiles` output vocabulary |

New tests must not call Ollama or Postgres unless an integration test is explicitly requested — mock the HTTP calls (`tests/CLAUDE.md`).

---

## Integration tests (`tests/integration/`)

**User-approved exception** to the "no Postgres in tests" rule (ADR 0004 and ADR 0005). They run against a throwaway `finrag_test` database on local Postgres with pgvector 0.8.2 (created 2026-11-11).

### Safety rails (`tests/integration/conftest.py`)
All checked **before any connection**:
1. `FINRAG_TEST_DATABASE_URL` must be set, otherwise every test is skipped
2. The URL's database must be exactly `finrag_test` — otherwise the suite fails
3. Before dropping anything, the database's own marker is read: only a database marked `test`, or an unmarked one holding no user data, may be dropped
4. The test engine carries the environment guard, and `FINRAG_ENV=test` → a `production`-marked database refuses on connect
5. Every test runs inside a transaction (outer transaction + savepoints) that is rolled back afterwards — nothing persists
6. Schema built fresh with an unguarded engine (setup writes the marker), then tests get a guarded engine

### What they cover
| Test file | What it tests |
|---|---|
| `test_upload_pipeline.py` | Upload → validate → store → label end-to-end against the real DB |
| `test_read_queries.py` | The four thin SQL queries (rows, periods, merchant names, search) |
| `test_categorization_db.py` | Label changes, SCD2 history, change log, undo against the DB |
| `test_locking_and_reset.py` | `data_version` locking, preview token staleness (409), reset guards |

---

## Manual test script

`tests/test_llm.py` calls live Ollama. It is a **manual script**, not a pytest test: listed in `collect_ignore` (`tests/conftest.py:12`) so it stays out of `pytest tests/` and bare `pytest`.

Never commit real statements as test fixtures — use synthetic statements (`tests/CLAUDE.md`).

---

## Form eval (read path)

`python -m evals.read_path.run [--rewrite]` — manual, needs Ollama with `llama3.2` (`evals/CLAUDE.md`).

### How it works
- Sends each synthetic question from `evals/read_path/questions.yaml` to the served model through the production prompt and few-shot set (`query_router.TEACH_EXAMPLES`)
- Compares the returned form field by field against the expected form
- Prints accuracy overall, per type and per field
- With `--rewrite`, also measures how often placeholder rewrites pass the check

### Teach/test split
- Few-shot examples live in `app/query_router.py` (`TEACH_EXAMPLES`) — the app uses them at runtime, so `app/` must not import `evals/`
- The eval **refuses to run** if any test question overlaps `TEACH_EXAMPLES` (`run.py:54-58`)
- Questions are synthetic only: invented merchants, committed to the repo; never add a real question or merchant (`evals/CLAUDE.md`)
- A local set of real, user-labelled questions was declined, to avoid storing asked questions

### Measured results (wrap-up, 82 questions, served Q4 llama3.2 through Ollama, temperature 0)

| Measure | Result |
|---|---|
| Whole form correct | 31/82 (37.8%); baseline 31/79 (39.2%) |
| Question type correct | 57–63 of 82 |
| Weakest types | `search` 0/7, `largest` 0/4, `savings_opportunities` 0/4, `ranking` 1/8 |
| Weakest field | `period`, 46–48 of 72 (invented years, unfilled `--MM`, extra second period) |
| Latency | median ≈ 1.0 s per question |

Prompt-side fixes have hit their limit on this model (findings section 3). A larger local 7–8B model is the recommended follow-up (findings recommendation 2), needs a new-model ADR.

---

## Search eval

`python -m app.eval_search` — runs against the user's own database (`DATABASE_URL`), at the production cutoff and cap. It prints hits and misses per query, so its output contains real merchant names.

- Cases live **outside the repo** in `~/.finrag/search_eval.yaml` (or `FINRAG_SEARCH_EVAL`) — they name real merchants, so that file is never read or committed
- `evals/search_eval.example.yaml` shows the format with invented names:
  ```yaml
  - query: grocery shopping
    expected: ["Grocer Mart", "Fresh Fields"]
  - query: parking
    expected: ["Metro Parking Meters"]
  ```
- The 0.75 cutoff was only ever exercised with a cap of 5; at cap 50 it is unverified (`app/read_config.py:28`) — the search eval should be re-run at the new cap
- Add a `--summary` mode that prints only scores, so the eval can run without printing real merchant text (findings recommendation 3)

---

## Known limits

- **Eval questions are synthetic.** Real phrasing may score differently, and refused real questions are logged locally (`~/.finrag/refused_questions.log`) for review.
- **Form accuracy 37.8%** — misreads are common. The answer is still exact for what was understood, and the "Understood as" line shows the reading on every answer.
- **No import-direction enforcement tool** — the module-boundary rules (read path never imports `categorization.py`/`llm.py`, etc.) are enforced by review, not by a lint rule (00-overview section 4.3).
- **Synthetic fixtures encode guesses** — one real statement found two layout facts that hundreds of unit tests could not; run `check_profiles` on each new statement before trusting it (findings section 1).
