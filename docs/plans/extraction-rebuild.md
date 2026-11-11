# Plan: extraction rebuild

Implements [ADR 0002](../adr/0002-profile-based-extraction.md) with the contracts and data decisions in [ADR 0003](../adr/0003-extraction-contracts-and-data.md), the read path in [ADR 0004](../adr/0004-code-owned-read-path.md), and user-owned categories in [ADR 0005](../adr/0005-user-owned-categories.md). There are five PRs, merged in order, each on its own branch (PR1, the categories PR, the read-path PR, PR2, PR3), plus a data step the user runs between PR2 and PR3.

Schema changes:
- `migrations/0001` adds `section` (PR1, already applied).
- The categories PR then **resets** the database: all tables are dropped and recreated from the models, with categories seeded. The data is dummy (user, 2026-11-11; ADR 0005 amends ADR 0003).
- Incremental migrations resume once real data exists.

Rules for every PR:
- Run `senior-reviewer` on the diff, fix every blocker, then ask the user to confirm the commit.
- Tests use only synthetic statements. Never use real statements as fixtures, and never read `uploads/` or `.env`.
- Unit tests do not call Ollama or Postgres. HTTP calls are mocked.
- Stage files by name.
- Update the Layout section of `.claude/CLAUDE.md` in the same PR that adds or removes a module.
- SQL in `migrations/` is written by Claude and run only by the user.

Do not upload statements from the moment PR1 merges until PR2 merges (ADR 0003, decision 3).

---

## PR1: StatementData contract, validation, error mapping

Branch: `feat/statement-validation`

**Goal.** One bank-agnostic description of a parsed statement, one validator implementing ADR 0002 Decisions 4 and 5, and the 409/422 error mapping. The validator is built and tested here but **not wired into upload** (ADR 0003, decision 3). PR2 wires it.

**Preconditions**
- The user has run `migrations/0001_add_transaction_section.sql` **before** PR1 code runs. Every ORM select on `Transaction` fails until the column exists. That includes `app/main.py:67`, `app/pipeline.py:74`, `app/query_router.py:218`, `app/reembed.py:12` and `app/eval_search.py:45`, so reads break as well as uploads.
- Uploads are paused (ADR 0003, decision 3). After PR1, a legacy `ValueError` from `app/extractor.py:44,219,226` is no longer mapped to 409 and becomes a 500. That is accepted for the paused window, and PR2 removes the legacy call.
- Rollback: revert PR1, then the user runs the `DROP COLUMN` line in the script's comment.

**Files**

| File | Change |
|---|---|
| `migrations/0001_add_transaction_section.sql` (new) | `ALTER TABLE transactions ADD COLUMN IF NOT EXISTS section TEXT NULL;` `IF NOT EXISTS` is there because on a fresh database `create_tables()` already creates the column. The undo line goes in a comment. It is superseded by the categories PR's reset, which creates `section` as `NOT NULL`. |
| `app/statement.py` (new) | `StatementData`, `Row`, `SummaryLine`, `Mismatch` dataclasses, the `CHECKS` constant (every check name, below), `validate(data) -> list[Mismatch]`, and exceptions `StatementRejected` (carries mismatches) and `DuplicateStatement`. No PDF, profile, bank or database code, and no import of `app.models`. |
| `app/pipeline.py` | Raise `DuplicateStatement`, not `ValueError`, for an existing period. Add `persist_statement(data: StatementData, filename: str) -> dict`. It validates, then inserts the statement (with `filename`, because `app/models.py:32` is `nullable=False`) and its rows in one transaction, and returns `{"statement_id", "period_start", "period_end", "total_transactions"}`, the shape `app/main.py:35-38` already uses. Each row stores `section` and its signed `amount`. `is_credit` is derived from the sign (`'true'` if `amount > 0`, else `'false'`, so a 0.00 row is a charge) and kept as text, because `app/query_router.py:20,129` depends on it (ADR 0003, decision 8). Rows in sections `fees`/`interest` get their final merchant (printed description) and category (`Fees`/`Interest`) here, in the same insert, so an enrichment failure can never leave them as `Other`. Not called by upload yet. |
| `app/main.py` | Replace the blanket `except ValueError` (`app/main.py:29-30`): `DuplicateStatement` gives 409, `StatementRejected` gives 422. Anything else is a bug and stays a 500. |
| `app/models.py` | `Transaction.section = Column(String, nullable=True)`, matching the migration. |
| `requirements-dev.txt` (new) | `pytest`, pinned. |
| `pytest.ini` (new) | `testpaths = tests/unit`. |
| `tests/conftest.py` (new) | `collect_ignore = ["test_extractor.py", "test_llm.py"]`. These are manual scripts: `tests/test_extractor.py:22,38,50` take a `filepath` argument, and `tests/test_llm.py:23` calls live Ollama. Ignoring them protects `pytest tests/` as well as bare `pytest`. Also sets `os.environ["DATABASE_URL"]` to a dummy `postgresql://` URL before any `app` import. `app/models.py:13` calls `create_engine` at import time, and `load_dotenv()` does not override a variable that is already set, so tests never take credentials from `.env`. |
| `tests/unit/__init__.py`, `tests/unit/test_statement.py`, `tests/unit/test_pipeline.py`, `tests/unit/test_upload_api.py` (new) | Run with `python -m pytest` from the repo root. |
| `.claude/CLAUDE.md` | Tests section: dev install, `python -m pytest`, where unit tests live, that manual scripts remain. Layout: add `statement.py` and `migrations/`. Running locally: run the scripts in `migrations/` in order after `create_tables()`. |

**Contract (ADR 0003, 5a)**

```
Row:          trans_date: date | None, post_date: date | None, description: str,
              amount: Decimal (signed: + payment/refund/credit, - charge/fee/interest), section: str
SummaryLine:  value: Decimal, sign: int (+1 | -1 | 0 for new_balance), section: str | None
StatementData:
              period_start: date, period_end: date, rows: list[Row],
              section_totals: dict[str, Decimal] | None   # printed, keyed by section
              summary: dict[str, SummaryLine] | None      # must contain "new_balance"
              reader_mismatches: list[Mismatch]           # problems the reader found; default empty
Mismatch:     check: str, section: str | None, expected: str, actual: str
```

- `trans_date` is typed optional, and `amount` is unconstrained, so that the validator owns the checks for them (`row_invalid`). The reader never invents a date except under `undated_rows`.
- `reader_mismatches` is how the reader reports problems without raising, such as an unparseable cell or an unknown summary label. `validate()` returns them first, followed by its own. Problems that leave nothing to validate (`no_text_layer`, `period_not_found`) are raised by the reader directly as `StatementRejected`.
- `SummaryLine.section` carries the summary-to-section pairing, so the validator needs no profile knowledge.
- `post_date` is read but not stored.
- `Mismatch.expected` and `actual` are strings, and money is `str(Decimal)`, never a float. FastAPI cannot serialise `Decimal` inside `HTTPException.detail`: on fastapi 0.135.3 it raises `TypeError`, which would turn the 422 into a 500.

The 422 body (ADR 0003, 5b) is `{"detail": {"reason": "statement did not reconcile", "mismatches": [Mismatch, ...]}}`.

**Check names.** This is the one list. `CHECKS` in `app/statement.py` holds exactly these, and `check_profiles.py` may print nothing else.

| Check | Raised by | Meaning |
|---|---|---|
| `no_text_layer` | reader (raises) | The PDF has no extractable words, for example a scan. |
| `period_not_found` | reader (raises), validator | The period pattern was not found, or the period is missing, not a date or reversed. |
| `period_mismatch` | reader | Pages disagree on the period. |
| `header_not_found` | reader | The table header row was not found. |
| `summary_not_found` | reader | The summary start/end anchor or `right_column_anchor` is missing. Never falls back to full width. |
| `unknown_summary_label` | reader | A row inside the summary block matches no profile label. |
| `summary_incomplete` | reader, validator | A profile summary line or `new_balance` is missing; a summary value is unreadable or not a `SummaryLine`; a sign is not ±1 (0 for `new_balance`); a line names an unknown section; two lines name the same section; or a section with a printed total has no summary line. |
| `row_invalid` | reader, validator | An unparseable date or amount cell, an amount that is not a finite `Decimal` with at most 2 decimal places within `Numeric(10,2)`, an unknown section, or a null or non-`date` `trans_date`. Either sign is valid, and so is zero (ADR 0003, 4b and 8). |
| `no_rows` | validator | `rows` is empty. |
| `date_window` | validator | A `trans_date` is outside `[period_start - 7 days, period_end]`. |
| `no_printed_totals` | validator | `section_totals` or `summary` is `None`. This is ADR 0002 Decision 5's "cannot verify: no printed totals". |
| `section_total` | validator | Per section, the signed sum of `amount` does not equal the printed total times minus that section's summary sign (payments `+total`, purchases `−total`), compared as exact `Decimal`. Also: a printed total that is unreadable or for an unknown section. A section with rows but no printed total also fails, and so does a printed total with no rows unless it is zero. |
| `summary_equation` | validator | `sum(sign × value)` does not equal `summary["new_balance"].value`. |
| `unexpected_error` | `check_profiles.py` | Any exception the reader or validator did not turn into a mismatch. Printed instead of the exception, which is never shown. |
| `summary_section` | validator | A `SummaryLine` with a `section` does not equal `section_totals[section]`. A summary line whose section has no printed total must be zero (this covers Cash Advances). |

The validator reports every mismatch, not just the first.

**Tests**
- Validator: one test per check, plus a fully valid statement that passes.
  - Multiple mismatches are all reported.
  - Off by one cent fails.
  - 7 days before start passes, 8 days fails, and one day after end fails.
  - A zero amount and either sign pass. An amount with 3 decimal places, beyond `Numeric(10,2)`, a float or NaN fails.
  - A `None` date fails as a mismatch, not `TypeError`.
  - Nonzero Cash Advances with no section fails.
  - A summary without `new_balance` gives `summary_incomplete`, not `KeyError`.
  - `reader_mismatches` are returned first.
  - Every check name produced is in `CHECKS`.
- Pipeline (`SessionLocal` mocked):
  - on rejection, `persist_statement` calls no `add` or `commit`,
  - on success, it adds one statement with `filename` and N rows with `section`, a signed `amount` and an `is_credit` derived from the sign,
  - fee and interest rows get `Fees`/`Interest` in the same insert,
  - it returns the documented dict,
  - a duplicate raises `DuplicateStatement`.
- Endpoint (FastAPI `TestClient`, pipeline mocked). `UPLOAD_DIR` is monkeypatched to `tmp_path`, so no test ever writes into `uploads/` (`app/main.py:23-25` saves the file before the pipeline runs). A test asserts that `uploads/` is untouched.
  - `StatementRejected` gives a 422 whose JSON body matches the contract, with money as strings,
  - `DuplicateStatement` gives 409,
  - an unrelated `ValueError` is not reported as 409.

**Done when**
- `python -m pytest` and `python -m pytest tests/` are both green without Ollama, Postgres or `.env`.
- The 422 body serialises.
- Upload behaviour is otherwise unchanged.
- `CLAUDE.md` Tests and Layout are updated.
- `senior-reviewer` reports no blockers.

---

## Categories PR: user-owned categories with deterministic rules (ADR 0005)

Branch: `feat/user-owned-categories`. **Merges after PR1 and before the read-path PR.**

**Goal.**
- Categories and merchants are deterministic and consistent: per field, section, then pin, then rule, then memory, then model (temperature 0, constrained).
- Every change is previewed, token-checked, logged, versioned (SCD2) and undoable in reverse order.
- Categories are an SCD2 dimension with durable IDs.

**Database reset.** After merge, the user runs `python -m scripts.reset_db --env production --yes --confirm-name finrag --allow-production`, or asks Claude to run it. (`--allow-production` is needed once, because today's dummy `finrag` holds statements and has no marker yet.)
- It drops all tables and recreates them through `app/db_setup.py` on that explicit target, never the import-time engine: `section NOT NULL`, the ADR 0005 tables and constraints, then the seed.
- It prints the target before acting and refuses without `--yes`.
- It refuses a target marked `production`, or an unmarked one holding statements, unless given `--allow-production`. It requires `--yes` and `--confirm-name` equal to the database name, never re-marks a database, uses `DATABASE_URL` loaded in its own process (or `--database-url`), and prints only database, host and port. After this reset `finrag` is marked `production`, so a later reset refuses by default (ADR 0005, Implementation decisions A).
- `uploads/` is not touched.
- From then until PR2, the legacy upload path (which writes no `section`) fails loudly with a 500. Uploads are paused anyway (ADR 0003, decision 3), and the database stays empty.

**Files**

| File | Change |
|---|---|
| `scripts/reset_db.py` (new) | The marker-keyed guard above; calls `db_setup`. |
| `app/db_guard.py` (new) | The environment-marker check on every engine's `connect` event, comparing with `FINRAG_ENV`. |
| `app/sessions.py` (new) | The write-path session factory (`new_session`, `set_session_factory`, `reset_session_factory`) and FastAPI's `get_session`. |
| `app/db_setup.py` (new) | `create_tables(engine)`, moved from `app/models.py:100` (the pgvector extension, then `Base.metadata.create_all`), and `seed(session)`: when `category_ids` is empty, insert the system categories (fixed IDs), the defaults and the single `data_version` row; otherwise do nothing. Imports `models` and `categories`. |
| `app/categories.py` (new) | The seed data only: default names, and the system categories `Fees`, `Interest` and `Other` with fixed IDs. |
| `app/models.py` | Models for every ADR 0005 table: `category_ids`, `category_versions`, `category_rules`, `label_memory` (per-field sources), `transaction_label_history`, `change_log`, `review_items` and `data_version`. Also every ADR 0005 constraint (the partial unique indexes, including the non-archived-name index; the `pending` check constraint), and the new `transactions` columns (`category_id`, per-field source and rule ID; no pinned flags). `section` becomes `nullable=False`. **Transitional:** `transactions.category` (text) becomes nullable with no default, NULL while `pending`, and is dropped by the read-path PR. `create_tables` is removed from here. |
| `app/category_store.py` (new, read only) | `normalise()` and `description_key()` (with an empty-key fallback); current category versions and names in bulk, including archived (the one `valid_to IS NULL` resolver); point-in-time lookups. Imports only `models`. |
| `app/embedding_text.py` (new) | `build_embedded_text(row, category_name)` and `is_stale(row, category_name)`, moved from `app/embedder.py:13-17`. No heavy imports. |
| `app/categorization.py` (new, writes) | Rule validation (compile, nested-repetition check, length cap 200); per-field precedence; equal-priority conflicts (refused at creation; at ingest a `conflict` review item and a fall-through); memory per field (never a category for `none_fit` or failures; one model call per key per batch); `data_version` and preview tokens (payload hash plus version, `SELECT … FOR UPDATE`); apply (current columns, the transitional text, history, change log and version bump in one transaction); pins as change-log entries; soft-delete of rules; fall-through to memory or `pending`; reverse-order undo as a compensating change; rename, merge, delete and archive with system-category guards; `review_items` creation and resolution. |
| `app/llm.py` | Temperature 0. Output constrained to current enrichment categories (from `category_store`) or `none_fit`, plus a merchant name and an optional suggestion; validated. Invalid output leaves the row `pending`. |
| `app/pipeline.py` | `persist_statement` inserts section rows with `Fees`/`Interest` (source `section`) and other rows `pending` (with a `pending` review item), opens history versions, and bumps `data_version`. `enrich_transactions` calls the model outside the lock, then takes the `data_version` lock, re-checks each row is still `pending`, applies per-field precedence through `categorization`, writes memory, history and review items, and bumps the version. |
| `app/embedder.py` | Keeps only the model; text building moves to `embedding_text.py`. |
| `app/reembed.py` | Resolves names in bulk through `category_store`; uses `embedding_text`. |
| `app/main.py` | The ADR 0005 endpoints, plus `POST /embeddings/refresh` and `POST /enrichment/retry`. Background re-embed after applies; background `pending` enrichment after uploads, after applies and at startup. A **loopback-only middleware** (403 for non-loopback clients). `GET /statements/{id}/transactions` returns `category: null` and `category_source: "pending"` for unenriched rows. |
| `app/query_router.py`, `app/eval_search.py` | No change. They keep reading the transitional text column, which has no rows in this window. |
| `migrations/0001_add_transaction_section.sql` | Comment only: the `NOT NULL` step is now done by the reset. |
| `pytest.ini` | `testpaths = tests/unit tests/integration`; register the `integration` marker; `addopts = -m "not integration"`. Moved here from the read-path PR, since this is the first PR with integration tests. |
| `tests/integration/conftest.py` (new) | Reads `FINRAG_TEST_DATABASE_URL`; skips if unset; **refuses unless the database name is exactly `finrag_test`**; builds its own engine; `create_tables(engine)` and `seed`; drops everything after. Moved here from the read-path PR. |
| `tests/unit/test_pipeline.py` | Updated for `pending`, section categories by ID, review items and history versions. |
| `tests/unit/test_categorization.py`, `tests/unit/test_categories_api.py`, `tests/unit/test_db_setup.py` (new); `tests/integration/test_categorization_db.py` (new) | See Tests. |
| `.claude/CLAUDE.md` | Layout: `category_store.py`, `categorization.py`, `categories.py`, `db_setup.py`, `embedding_text.py`, `scripts/reset_db.py`, `tests/integration/`. Write path: `pipeline.py -> categorization.py, category_store.py, llm.py, embedding_text.py, embedder.py`. Running locally: setup becomes `python -c "from app.db_setup import ..."` (replacing `app.models.create_tables`), and the reset command with its guard. A rule: the API is loopback-only (unauthenticated write endpoints). Tests: the integration suite, how to run it, and the user-approved exception. Data safety: the carve-out letting the integration suite write to `finrag_test` only, never `finrag`. |

**Unit tests (no Ollama, no Postgres).** Invented descriptors only.
- **Normalisation:** `GROCER MART #101 SPRINGFIELD` and `GROCER MART #202 SPRINGFIELD` share a key; an all-digit description falls back to a non-empty key.
- **`db_setup`:** the seed on an empty database creates the system categories, the defaults and the `data_version` row; a second `seed` after a default was renamed or archived does **not** resurrect it.
- **`reset_db`:** refuses without `--yes`, with a wrong `--confirm-name`, on a `production` marker or an unmarked database holding statements (without `--allow-production`), and when re-marking; never touches the import-time engine (asserted by patching it to fail).
- **Rules:** contains and regex; an invalid regex and nested repetition refused; the length cap; the highest priority wins per field; an equal-priority conflict refused at creation and flagged at ingest (the field falls through, `needs_review`).
- **Precedence per field:** section beats pin beats rule beats memory beats model. A merchant-only pin does not freeze the category. Rules never change pinned fields or section rows.
- **The model** is mocked:
  - only enrichment categories or `none_fit` are accepted,
  - invalid output leaves the row `pending`,
  - `none_fit` gives `Other` with `needs_review` and is **not** remembered,
  - memory is written once per key and reused, and one key in a batch means one call,
  - a merchant is remembered even when the category was `none_fit`,
  - an apply that labels a row while its enrichment call is in flight wins: enrichment re-checks `pending` under the lock and skips it.
- **Preview and apply:** a token from an earlier `data_version` (after a change **or an upload**) gives 409; a different payload with the same token gives 409; apply writes the current columns, the derived text, the history and one change-log entry together.
- **Rule delete:** soft-deleted; rows fall to memory or `pending`; no model call in the request; the preview counts both.
- **Pins** are change-log entries; a pin set after change #N blocks undoing #N on that row.
- **Undo:**
  - a compensating change, so history stays append-only and point-in-time is correct across an undo,
  - refused when a later change touched the same rows, rules, memory or category, listing the blockers,
  - rows ingested after the change follow the current rules.
- **Rename:** no transaction labels change (only the transitional text copy); a new version opens; point-in-time shows the old name for old dates; the embedded text goes stale.
- **Merge:** rows, rules, memory and category pins are re-pointed; the source gets an open archived version and its name becomes reusable; an archived target is refused. Delete with `into` merges; delete without it withdraws the category (rules retired or reduced to their merchant part, memory forgotten, pins dropped) and its rows go `pending` for the model; an explicit move into `Other` is refused. System categories are refused for rename, merge, delete and as targets; pins on section rows are refused.
- **The stale-embedding check** (`embedding_text.is_stale`) detects a relabel, a rename and a pending row being labelled. `build_embedded_text` takes the name and does no database access.
- **Review items:** `none_fit` stores the suggestion; a conflict stores the rule IDs; a `pending` row gets one; resolution records the change ID.
- **The loopback middleware:** a request from a non-loopback client gets 403.
- **Endpoints:** shapes, preview-then-apply, and 409s.

**Integration tests (`finrag_test`)**
- `create_tables(engine, "test")` and the seed on an empty `finrag_test`; `reset_db --database-url <finrag_test> --env test --confirm-name finrag_test --yes` end to end, with the password never printed.
- The constraints: a second open version per category or per transaction, a duplicate memory key, a duplicate current non-archived name (case-insensitive) and a null `category_id` with a non-`pending` source are each rejected **by the database**. An archived name can be reused.
- Current columns equal open history after ingest, apply, undo, rename and merge.
- `SELECT … FOR UPDATE` serialises two concurrent applies (one gets 409), and an apply against a concurrent enrichment write.
- Point-in-time queries.

**Done when**
- Unit and integration suites are green.
- `scripts/reset_db.py` works on `finrag_test`.
- `senior-reviewer` reports no blockers.

---

## Read-path PR: code owns every number (ADR 0004)

Branch: `feat/code-owned-read-path`. **Merges before PR2**, which is the first change that stores signed rows.

**Goal.** The LLM only fills a validated form and arranges code-written facts. Thin SQL fetches rows, and pure Python computes every number. Code writes the exact answer; a checked placeholder rewrite makes it read naturally.

**Expected effect.** After the categories PR's reset, the database is empty until real statements are uploaded after PR2, so `/query` answers "no data" with coverage `none`. There are no legacy rows to refuse (ADR 0005 amends ADR 0004).

**Files**

| File | Change |
|---|---|
| (categories) | Already provided by the categories PR (ADR 0005): validation uses current category names from `app/category_store.py`. The read path never imports `categorization.py` or `llm.py`. |
| `app/read_settings.py` (new) | All thresholds and timeouts (ADR 0004 Consequences), the closed glue-word list for the rewrite check, and `FINRAG_REFUSED_LOG`, defaulting to `~/.finrag/refused_questions.log` and expanded at runtime. |
| `app/question_form.py` (new) | The JSON schema (12 types plus `unsupported`, fields from ADR 0004 detail 2) and the Ollama structured-output call (30 s timeout; unreachable or timed out gives 503). Validation and resolution, all pure functions: categories against current category names passed in from `category_store` (the function stays pure); merchant matching (normalise, contains with at least 3 characters, or at least 90% similar); trip matching (2c); periods resolved against an injected "today", including dates without a year (most recent occurrence started on or before today; ADR 0004 2a′); every relative token's fixed range; `ytd`; defaults per type; same-length "before" ranges. |
| `app/read_queries.py` (new) | The five queries of ADR 0004 (Testing; the legacy-row check is removed by ADR 0005), each taking an injected session: rows by date range only (with `section`, `needs_review`, `category_source`, has-embedding, `embedded_text` for the stale check, and the current category name from `category_store`); statement periods; distinct merchant names; trips; the pgvector search (date range and direction applied in SQL before the cap, direction passed in as parameters). |
| `app/calculations.py` (new, pure) | Everything numeric over `Decimal`, given rows and statement periods: day coverage (full, partial, none); direction, category and merchant filters (Definitions 1, with 0.00 rows excluded); the 12 types' results; refunds beside spending; per-day averages; month buckets; medians; the anomaly and recurring rules; counts of `needs_review` and `pending` rows for every answer, and stale-embedding (via `embedding_text.is_stale`) and no-embedding rows for search. Pending rows (null category) stay in every total without a category filter, and appear as "Uncategorised (pending)" in category breakdowns, rankings and `category_change` (ADR 0005 amends ADR 0004). It also exports the direction definitions as parameters for the search query. |
| `app/answers.py` (new) | Facts rendered as tokens; `answer_exact`; tables; the "Understood as" line; the placeholder-rewrite call (15 s timeout) and its checks (each token once, no digits outside tokens, no comparison, negation or quantity words outside tokens, length); fixed number and date formatting. |
| `app/query_router.py` | Orchestration only: form, then validate, then fetch, then calculate, then answer, then rewrite. It appends unsupported or not-understood questions to the refused-question log, ignoring write failures with a warning. `_classify`, `_generate_sql`, `_validate_sql`, `_schema_for` and `_format_answer` are removed. |
| `app/main.py` | The `/query` contract (ADR 0004, detail 6) and the 503. |
| `app/models.py`, `app/eval_search.py` | Drop the transitional `transactions.category` text column (ADR 0005); `eval_search.py` reads names through `category_store`. The schema change is applied by a reset (dummy data) or an `ALTER TABLE … DROP COLUMN` the user runs. |
| `demo.py` | Shows `understood_as` and `answer`; handles `refused` and 503. |
| `requirements.txt` | Add `pyyaml` (eval question files). PR2's row for it becomes a no-op. |
| `evals/read_path/questions.yaml`, `evals/read_path/teach.yaml` (new) | About 80 synthetic test questions with expected forms (including `unsupported`), and a separate few-shot set. Invented merchants and trips. |
| `evals/read_path/run.py` (new) | Manual eval with Ollama up. Reports accuracy overall, per type and per field, plus the rewrite-rejection rate. |
| `app/eval_search.py` | Use the search cutoff and cap from `read_settings.py` instead of its hardcoded threshold 1.0 and limit 10, and run only under `if __name__ == "__main__"` rather than on import. That makes it the cap-50 search eval. |
| `tests/unit/test_calculations.py`, `test_question_form.py`, `test_answers.py`, `test_query_api.py` (new) | Unit tests, below. |
| `tests/integration/test_read_queries.py` (new) | Integration suite, below, using the conftest from the categories PR. |
| `.claude/CLAUDE.md` | Layout: the new modules and `evals/`. Read path: `main.py -> query_router.py -> question_form.py, read_queries.py, calculations.py, answers.py, embedder.py, embedding_text.py, category_store.py, read_settings.py -> models.py`. Data safety: "never read the refused-question log". (The integration suite rule and carve-out were added by the categories PR.) |

**Unit tests (no Ollama, no Postgres; the LLM is mocked)**
- **`calculations`:**
  - each of the 12 types on synthetic signed rows: spending shown positive, rankings correct in both orders, 0.00 rows never counted, refunds stated beside spending, never netted;
  - a `pending` row (null category) counts in an unfiltered total, appears as "Uncategorised (pending)" in a category ranking, is excluded from a category-filtered total, and is counted in the notes;
  - direction definitions: `payments` against `refunds` against `received`;
  - day coverage full, partial and none; `compare` with unequal lengths deciding on per-day average; refusal when a comparison range is partial;
  - `monthly_trend` labelling partial months; history types below their minimums;
  - the anomaly and recurring thresholds at their edges.
- **`question_form`:**
  - an unknown category or `unsupported` is refused;
  - merchants: "costco" finds "Costco Wholesale"; "costcoo" matches by similarity; "at" does not match "Target"; punctuation is normalised; no match lists the closest names;
  - trips: exact, contains, ambiguous, intersection with a period, empty intersection, end before start;
  - periods: every relative token against a fixed today; `ytd` and its default; the same-length "before"; per-type defaults;
  - malformed model JSON, and out-of-range values (`limit` 0 or 51, an unknown `type`, a bad month), are refused as "not understood";
  - yearless dates on a fixed today: "December" on 11 Nov 2026 is Dec 2025, "May 5 to July 5" is 2026, a wrapping range rolls back a year;
  - a typed "grocer marts" against the stored invented merchant "GROCER MART S #101" does not match (no containment, under 90% similar) and lists the closest names. A known limit, pinned so a change is deliberate.
- **`answers`:**
  - the rewrite is rejected for a missing or duplicated token, a digit outside tokens, over-length text, and **every word on the closed list** (one parametrised case per word, including approximators, exclusions and direction verbs such as "received", "saved", "almost", "except");
  - a faithful arrangement is accepted;
  - tables are never sent for rewriting;
  - formatting: `$1,234.50`, half-up cents, one-decimal percentages, `5 Dec 2025`.
- **`/query`:**
  - every response field, with money and dates as strings;
  - refused is a 200; a model that is down is a 503; a rewrite timeout gives `answer_exact`;
  - the refused-question log written to a temp path, only for unsupported or not understood, and a write failure ignored. An autouse fixture points `FINRAG_REFUSED_LOG` at a temp path for **every** test, so no test can append to the real log.

**Integration tests (`pytest -m integration`, user-approved)**
- Uses the `tests/integration/conftest.py` from the categories PR (name guard, own engine, `create_tables(engine)`), adding synthetic rows and statements.
- The fetch query returns exactly the rows for date, category, merchant and direction filters (including boundary dates).
- The pgvector search respects the cutoff, cap and direction.
- Statement periods, distinct merchant names, trips and the no-embedding flag come back correctly.

**Done when**
- `python -m pytest` is green.
- `python -m pytest -m integration` is green against `finrag_test`.
- `evals/read_path/run.py` runs and its baseline scores are recorded in the PR (scores, not a gate).
- `demo.py` works against the new response, showing "no data" until real statements are uploaded.
- `senior-reviewer` reports no blockers.

---

## PR2: synthetic fixtures, position-based reader, Wells Fargo profile, check script, wiring

Branch: `feat/profile-reader`

**Goal.** Read a Wells Fargo credit card PDF into `StatementData` using word positions and a YAML profile, with no Wells Fargo strings in the new modules. Wire the reader and validator into upload. Prove it on synthetic statements, and give the user a script to prove it on real ones without Claude seeing them.

**Files**

| File | Change |
|---|---|
| `profiles/wells_fargo.credit_card.v1.yaml` (new) | All bank-specific strings and layout facts (shape below). |
| `app/profile.py` (new) | Load the YAML into a typed `Profile` using `yaml.safe_load`. Reject unknown keys, missing keys and wrong types at load time (the lesson of #6 and #8, and the guard against YAML implicit typing). Section keys must come from the canonical set in `app/statement.py`: `payments`, `other_credits`, `purchases`, `cash_advances`, `fees`, `interest`. The pipeline's fee and interest handling depends on these keys across banks. |
| `app/reader.py` (new) | `read_statement(pdf_path, profile) -> StatementData`. Owns PDF text access and period parsing, which move here from `extractor.py` and `app/pipeline.py:13-14`. Rules below. |
| `app/pipeline.py` | `extract_from_pdf` loads the single profile, then calls `read_statement`, then `persist_statement`. The guard is live from here. The profile is found as the only `*.yaml` in `profiles/`; zero or several is a startup error. That keeps any bank name out of `app/` and adds no selection logic (ADR 0002, Decision 3). Fee and interest rows are handled by the categories PR's section precedence (they never reach the model). |
| (categories) | No change: `Fees` and `Interest` are system categories seeded by the categories PR (ADR 0005). This PR's rows in those sections get their IDs via `persist_statement`. |
| `requirements.txt` | No change: `pyyaml` was added by the read-path PR. |
| `tests/fixtures/__init__.py`, `tests/fixtures/make_wells_fargo.py` (new) | Synthetic PDF generator, below. |
| `tests/unit/test_profile.py`, `tests/unit/test_reader.py`, `tests/unit/test_check_profiles.py` (new) | See Tests. |
| `scripts/check_profiles.py` (new) | Below. |
| `.claude/CLAUDE.md` | Layout: add `profile.py`, `reader.py`, `profiles/` (already listed) and `scripts/`. Write path: `pipeline.py -> reader.py, statement.py`. |

**Reader rules**
- **Words and rows.** Uses `page.get_text("words")`. Groups words into rows by y within `y_tolerance`.
- **Period.** Reads the period line on every page where it appears. All occurrences must agree, or the result is `period_mismatch`.
- **Table.**
  - Finds every occurrence of the header row, which handles the continuation page.
  - Column boundaries come from the header words' x positions.
  - Empty cells are allowed, such as the blank "Card Ending in" on payments.
  - Never uses page numbers.
- **Sections.**
  - Tracks sections by exact heading, and only inside the table (after a header row).
  - A section total is read only from a line matching that section's `total` pattern, which requires `FOR THIS PERIOD`, so the year-to-date box (`... IN <year>`) never matches.
  - A section **closes at its total line**. Lines after it are not rows until the next section heading or header row, so the year-to-date lines can never become undated rows.
- **Undated rows.** In sections listed under `undated_rows`, a row whose date cell is **empty** gets `period_end` (ADR 0003, 4a). A date cell that is non-empty but unparseable is `row_invalid`, in every section, so a misread date is never silently replaced. In other sections, an empty date is `row_invalid`.
- **Zero rows.** Printed 0.00 rows are kept (ADR 0003, 4b).
- **Summary.**
  - The block runs vertically from the row containing `summary.start_label` to the row containing `summary.end_label`.
  - Horizontally, it is read only left of the profile's `summary.right_column_anchor` words.
  - If any of the three is not found, the result is `summary_not_found`. It never falls back to full width or to the whole page.
  - Inside the block, a label may span two consecutive rows. A leading `+`, `-` or `=` word is stripped before matching, since the printed prefixes may or may not be separate words (resolved by `check_profiles.py` on real statements, not by the fixture).
  - A row matching no profile label is `unknown_summary_label`. A missing profile line is `summary_incomplete`.
  - Section headings outside the table, such as `Payments` in the summary, are never treated as sections.
- **Year resolution.**
  - Parse month and day as integers, never `strptime('%m/%d')`, which fails on 02/29 with no year.
  - Pick the year from `{period_start.year, period_end.year}` whose date falls in `[period_start - 7 days, period_end]`.
  - If neither fits, or the day is invalid, the result is `date_window`.
- **Cell errors.** An unparseable date or amount cell becomes a `Mismatch` (`row_invalid`). It never raises, and it never echoes the cell text.

**Fixture generator.** Uses PyMuPDF to write PDFs, so there is no new dependency. PDFs are generated into `tmp_path` at test time, so no binaries are committed. Each fixture has:
- the 7-column header,
- the 5 sections with their `TOTAL ... FOR THIS PERIOD` lines,
- a two-column Account Summary with a wrapped purchases label and a Cash Advances line,
- a year-to-date box with `TOTAL FEES CHARGED IN <year>` and `TOTAL INTEREST CHARGED IN <year>`,
- the period line on every page,
- the header repeated on a continuation page,
- an unnumbered extra page,
- undated interest rows, including a 0.00 variant,
- a Fees section with only a $0.00 total,
- payment rows with a blank card cell,
- invented merchants and amounts.

**Check script (`scripts/check_profiles.py`, ADR 0003, 7).** Takes PDF paths or a directory. For each file it runs `read_statement` + `validate` inside a catch-all and prints exactly one line:
- `PASS <name>`, or
- `FAIL <name>` followed by the failing check names from `CHECKS`, each optionally suffixed with `:<section key>`. For example: `FAIL stmt.pdf section_total:purchases summary_equation`.
- An unexpected exception prints `FAIL <name> unexpected_error`. The exception and its class are never printed. Only check names and section keys may follow the file name (`.claude/CLAUDE.md`, Data safety).

`<name>` is the file's basename by default. With `--numbered` it is `#1`, `#2`, …, for when you paste output to Claude and your filenames contain dates or account digits.

Nothing else is ever printed, on stdout or stderr: no amounts, dates, merchants, labels, mismatch values or tracebacks. MuPDF's own warnings are switched off (`fitz.TOOLS.mupdf_display_errors(False)`).
- It exits non-zero if any file fails.
- It imports only `app.profile`, `app.reader` and `app.statement`. Never `app.pipeline` or `app.models`, because `app/models.py:9,13` loads `.env` and builds an engine at import.
- No database, no Ollama, no writes.
- The user runs it; Claude never reads its input files.

**Profile shape (ADR 0003, 5d)**

```yaml
profile: wells_fargo.credit_card
version: 1
period:
  pattern: 'Statement Period (\d{2}/\d{2}/\d{4}) to (\d{2}/\d{2}/\d{4})'
  date_format: '%m/%d/%Y'
table:
  header: [Card Ending in, Trans Date, Post Date, Reference Number, Description, Credits, Charges]
  columns: {trans_date: Trans Date, post_date: Post Date, description: Description,
            credit: Credits, debit: Charges}
  y_tolerance: <points, tuned on fixtures>      # unverified
sections:
  - {key: payments,      heading: 'Payments',                                     total: '^TOTAL PAYMENTS FOR THIS PERIOD'}
  - {key: other_credits, heading: 'Other Credits',                                total: '^TOTAL OTHER CREDITS FOR THIS PERIOD'}
  - {key: purchases,     heading: 'Purchases, Balance Transfers & Other Charges', total: '^TOTAL PURCHASES, BALANCE TRANSFERS & OTHER CHARGES FOR THIS PERIOD'}
  - {key: fees,          heading: 'Fees Charged',                                 total: '^TOTAL FEES CHARGED FOR THIS PERIOD'}
  - {key: interest,      heading: 'Interest Charged',                             total: '^TOTAL INTEREST CHARGED FOR THIS PERIOD'}
undated_rows: [interest, fees]        # fees unverified: no fee rows seen yet
summary:
  start_label: 'Previous Balance'
  end_label: 'New Balance'
  right_column_anchor: 'Total Credit Limit'
  lines:
    previous_balance: {label: 'Previous Balance', sign: +1}
    payments:         {label: 'Payments', sign: -1, section: payments}
    other_credits:    {label: 'Other Credits', sign: -1, section: other_credits}
    cash_advances:    {label: 'Cash Advances', sign: +1, section: cash_advances}   # no known heading: must be 0
    purchases:        {label: 'Purchases, Balance Transfers & Other Charges', sign: +1, section: purchases}
    fees:             {label: 'Fees Charged', sign: +1, section: fees}
    interest:         {label: 'Interest Charged', sign: +1, section: interest}
    new_balance:      {label: 'New Balance', sign: 0}
```

- The strings come from the user's statement (ADR 0003 Context).
- The summary equation is expressed as signs, so there is no expression parser.
- `y_tolerance` and the column x bounds are tuned on synthetic fixtures, which encode a guess about the real layout. Only `check_profiles.py` on real statements is evidence.

**Tests**
- A valid synthetic statement reconciles. Every row's date, description, amount, side and section match the generator's ground truth.
- Each layout feature has its own test:
  - a continuation page with a repeated header, with no rows duplicated or dropped,
  - a multi-line description stays one row,
  - an amount in the Credits column gives a positive `amount`, and one in Charges a negative `amount`,
  - a blank card cell is read,
  - undated interest rows get `period_end`,
  - 0.00 interest rows are stored,
  - a Fees section with a $0.00 total and no rows passes,
  - the year-to-date box neither sets the fees or interest total nor creates rows,
  - an unreadable date cell in the interest section is `row_invalid`, not `period_end`,
  - a missing summary anchor gives `summary_not_found`,
  - the wrapped purchases summary label is read,
  - right-column summary values are not mixed into the left column,
  - a December-to-January statement assigns the correct years, and a 02/29 row in a leap year parses.
- Broken variants, each rejected with the right check:
  - a dropped row,
  - a total off by 1 cent,
  - a missing total line,
  - a missing summary,
  - a summary equation off by 1 cent,
  - a date 8 days before the period,
  - zero rows,
  - an unparseable amount cell,
  - an unknown summary label,
  - a period that differs on page 2,
  - nonzero Cash Advances.
- Profile loader: rejects an unknown key, a missing section, and a string where a list is expected. An unquoted `no` in a label is a type error, not `False`.
- Pipeline (Ollama and DB mocked):
  - fee and interest rows never reach `call_llm_batch` and get `Fees`/`Interest`,
  - a reconciling synthetic upload stores rows with sections,
  - a broken one returns 422 and stores nothing.
- `git grep` test: `statement.py`, `profile.py` and `reader.py` contain none of the profile's heading strings. `extractor.py` still does until PR3.
- `check_profiles.py` on fixtures:
  - every token after `<name>` is in `CHECKS`, optionally with `:<canonical section key>`, and nothing else,
  - a fixture that makes the reader raise prints `FAIL <name> unexpected_error`,
  - `--numbered` prints no filename,
  - a broken fixture's output contains none of its amounts, dates or merchants,
  - stderr is empty,
  - the script does not import `app.models`,
  - the exit code is correct.

**Done when**
- `python -m pytest` is green.
- Upload of a synthetic statement through the pipeline stores reconciled rows with sections.
- A broken one returns 422 and stores nothing.
- `scripts/check_profiles.py` runs on fixtures.
- `senior-reviewer` reports no blockers.
- The user has the command to run on their real statements.

---

## Data step (user, between PR2 and PR3)

The database was reset by the categories PR, so there is nothing to delete or convert.

1. Run `python scripts/check_profiles.py <your statements>` (add `--numbered` if you will paste the output) and get `PASS` on every file. If any line says `FAIL`, paste the output so the profile can be fixed. Do not continue until every file passes.
2. Optional: if anything you care about has been entered since the reset, back it up first: `pg_dump -Fc finrag > finrag_backup.dump`, kept outside the repo.
3. Make sure Ollama is running with `llama3.2`. Enrichment failures leave rows `pending`, retried in the background (ADR 0005), so they do not block uploads.
4. Upload every statement through `/upload`.
5. Review `none_fit`, `pending` and conflict rows through the review endpoint, and add rules for anything mislabelled.
6. Tell Claude the steps are done.

---

## PR3: delete the legacy extractor

Branch: `chore/remove-legacy-extractor`

**Precondition.** The user confirms the data step is complete (all real statements PASS and are uploaded).

**Goal.** Remove the LLM-described structure and the line state machine, so there is one extraction path.

**Files**

| File | Change |
|---|---|
| `app/extractor.py` | Delete. Text access and period parsing already moved to `reader.py` in PR2. |
| `app/pipeline.py` | Remove any remaining `extractor` imports. |
| `tests/test_extractor.py` | Delete, and remove it from `collect_ignore`. |
| `.claude/CLAUDE.md` | Layout: remove `extractor.py`. Also drop it from the "Ollama URL hardcoded in" line. |

**Tests.**
- The full `python -m pytest` suite and `python -m pytest -m integration` (against `finrag_test`) stay green.
- `git grep -n extractor -- app tests` returns nothing. (`git grep` skips stale `__pycache__` files.)
- The heading-strings `git grep` test now covers all of `app/`.

**Done when**
- `extractor.py` is gone and nothing imports it.
- Issues #6, #7 and #8 can be closed, because the code they describe no longer exists.
- `CLAUDE.md` Layout matches the tree.
- `senior-reviewer` reports no blockers.

---

## Not in these PRs

- #10: `/upload` is `async def` but does blocking work, so it blocks the event loop. ADR 0002 keeps upload synchronous, but the fix (a plain `def`, so FastAPI runs it in a threadpool) is a separate small PR.
- #3 item 4 (enrichment failure after the statement commits) is addressed by ADR 0005: failures leave rows `pending`, counted and retried, never silently labelled.
- #12: statement identity across cards. Deferred in ADR 0002. This is also the trigger to adopt Alembic (ADR 0003, decision 1).
- Rejected PDFs stay in `uploads/` (`app/main.py:23-25`).

## Still open

- **Fee row dates.** No fee rows have been seen yet. With `fees` in `undated_rows`, a fee row parses whether or not it prints a date, so nothing will flag the difference. Check the first statement that has a fee.
- **The Cash Advances section heading.** Unknown. A statement with a cash advance fails closed (`summary_section:cash_advances`, and rows landing in the wrong section give `section_total`) until the heading is added to the profile.
