# Data and storage: tables, constraints, signed amounts, the environment marker

Status: drafted 2026-11-12. Verified against `app/models.py`, `app/db_setup.py`, `app/db_guard.py`, `scripts/reset_db.py` on `main` at `2fdd28a`. Where this guide and the code disagree, the code wins.

Related guides: [00-overview](00-overview.md) (whole system), [01-ingestion](01-ingestion.md) (rows written here), [02-categorization](02-categorization.md) (labels live in these tables), [03-read-path](03-read-path.md) (rows read from here), [05-api-security-ops](05-api-security-ops.md) (running locally, loopback-only).

Decision: [ADR 0003](../adr/0003-extraction-contracts-and-data.md) (contracts, signed amounts); [ADR 0005](../adr/0005-user-owned-categories.md) (category/merchant tables, SCD2, data_version).

---

## What this doc covers

- All 11 tables and what each holds
- Money: `Numeric(10, 2)`, `Decimal` only, never float
- Signed amounts: positive = money back to you, negative = money out
- The environment marker: `finrag` is `production`, `finrag_test` is `test`
- pgvector: 384-dim vectors, no index yet
- Creating/seeding/resetting: `db_setup`, `reset_db`, guards

---

## At a glance

| Table | Holds | Key constraints |
|---|---|---|
| `statements` | One uploaded PDF statement | unique(`period_start`, `period_end`) — same period can't be uploaded twice |
| `transactions` | Every transaction row, signed, with labels | `amount Numeric(10, 2)`; `category_id IS NULL` ⟺ `category_source = 'pending'` |
| `transaction_label_history` | SCD2 label history per transaction | one open version per transaction (`valid_to IS NULL`) |
| `category_ids` | Durable category identity | `system_key` unique; `fees`/`interest`/`other` are system |
| `category_versions` | SCD2 category names | one open version per category; current non-archived names unique case-insensitively |
| `category_rules` | User label rules over printed descriptions | `kind IN ('contains','regex')`; at least one of `sets_category_id` / `sets_merchant`; pattern 1–200 chars |
| `label_memory` | Remembered label per description key | `description_key` PK; `category_id` and `category_source` are both null or both set, and the same for `merchant` and `merchant_source` |
| `change_log` | Every logged change (preview→apply) | `kind`, `payload`, `before`, `after`, `touched`, optional `undoes` |
| `review_items` | Rows needing attention (`none_fit`, `pending`, `conflict`) | `kind IN (...)`; `resolved_at` set when resolved, never deleted |
| `data_version` | Single counter row | `id = 1`; bumped by every logged change, upload, enrichment write |
| `db_environment` | Single row marking the database's environment | `id = 1`; `environment IN ('production', 'test')` |

- **`embedding` column** on `transactions`: `Vector(384)` from all-MiniLM-L6-v2; 384 dimensions.
- **No pgvector index is defined** yet — search is a sequential scan (fine at thousands of rows; HNSW around 10⁵).
- **Setup:** `python -m scripts.reset_db --env production|test --yes --confirm-name <db>` drops everything, recreates, marks, seeds.
- **Every connection** checks the marker against `FINRAG_ENV` (default `production`); mismatch or missing marker refuses to connect.

---

## Tables

### `statements`
One uploaded PDF statement.
- `id`, `filename`, `period_start`, `period_end`, `uploaded_at`
- `uq_statement_period`: the same period can never be inserted twice (the database-level duplicate guard, `app/pipeline.py:90-97`)

### `transactions`
One row from a statement section (ADR 0003 decision 8: amounts are signed).
- `statement_id` → `statements.id`
- `trans_date`, `raw_description` (original PDF text), `amount`
- `is_credit` (`"true"`/`"false"`, derived from the sign — nothing on the read path reads it any more; ADR 0004 removed model-written SQL)
- `section` — one of `payments`, `other_credits`, `purchases`, `cash_advances`, `fees`, `interest` (`app.statement.SECTIONS`)
- Label columns: `category_id`, `category_source`, `category_rule_id`, `merchant`, `merchant_source`, `merchant_rule_id`
- `needs_review` — set when a label couldn't be decided (`none_fit`, rule conflict, model failure); details live in `review_items`
- `embedded_text` — merchant + raw_description + category (`app/embedding_text.py`)
- `embedding` — `Vector(384)`, nullable

Money rules: `amount` is `Numeric(10, 2)` (up to 10 digits total, 2 decimal places) — the right type for money, never `Float`. Parsed into `Decimal`, compared exactly with no tolerance for reconciliation. Money leaves the API as strings.

### `transaction_label_history`
SCD2 history mirroring `transactions`' label columns exactly, plus `valid_from`, `valid_to`, `change_id` (nullable for ingest). One open version per transaction (`valid_to IS NULL`). Append-only: closed versions are never reopened or edited.

### `category_ids`
Durable category identity — the FK target for every category reference.
- `category_id`, `is_system`, `system_key` (`fees`, `interest`, `other` for system; null for user categories), `created_at`
- Names live in `category_versions`, so a rename never touches references.

### `category_versions`
SCD2 names of a category.
- `category_id` → `category_ids`, `name`, `archived`, `valid_from`, `valid_to` (null = current), `change_id`
- `uq_category_open_version`: one open version per category
- `uq_category_current_name`: current, non-archived names unique ignoring case (archived names are free for reuse)

### `category_rules`
User rules over the printed description (`raw_description`), soft-deleted.
- `pattern`, `kind` (`contains` / `regex`), `priority`, `sets_category_id` (nullable FK), `sets_merchant` (nullable text), `deleted`, `created_at`, `updated_at`
- `ck_rule_has_target`: at least one of `sets_category_id` / `sets_merchant` set
- `ck_rule_pattern_length`: pattern 1–200 chars

### `label_memory`
Remembered label per normalised description key (ADR 0005, decision 3b).
- `description_key` (PK), `category_id` (nullable FK), `category_source` (`model`/`user`), `merchant`, `merchant_source`, `created_at`, `updated_at`
- Never holds a `none_fit` or failed model result.

### `change_log`
Every logged change (rules, categories, memory, pins, undo).
- `change_id`, `kind`, `payload`, `before`, `after`, `touched` (what the change touched — drives the reverse-order undo check), `undoes` (nullable FK to itself), `created_at`

### `review_items`
Rows needing the user's attention.
- `transaction_id` → `transactions`, `kind` (`none_fit`, `conflict`, `pending`), `detail` (JSON), `created_at`, `resolved_at`, `resolved_by_change_id`
- Resolution is recorded, never deleted.

### `data_version`
Single counter row (`id = 1`). Bumped by every logged change, every upload, every enrichment write that changed something. Preview tokens carry it; applies lock it with `SELECT … FOR UPDATE`.

### `db_environment`
Single row marking what this database is.
- `id = 1`, `environment IN ('production', 'test')`
- `finrag` is `production`; `finrag_test` is `test`. Like Rails keeping its environment inside the database.

---

## Money: `Decimal`, never float

| Layer | How |
|---|---|
| Storage | `transactions.amount` is `Numeric(10, 2)` (`app/models.py:228`) |
| Parsing | Reader parses money with a strict regex into `Decimal` (`_money`, `app/reader.py:57`) |
| Validation | `statement._is_money` (`app/statement.py:186`) accepts only a finite `Decimal` that fits `Numeric(10, 2)` exactly (`MAX_AMOUNT`, `MAX_DECIMAL_PLACES`, `:59-60`) |
| Read path | Every number computed in `app/calculations.py` over `Decimal` (`ZERO`, `CENT`, `ROUND_HALF_UP`) |
| API | Money leaves as strings (`str(Decimal)`) — `query_router._jsonable` (`app/query_router.py:76-85`), `main.list_transactions` (`app/main.py:157`) |

A value that would be rounded or refused at insert would mean the stored number differs from the validated one — so validation happens before any database work.

---

## Signed amounts (ADR 0003, decision 8)

`Row.amount` is signed from the cardholder's side:
- **Positive** — money back to you: payments, refunds, credits
- **Negative** — money out: charges, fees, interest

A single wrong sign moves a section's sum by twice its amount, so reconciliation always catches it. The sign is the only record of direction — there is no separate credit flag that could disagree (`Row.is_credit` was removed from the contract; `transactions.is_credit` survives as a derived column at insert).

Section totals are checked as **minus the summary sign times the printed total** (summary is written from the balance's side, rows from the cardholder's side).

---

## The environment marker (`app/db_guard.py`)

Each database records what it is in `db_environment`. Every new connection checks the marker against `FINRAG_ENV` (default `production`):
- Missing marker → refused ("set it up with `python -m scripts.reset_db`")
- Marker mismatch → refused ("database is marked `production` but `FINRAG_ENV` expects `test`")

Setup code (`db_setup`, `reset_db`) uses **unguarded** engines because it writes the marker. Tests swap the session factory (`app/models.set_session_factory`) — a test that forgets to swap hits a `production` marker and fails on connect instead of writing to real data.

---

## pgvector

- `embedding = Column(Vector(384))` on `transactions`
- Model: all-MiniLM-L6-v2, 384 dimensions, runs in-process (`app/embedder.py`)
- Vectors are never loaded for calculation — `read_queries` only checks `has_embedding` (`app/read_queries.py:26`)
- **No index defined yet** — search is a sequential scan (fine at thousands of rows; an HNSW index around 10⁵ rows, per 00-overview section 8)

---

## Create, seed, reset

### `app/db_setup.py` — `create_tables(engine, environment)`
- `CREATE EXTENSION IF NOT EXISTS vector` (pgvector must exist before `Vector(384)` columns)
- `Base.metadata.create_all` — only adds what's missing, safe on an existing database
- Writes (or refuses to overwrite) the `db_environment` marker
- Calls `seed(session)`: inserts `DataVersion(id=1, version=0)` if absent, then categories only when `category_ids` is empty

### Seed data (`app/db_setup.py`)
System categories (cannot be renamed, merged, deleted, targeted by rules/pins, or offered to the model):
- `fees` → "Fees", `interest` → "Interest", `other` → "Other"

User-owned defaults (belong to the user after seeding; never re-created):
- Groceries, Dining & DoorDash, Travel & Transport, Shopping, Telephone & Internet, Entertainment, Health & Fitness

### `scripts/reset_db.py` — full drop-and-recreate
```
python -m scripts.reset_db --env production|test --yes --confirm-name <db> [--allow-production] [--database-url URL]
```

Guards, all checked before anything is dropped:
1. `--yes` required
2. `--confirm-name` must equal the target database's name exactly
3. `--env` required (so nothing is marked `production` by default)
4. Target's environment marker: a database marked `production`, or an unmarked one holding user data, needs `--allow-production`
5. Target is explicit: `--database-url`, else `DATABASE_URL` from `.env` loaded by this process (only host and database name printed, never the password)

Uploaded PDFs in `uploads/` are never touched.

---

## Known limits and loose ends

- **No pgvector index** — sequential scan; fine at thousands, HNSW around 10⁵ rows (00-overview section 8)
- **`is_credit` is still a `String` column** (`"true"`/`"false"`), derived from the sign. Nothing on the read path reads it; a Boolean change would be a schema change needing an ADR (issue #4.8)
- **`reset_db` still lists `trips` in `DATA_TABLES`** (`scripts/reset_db.py:33`) — harmless (each table is checked with `to_regclass` first) but stale; trips were removed as out of scope
- **Schema reset vs. migrations** — the schema is rebuilt by `reset_db` (ADR 0005 amends ADR 0003 decision 1); Alembic is to be revisited at the next schema change with real data (#12)
