# ADR 0005: User-owned categories with deterministic rules

- **Status:** Accepted
- **Date:** 2026-11-11
- **Builds on:**
  - [ADR 0003](0003-extraction-contracts-and-data.md): decision 2b, fees and interest get fixed categories and skip the LLM.
  - [ADR 0004](0004-code-owned-read-path.md): the read path computes exactly over stored labels.
- **Lands:** as the categories PR, after PR1 and **before** the read-path PR and PR2. See the [extraction rebuild plan](../plans/extraction-rebuild.md).
- **Amends ADR 0003:**
  - **Decisions 1 and 6, data.** The database holds only dummy data (user, 2026-11-11). Instead of incremental migrations and a delete-and-re-ingest of legacy rows, the categories PR ships a **reset**: drop all tables and recreate the schema from the models, with categories seeded.
    - Migrations planned as `0002` to `0004` are replaced by that reset.
    - `transactions.section` is `NOT NULL` from the reset onward.
    - Decision 1's "revisit Alembic at the next schema change (#12)" stands. Incremental migrations, and the Alembic question, matter again once real data exists. Until then a reset is cheaper and safer than a migration.
  - **The consequence "re-ingesting regenerates merchant names and categories"** no longer holds: memory, rules and pins (below) make labels stable across re-ingests.
  - **"Three SQL scripts"** becomes `0001` plus the reset script.
  - **"The read path must refuse to answer" until the data step** (ADR 0003 Consequences) is moot: after the reset there are no legacy rows.
- **Amends ADR 0004:**
  - **Category validation** uses the current versions of the categories tables, not a code list. `app/categories.py` becomes the seed only. The enrichment set is the current, non-archived, non-system categories (excluding `Fees`, `Interest` and `Other`).
  - **The row fetch** (ADR 0004 :270) **left**-joins `transactions.category_id` to the current category name through `app/category_store.py`, and also returns `category_source` and `embedded_text`. Pending rows (null `category_id`) are kept:
    - they count in every total that has no category filter,
    - they appear as **"Uncategorised (pending)"** in category breakdowns, rankings and `category_change`,
    - they are excluded from category-filtered totals and from `anomalies` history, but counted in the notes for that period.

    A pending row is never dropped from a total or silently counted as `Other`. The `/query` response is unchanged: it still shows names.
  - **The integration suite** (ADR 0004 :280) creates its schema with `create_tables(engine)` on the `finrag_test` engine it is given. Every setup function takes an explicit engine (decision on module boundaries). The suite's guard, marker and `CLAUDE.md` carve-out move to the categories PR, which is the first to need them.
  - **The notes** count `needs_review` and `pending` (not yet enriched) rows in the period for every answer, and stale-embedding rows for `search` (the only type that uses embeddings). Off-list categories are now impossible, and are no longer counted.
  - **The legacy-row refusal is removed:** after the reset no unsigned or section-less rows can exist.
  - **The "enrichment output" limit is narrowed:** labels are consistent per description and correctable, though model-chosen labels are not individually confirmed (decision 3c).
  - **Search's "drop the total" revisit** (ADR 0004 detail 5) is not triggered yet. Category totals now exist but are not proven better for "looks like" questions; revisit after real use.

## Context

ADR 0004 made every number on the read path deterministic *over the stored labels*. The labels themselves are not deterministic:

- **`app/llm.py`** sends batches of 15 descriptions (`:17`) to llama3.2, which returns a merchant name and a category for each.
  - The call sets no temperature (`:50-54`), so the model's default randomness applies.
  - The returned category is written without validation (`app/pipeline.py:176`; issue #4 item 1).
- **Identical descriptions get different labels.** One store appears under two merchant names and two categories (issue #2), so a category total or merchant ranking can be wrong while being computed exactly.
- **`needs_review`** is set only when the model skipped a row (`app/pipeline.py:178-180`).
- **There is no way to correct a label** except hand-written SQL.
- **The embedding text is merchant + description + category** (`app/embedder.py:17`), so any label change makes that row's embedding stale.

Agreed direction (ADR 0004, "Growth"):
- categories are the user's list,
- the model may only pick from it, and may only propose a new one,
- the user approves categories and rules that apply to all past and future rows.

## How this is done elsewhere

General knowledge of public behaviour, no figures quoted.

- **Large scale (banks, aggregators, budgeting apps):**
  - descriptors are resolved to a merchant entity (a curated merchant database plus fuzzy matching, plus card-network merchant category codes),
  - a trained classifier with confidence thresholds assigns the category, and low confidence goes to review,
  - the category hangs off the merchant entity, so the same merchant is always labelled the same,
  - per-user override rules live in database tables, with "apply to past transactions",
  - label and rule changes are logged for support, undo and audit.
- **Small scale:**
  - self-hosted apps (Firefly III, Actual Budget) keep an ordered rules engine in their own database,
  - plain-text accounting (beancount, hledger) keeps import rules in private files.
- **What applies here:**
  - PDF statements carry no merchant category codes, and there is no merchant database.
  - At this scale (one user; thousands of rows, unmeasured), rules plus a remembered decision per normalised description give the property that matters: **the same description always gets the same label.**

## Decisions

All examples below are invented descriptors, not statement text.

### 1. Where categories and rules live

**Database tables, with a change log.** Rules contain real merchant text, so they live only in the user's database, never in the repo, fixtures or docs.

| Table | Content |
|---|---|
| `category_ids` | `category_id` (primary key, the durable identity), `is_system`, `created_at`. The foreign-key target for every category reference. |
| `category_versions` | `category_id` (FK), `name`, `archived`, `valid_from`, `valid_to` (empty = current), `change_id`. SCD2 (decision 7). |
| `category_rules` | `rule_id`, `pattern`, `kind` (`contains` or `regex`), `priority`, `sets_category_id` (nullable FK), `sets_merchant` (nullable text), `deleted` (soft delete), timestamps. At least one of the two targets is set. |
| `label_memory` | `description_key` (**unique**), `category_id` (nullable FK), `category_source` (`model` or `user`), `merchant` (nullable), `merchant_source`, timestamps. Each field is remembered independently: a merchant is remembered even when the category was `none_fit`. |
| `transaction_label_history` | SCD2 per transaction (decision 5). |
| `change_log` | `change_id`, kind, payload, before and after snapshots, `undoes` (nullable), timestamp. Pins, rule and category edits, memory corrections, undo and merges are all change-log entries. |
| `review_items` | `review_id`, `transaction_id`, `kind` (`none_fit`, `conflict`, `pending`), `detail` (JSON: the model's suggestion, or the conflicting rule IDs and targets), `created_at`, `resolved_at`, `resolved_by_change_id`. The review endpoint reads this. It is not SCD2; resolution is recorded, never deleted. |
| `data_version` | A single row, created by the seed, bumped by every logged change, **every upload and every enrichment write** (decision 4). |

New columns on `transactions`:
- `category_id` (nullable FK),
- `category_source` (`section`, `pin`, `rule`, `memory`, `model`, `none_fit`, `pending`), `category_rule_id`,
- `merchant_source` (`pin`, `rule`, `memory`, `model`, `pending`), `merchant_rule_id`.

A field is pinned exactly when its source is `pin`, so there is no separate pinned flag that could disagree.

**Database-enforced invariants (constraints, not just tests):**
- one open version per category (a partial unique index on `category_versions(category_id) WHERE valid_to IS NULL`),
- one open history version per transaction (the same pattern on `transaction_label_history`),
- unique `label_memory.description_key`,
- unique current name among non-archived categories, case-insensitive (a partial unique index on `lower(name) WHERE valid_to IS NULL AND NOT archived`). Archiving opens a new current version with `archived = true`, so archived categories stay visible to the resolver and their names become free for reuse,
- `category_id IS NULL` exactly when `category_source = 'pending'` (a check constraint),
- the names `none_fit` and `pending` are reserved and cannot be used as category names.

Declined:
- a private YAML file (file and database drift),
- YAML in the repo (it would commit merchant text).

### 2. What a rule matches, and precedence

- **2a. Rules match the printed description** (`raw_description`), never the LLM-written merchant.
- **One normaliser** (`category_store.normalise`) is used by rules, memory keys and ADR 0004's merchant matching: case-folded, punctuation removed, spaces collapsed. Memory keys additionally drop digits (3b).
- **2b. A pattern is `contains` or a regular expression.**
  - A regex is compiled at creation, and an invalid pattern is refused.
  - Patterns with nested repetition (for example `(a+)+` or `(\w*)*`), the shape that causes catastrophic backtracking, are refused by a static check.
  - **Accepted risk:** Python's `re` has no timeout and the static check is not exhaustive, so a pathological pattern can still stall a request. If that happens inside an apply, it also holds the `data_version` lock and blocks uploads and other applies until the process is restarted. Short descriptions do not bound backtracking. The only pattern author is the local user, so this is an accepted, self-inflicted risk.
  - A pattern length cap (200 characters, an arbitrary starting value) bounds mistakes, not backtracking.
- **2c. When several rules match, the highest `priority` wins, decided separately for category and for merchant.**
  - A new or edited rule whose priority equals that of a rule matching some of the same existing rows with a different target for the same field is refused, showing the conflict.
  - **At ingest**, if two equal-priority rules still match a new row with different targets for a field, neither applies to that field. The row is flagged `needs_review`, a `conflict` review item records the rule IDs, and the field falls through to memory and then the model. Fail loudly; a tie never decides.
- **2d. Order of sources, per field, strongest first:**
  1. **Section** (Fees, Interest; category only). It cannot be overridden or pinned.
  2. **A pin on this row** for this field.
  3. **Rules**, by priority (2c).
  4. **Memory** for the row's description key.
  5. **The model** (decision 3).

  A rule never overrides a pin. Only the user removes a pin.

### 3. When no pin or rule applies

- **3a. The model is constrained.**
  - Temperature 0.
  - Output constrained (Ollama structured output) to one of the current enrichment categories or `none_fit`, plus a merchant name and an optional suggestion of up to 3 words (arbitrary). Validated by code.
  - Temperature 0 does not guarantee identical output across model or quantisation changes. Memory is what gives consistency.
- **3b. Memory, keyed by normalised description without digits.**
  - Two invented descriptors, `GROCER MART #101 SPRINGFIELD` and `GROCER MART #202 SPRINGFIELD`, share the key `GROCER MART SPRINGFIELD`. Once a key has a label, later rows reuse it without asking the model.
  - If removing digits leaves an empty key, the key falls back to the normalised text with digits kept.
  - Within one batch, rows with the same key are sent to the model once.
- **3c. Model labels are final** (no confirmation step).
  - **Only real categories are remembered.** `none_fit` and model failures are **never** written to memory.
  - A `none_fit` row gets `Other`, source `none_fit`, `needs_review`, and a `none_fit` review item holding the suggestion. The next row with that key is asked again, and is flagged again if still `none_fit`.
  - **Model unavailable or invalid output:** the row stays `pending` (no category, `needs_review`, a `pending` review item). Pins, rules and memory still resolve without the model; only fields nothing else decides stay pending. A row counts as pending if either field is.
  - Rows are inserted `pending` without a review item; one is opened only if enrichment leaves a field pending.
  - **Retrying `pending`:** a background enrichment pass runs after each upload, after each apply, and at app startup. `POST /enrichment/retry` runs it on demand. FastAPI background tasks are lost on restart, which is why startup and the endpoint exist.
  - The read path reports `pending` and `needs_review` counts (ADR 0004 notes). A missing label is never silently counted as `Other`.
  - **Accepted risk:** a wrong *remembered* label repeats until corrected. Memory makes labels consistent, not correct.
- **Pre-enrichment state:** non-section rows are inserted `pending`. Section rows get `Fees` or `Interest` at insert (ADR 0003, 2b).
- **Enrichment is serialised like any change.** Each enrichment write takes the `data_version` lock, re-checks that each row is still `pending` (an apply may have labelled it meanwhile), writes labels, history and memory, and bumps `data_version`. So an enrichment run invalidates open previews (409), and can never overwrite a label set by a rule or pin that landed first. The model call itself happens outside the lock.

### 4. How the user reviews and corrects

**API endpoints**, thin handlers over `app/categorization.py`.

- **Endpoints:**
  - reads: `GET /categories`, `GET /rules`, `GET /memory`, `GET /review` (`none_fit`, `pending` and conflict rows, grouped by key, with suggestions), `GET /changes`;
  - every change goes through one pair, `POST /changes/preview` and `POST /changes/apply`, with a `kind`:
    - categories: `create_category`, `rename_category`, `merge_category`, `delete_category`,
    - rules: `create_rule`, `edit_rule`, `delete_rule`,
    - memory: `set_memory`,
    - pins: `pin`, `unpin` (per field, one row; no token needed),
    - undo: `undo`.

    One pair rather than one route per resource keeps the preview, token, lock and log logic in one place;
  - `POST /embeddings/refresh` and `POST /enrichment/retry`.
- **Preview, then apply.** Any change that can relabel more than one row is two calls.
  - The preview returns the effect: rows that would change, how many fall through to memory or the model (decision 5), and example descriptions. It also returns a token binding **the exact change payload** (a hash) and the **`data_version`** at preview time.
  - The apply call must send the same payload and token. It takes a row lock on `data_version` (`SELECT … FOR UPDATE`), checks the version, applies, writes the log and bumps the version, all in one database transaction.
  - A changed version (a later change **or an upload**) or a different payload returns **409**, preview again.
- **Security.** Like every endpoint, these have no authentication and write to real data. **They must only be reachable from this machine.** Two layers:
  - run uvicorn bound to `127.0.0.1` (its default, and the documented command),
  - a middleware rejects any request whose client address is not loopback with 403. The app cannot see uvicorn's bind address, but it can see each request's source, so this works even if someone binds to `0.0.0.0`.
  - Binding to localhost does not stop requests from a browser page on the same machine. Accepted for a single-user local tool.
  - Any UI, container or remote access needs an authentication decision first.

Declined: a command-line tool, or both CLI and endpoints.

### 5. Applying changes, history and undo

- **5a. SCD Type 2 history of labels, with current values on `transactions`.**
  - `transaction_label_history` has one row per version: `transaction_id`, `category_id`, `category_source`, `category_rule_id`, `merchant`, `merchant_source`, `merchant_rule_id`, `valid_from`, `valid_to` and `change_id` (nullable for ingest).
  - It mirrors every label column on `transactions` exactly, so "current equals open history" is a strict equality, tested after every operation.
  - Applying a change closes affected open versions and opens new ones. History is **append-only**: closed versions are never reopened or edited.
  - **Point in time:** a row's labels on a date are the version valid on that date, and the category name is the category version valid on that date (decision 7).
- **Rule delete and pin removal.**
  - Rules are **soft-deleted** (`deleted`), so history's `rule_id` stays valid.
  - Rows that lose a rule or a pin fall through precedence: to memory if their key has an entry, otherwise to `pending`, then to the background enrichment batch. The model is never called on the request path.
  - The preview states how many rows go to memory and how many to `pending`.
- **Undo (option A: reverse order, as a compensating change).**
  - Undo never rewrites history. It is a **new** change (`undoes = #N`) that opens new versions restoring the labels, and restores rules, memory entries and category versions from #N's before-snapshot.
  - It is allowed only if **no later change touched any row, rule, memory entry or category that #N touched**. Otherwise it is refused, listing the later changes to undo first.
  - Rows ingested after #N are not "touched by #N". They follow whatever rules exist after the undo, through normal precedence. For example, undoing "create rule" soft-deletes the rule and those rows fall through as in a rule delete.
  - Undo is previewed and token-checked like any change. Pins are change-log entries, so a pin set after #N blocks undoing #N on that row, and undo can never overwrite a pin.
  - Declined: undoing only "what is still in place" (option B), which leaves partial states.
- **5b. Embeddings are refreshed in batches.**
  - A row is stale exactly when its stored `embedded_text` differs from what `build_embedded_text` would produce now.
  - After an applied change, a background task re-embeds the changed rows in batches. `POST /embeddings/refresh` recovers anything a lost task missed, because staleness is derived, not flagged.
  - `build_embedded_text(row, category_name)` takes the name as an argument, with no database lookup inside, so callers resolve names in bulk and there is no per-row query. A pending row is embedded with an empty category name; once labelled, it is stale and refreshed.
  - `build_embedded_text` and the stale predicate `is_stale(row, category_name)` move to a new `app/embedding_text.py` with no heavy imports, so both write and read paths (including the pure calculations) can use them. `app/embedder.py` keeps only the model.
  - Search answers state how many rows in their period are stale.

### 6. Merchant names

Merchants get the same treatment as categories:
- model-written names on first sight (temperature 0),
- memory per key,
- rules that set the merchant (alone or with a category),
- per-field pins,
- the same precedence, preview, history and undo.

For example, rules for the invented descriptors `MEGASTORE WHSE` and `MEGASTORE FUEL` can both set merchant `Megastore`, joining them for rankings and recurring detection. This fixes issue #2's name split.

### 7. Renaming, merging and deleting categories

**Categories are an SCD2 dimension with a durable ID:** `category_ids` (identity, the FK target) plus `category_versions` (names over time).

- **Rename:** close the current version and open a new one. No rule or memory row changes, and no transaction *label* changes; it is logged and undoable. Until the read-path PR drops it, the transitional `transactions.category` text copy (Consequences) is updated in the same transaction, which is the only row write a rename makes. In that window the table has no rows (see Consequences), so this is a correctness rule more than a cost. The rename makes the embedded text of that category's rows stale, and the batch refresh picks them up.
- **Merge A into B:**
  - rows, rules, memory entries and pins are re-pointed from A's ID to B's (a category pin keeps its meaning, "this row is in that category", so it follows the merge),
  - previewed and token-checked, with transaction history versions,
  - A's current version is closed and a new current version opens with `archived = true`,
  - an archived target is refused.
- **Delete:** with a chosen user category, a merge into it. Without one, the category's labels are withdrawn and its rows re-decided (Implementation decisions, D). Either way the category is archived; nothing is hard-deleted.
- **System categories** `Fees`, `Interest` and `Other` cannot be renamed, merged, deleted, used as rule or pin targets, or offered to the model. Pins on section rows are refused.
- Archived categories are never offered to the model, and cannot be targets of new rules or pins.

Declined:
- names as text with rename as a bulk update,
- IDs without name history.

## Implementation decisions (categories PR, 2026-11-11)

Settled with the user while building, after the final review of this ADR.

- **A. Write-path sessions: one process-wide factory, guarded by an in-database environment marker.** (Review M2; the user chose the factory, option B, with a guard.)
  - `app/sessions.py` holds the factory (`new_session()`); the app never swaps it. Endpoints use FastAPI's `Depends(get_session)`; tests override the factory. An autouse fixture resets it after every test, even one that crashes.
  - Every database has a one-row `db_environment` marker: `finrag` is `production`, `finrag_test` is `test`. Like Rails' stored environment, it travels with the data.
  - Every engine's `connect` event (`app/db_guard.py`) compares the marker with `FINRAG_ENV` (default `production`; `tests/conftest.py` sets `test`). A mismatch or a missing marker refuses the connection, so a test that forgets to swap the factory fails loudly instead of writing to real data.
  - Setup code (`db_setup`, `reset_db`) uses unguarded engines, because it is what writes the marker.
  - Integration tests run each test inside a transaction that is rolled back afterwards.
  - Declined: passing a session into every top-level function (option A). The user preferred the factory, given the guard.
- **B. `/upload` waits for labelling** (option A). The reply keeps `enriched` and `needs_review` and adds `pending`. Embeddings are refreshed in the background afterwards. Declined: replying before the model runs (option B), until uploads feel slow or a UI can show progress.
- **C. Smaller choices made in code:**
  - the source columns are `NOT NULL` with value checks, so the pending check constraint cannot be bypassed by a NULL source;
  - `merchant` stays `NOT NULL`: it holds the printed description until labelled, with `merchant_source = 'pending'`;
  - section rows have `merchant_source = 'section'`;
  - memory learned during an enrichment pass is applied only from the next pass, so rows sharing a key are all recorded as labelled by that one model call;
  - the loopback middleware allows `127.0.0.1` and `::1`; tests present a loopback client;
  - undoing a merge or rename after the old name was reused is refused with 422 (the name check), not a constraint error.

- **D. Deleting a category without a target re-decides its rows** (user decision, after the code review found that "merge into `Other`" would also re-point rules and memory to `Other`, so future rows would land there silently):
  - rules that only set the category are soft-deleted; rules that also set a merchant keep the merchant part,
  - memory forgets the category (an entry that held nothing else is removed),
  - pins on it are dropped,
  - the category is archived, so it is never offered again, and any row still carrying it falls to `pending`,
  - the background pass then labels those rows from the remaining categories: one batched pass per delete, about one model call per 15 distinct descriptions (unmeasured latency); rules and other memory apply first; `none_fit` goes to `Other` **with** a review item,
  - undo restores the name, the rules, the memory entries and the pins from the change's before-snapshot.

  Moving rows into `Other` explicitly is refused; `Other` is only the flagged `none_fit` fallback. Declined:
  - moving every row to `Other` with a review flag (B: deterministic, but every row must be reviewed by hand),
  - the literal merge into `Other` (C: silent `Other` for future rows).
- **E. Code-review fixes (round 1, categories PR):**
  - undo ignores changes already compensated by an undo, so reverse-order undo works to any depth;
  - a change's category dependencies (rule targets, pin and memory categories) count as touched, and restorers refuse archived targets, so undo cannot leave a live pointer to an archived category;
  - `resolve` never keeps a model, memory or `none_fit` label whose category is archived;
  - a merchant that loses its rule or pin returns to the printed description;
  - `pending` review items open only after the model has tried;
  - enrichment bumps `data_version` only when it changed something;
  - pins on section rows are refused for both fields;
  - memory keys passed to `set_memory` are normalised;
  - `/upload` is a plain `def` (a worker thread), so labelling never blocks other requests (issue #10), and a labelling failure after the save reports the rows as `pending` rather than a 500;
  - `reset_db` requires `--env` and counts trips, rules and memory as data;
  - the integration fixture reads the marker before dropping anything.

## Module boundaries

- **`app/category_store.py`** (read only):
  - the normaliser and the description key,
  - current category versions and names in bulk (the one place that resolves `valid_to IS NULL`),
  - point-in-time lookups.

  Imported by the read path, `embedder` callers, `llm.py` and `categorization.py`. It imports only `models`.
- **`app/categorization.py`** (writes): precedence, memory, rule validation, preview tokens, apply, history, change log, undo, rename, merge and delete. Imported by `pipeline.py` and `main.py`, never by the read path.
- **`app/embedder.py`** and **`app/embedding_text.py`** import neither. They receive names as arguments.
- **`app/categories.py`:** the seed list only: default names, plus the three system categories identified by a unique `system_key` (`fees`, `interest`, `other`) on `category_ids`, not by fixed numeric IDs, so seeding never fights the ID sequence.
- **`app/db_setup.py`** (new): `create_tables(engine)` and `seed(session)`, imported by the reset script, the integration suite and the documented setup command. It imports `models` and `categories`; `models` imports neither, so there is no import cycle and the dependency direction stays "everything → `models.py`". `app.models.create_tables` (`app/models.py:100`) moves here.
  - **Seed key:** the seed runs **only when `category_ids` is empty**. It inserts the system categories (by `system_key`), the default categories, and the single `data_version` row. On a seeded database it does nothing, so a renamed, merged or archived default is never resurrected.
  - **Explicit target:** every function takes the engine or session it acts on. Nothing in setup uses the import-time engine in `app/models.py:13`.

## Consequences

- **Labels are deterministic and consistent:** the same normalised description gets the same category and merchant until the user changes it. Every change is previewed, logged, versioned and undoable in reverse order. That fixes issues #2 and #4 item 1.
- **Hard constraint upheld:** a model failure or `none_fit` is never remembered, never silently counted as `Other`, and always flagged (`pending` or `needs_review`).
- **Model-chosen labels are final.** A wrong remembered label repeats until corrected.
- **Schema reset:** all tables are dropped and recreated with the new schema and seed. This is acceptable because the data is dummy (user, 2026-11-11). It is run by the user, or by Claude only when explicitly asked, per `.claude/CLAUDE.md` Data safety.
  - **The reset guards itself by the database's own environment marker** (Implementation decisions, A):
    - it requires `--yes` and `--confirm-name` equal to the target's database name (an argument, not a prompt, so it also works when Claude runs it on request);
    - it refuses a target marked `production`, or an unmarked one that holds statements, unless given `--allow-production`. Today's dummy `finrag` is unmarked and holds statements, so its one-time reset is `python -m scripts.reset_db --env production --yes --confirm-name finrag --allow-production`;
    - it never re-marks a database (a `test` database cannot be reset as `production`, or the reverse);
    - its target is `--database-url`, else `DATABASE_URL` loaded from `.env` by the script's own process. It never uses the import-time engine, and prints only database, host and port, never the password.

    After the data step, rules, memory, pins and the change log exist only in the database and cannot be re-uploaded. `finrag` is then marked `production`, so the script refuses by default.
- **Transitional column:** until the read-path PR, `transactions.category` (text, nullable, no default) stays as a derived copy of the current category name, NULL while `pending`, written in the same transaction as every label change and rename. The old `/query` code (`app/query_router.py:21-23, 330`), `app/eval_search.py:84` and `tests/unit/test_pipeline.py:86,90` keep compiling and running.
  - **In practice the window has no rows:** the reset empties the database, the legacy upload path fails on `section NOT NULL` (500), and `persist_statement` is not wired until PR2, which merges after the read-path PR. The sync is still built and tested so the invariant holds if that ordering changes.
  - The read-path PR drops the column.
- **New modules:** `app/category_store.py`, `app/categorization.py`, `app/db_setup.py` (with `app/categories.py` as the seed). New local, unauthenticated endpoints, behind a loopback-only middleware.
- **Contract change:** `GET /statements/{id}/transactions` returns `category: null` with `category_source: "pending"` for unenriched rows.
- **Enrichment changes** (`app/pipeline.py`, `app/llm.py`): section, then pin, then rule, then memory, then model, per field. `pending` until enriched. `none_fit` and failures are never remembered.
- **Embeddings** are refreshed in batches after label changes and renames. Search states how many rows are stale.
- **Rules and memory hold real merchant text,** in the user's database only.
- **Build order:** PR1, then the categories PR, then the read-path PR, then PR2, then upload real statements, then PR3.
