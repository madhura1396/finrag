# Plan: extraction rebuild

Implements [ADR 0002](../adr/0002-profile-based-extraction.md) with the contracts and data decisions in [ADR 0003](../adr/0003-extraction-contracts-and-data.md). There are three PRs, merged in order, each on its own branch, plus a data step the user runs between PR2 and PR3.

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
| `migrations/0001_add_transaction_section.sql` (new) | `ALTER TABLE transactions ADD COLUMN IF NOT EXISTS section TEXT NULL;` `IF NOT EXISTS` is there because on a fresh database `create_tables()` already creates the column. The undo line goes in a comment. It stays nullable until the data step. |
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

## PR2: synthetic fixtures, position-based reader, Wells Fargo profile, check script, wiring

Branch: `feat/profile-reader`

**Goal.** Read a Wells Fargo credit card PDF into `StatementData` using word positions and a YAML profile, with no Wells Fargo strings in the new modules. Wire the reader and validator into upload. Prove it on synthetic statements, and give the user a script to prove it on real ones without Claude seeing them.

**Files**

| File | Change |
|---|---|
| `profiles/wells_fargo.credit_card.v1.yaml` (new) | All bank-specific strings and layout facts (shape below). |
| `app/profile.py` (new) | Load the YAML into a typed `Profile` using `yaml.safe_load`. Reject unknown keys, missing keys and wrong types at load time (the lesson of #6 and #8, and the guard against YAML implicit typing). Section keys must come from the canonical set in `app/statement.py`: `payments`, `other_credits`, `purchases`, `cash_advances`, `fees`, `interest`. The pipeline's fee and interest handling depends on these keys across banks. |
| `app/reader.py` (new) | `read_statement(pdf_path, profile) -> StatementData`. Owns PDF text access and period parsing, which move here from `extractor.py` and `app/pipeline.py:13-14`. Rules below. |
| `app/pipeline.py` | `extract_from_pdf` loads the single profile, then calls `read_statement`, then `persist_statement`. The guard is live from here. The profile is found as the only `*.yaml` in `profiles/`; zero or several is a startup error. That keeps any bank name out of `app/` and adds no selection logic (ADR 0002, Decision 3). `enrich_transactions` skips rows in sections `fees`/`interest` for `call_llm_batch`, since their category was set at insert (ADR 0003, 2b). They are still embedded. |
| `app/query_router.py` | Superseded by the read-path decision (option C: the LLM fills a structured form, code builds the query and states every number; ADR 0004, to be written). That PR must merge before PR2. |
| (read-path eval) | Moved to the ADR 0004 PR. |
| `requirements.txt` | Add `pyyaml`. It is a direct dependency now, no longer only pulled in by `transformers`. Left unpinned to match the file; pinning the whole file is a separate decision. |
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

Run in order. Claude writes the SQL in PR2 as `migrations/0002_delete_legacy_statements.sql` and `migrations/0003_require_transaction_section.sql`. Only the user runs them.

1. Run `python scripts/check_profiles.py <your statements>` (add `--numbered` if you will paste the output) and get `PASS` on every file. If any line says `FAIL`, paste the output so the profile can be fixed. Do not continue until every file passes.
2. **Back up first.** `pg_dump -Fc finrag > finrag_before_reingest.dump`, kept outside the repo. This is the rollback for every step below: `pg_restore --clean -d finrag finrag_before_reingest.dump`.
3. Make sure Ollama is running with `llama3.2`. Enrichment runs after the statement commits (`app/pipeline.py:53`, `app/main.py:32`, open issue #3 item 4), so an Ollama failure mid-upload leaves an unenriched statement that a retry rejects with 409.
4. Run `0002`. It deletes from `transactions` first, then `statements`, in one transaction (`app/models.py:54` has a foreign key with no cascade). These are the legacy, unverified data (ADR 0003, decision 6).
5. Re-upload every statement through `/upload`. If any upload fails partway (a 500, or an Ollama error), stop and restore the dump (step 2), then retry once the cause is fixed. Do not try to patch individual statements.
6. Run `0003`: `ALTER TABLE transactions ALTER COLUMN section SET NOT NULL;`. It fails loudly if any row lacks a section.
7. Tell Claude the steps are done. Delete the dump when you are satisfied.

---

## PR3: delete the legacy extractor

Branch: `chore/remove-legacy-extractor`

**Precondition.** The user confirms the data step is complete.

**Goal.** Remove the LLM-described structure and the line state machine, so there is one extraction path, and make the model match the now-required column.

**Files**

| File | Change |
|---|---|
| `app/extractor.py` | Delete. Text access and period parsing already moved to `reader.py` in PR2. |
| `app/pipeline.py` | Remove any remaining `extractor` imports. |
| `app/models.py` | `Transaction.section` becomes `nullable=False`, matching migration `0003`. |
| `tests/test_extractor.py` | Delete, and remove it from `collect_ignore`. |
| `.claude/CLAUDE.md` | Layout: remove `extractor.py`. Also drop it from the "Ollama URL hardcoded in" line. |

**Tests.**
- The full `python -m pytest` suite stays green.
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
- #3 item 4: enrichment failure after the statement commits.
- #12: statement identity across cards. Deferred in ADR 0002. This is also the trigger to adopt Alembic (ADR 0003, decision 1).
- Rejected PDFs stay in `uploads/` (`app/main.py:23-25`).
- **Mixed signs before the data step.** Between PR2 merging and the re-ingest, legacy rows (unsigned, `section IS NULL`) sit next to signed rows. The ADR 0004 read path must refuse to answer while any `section IS NULL` row exists, or questions are paused until the data step is done.
- LLM categories are not validated against the category list (`app/pipeline.py:86`, issue #4 item 1). The model could label a purchase `Fees` or `Interest`.

## Still open

- **Fee row dates.** No fee rows have been seen yet. With `fees` in `undated_rows`, a fee row parses whether or not it prints a date, so nothing will flag the difference. Check the first statement that has a fee.
- **The Cash Advances section heading.** Unknown. A statement with a cash advance fails closed (`summary_section:cash_advances`, and rows landing in the wrong section give `section_total`) until the heading is added to the profile.
