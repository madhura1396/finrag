# Plan: extraction rebuild

Implements [ADR 0002](../adr/0002-profile-based-extraction.md). There are three PRs, merged in order, each on its own branch.

Rules for every PR:
- Run `senior-reviewer` on the diff before committing and fix every blocker.
- Tests use only synthetic statements. Never use real statements as fixtures, and never read `uploads/` or `.env`.
- Unit tests do not call Ollama or Postgres. HTTP calls are mocked.
- Stage files by name.
- Update the Layout section of `.claude/CLAUDE.md` in the same PR that adds or removes a module.

Items marked **[confirm]** are contracts or schema changes proposed here. Each one is presented with options before its code is written, per the decision protocol in `CLAUDE.md`. Once chosen, each is recorded as an ADR in `docs/adr/` before its code lands.

---

## PR1: StatementData contract, validation, guard before commit

Branch: `feat/statement-validation`

**Goal.** One bank-agnostic description of a parsed statement, and one validator that implements ADR 0002 Decisions 4 and 5. The pipeline validates before opening a write transaction, so a rejected statement leaves no database rows and can be uploaded again.

**Preconditions**
- Waiting-on-you items 1 (migration), 3 (wiring) and 5 (contracts) are decided.
- The `section` migration has been run by the user **before** PR1 code runs. Every ORM select on `Transaction` fails until the column exists. That includes `app/main.py:67`, `app/pipeline.py:74`, `app/query_router.py:218` and `app/reembed.py:12`, so reads break as well as uploads.
- The rollback is the matching `ALTER TABLE transactions DROP COLUMN section`, run by the user, together with reverting PR1.

**Files**

| File | Change |
|---|---|
| `app/statement.py` (new) | `StatementData`, `Row`, `SummaryLine` dataclasses. `validate(data) -> list[Mismatch]`. Exceptions `StatementRejected` (carries mismatches) and `DuplicateStatement`. No PDF, profile or bank code. |
| `app/pipeline.py` | Run `validate()` before any write, and raise `StatementRejected` on any mismatch. Raise `DuplicateStatement`, not `ValueError`, for an existing period. Persist `section`. Keep storing `is_credit` as the strings `'true'`/`'false'`, converted from `Row.is_credit: bool`, because `app/query_router.py:20,129` depends on the string form. |
| `app/main.py` | Replace the blanket `except ValueError` (`app/main.py:29-30`): `DuplicateStatement` gives 409, `StatementRejected` gives 422. Any other exception is a bug and stays a 500. Parse errors never reach `main.py` as `ValueError`: the reader converts them to mismatches (PR2). |
| `app/models.py` | `Transaction.section`, nullable, because rows ingested before this change have none. **[confirm]** schema change. |
| `requirements-dev.txt` (new) | `pytest`, pinned. |
| `pytest.ini` (new) | `testpaths = tests/unit`. |
| `tests/conftest.py` (new) | `collect_ignore = ["test_extractor.py", "test_llm.py"]`. These are manual scripts: `tests/test_extractor.py:22,38,50` take a `filepath` argument, and `tests/test_llm.py:23` calls live Ollama. Ignoring them here protects `pytest tests/` as well as bare `pytest`. Also sets `os.environ["DATABASE_URL"]` to a dummy `postgresql://` URL before any `app` import. `app/models.py:13` calls `create_engine` at import time, which fails on `None`, and `load_dotenv()` does not override a variable that is already set, so tests never take credentials from `.env`. |
| `tests/unit/__init__.py`, `tests/unit/test_statement.py`, `tests/unit/test_upload_api.py` (new) | Validator, pipeline and endpoint tests. Run as `python -m pytest` from the repo root so `import app` resolves. |
| `.claude/CLAUDE.md` | Tests section: dev install, `python -m pytest`, where unit tests live, and that the manual scripts remain. Layout section: add `statement.py`. |

**Proposed contract [confirm]**

```
Row:          trans_date: date, post_date: date | None, description: str,
              amount: Decimal, is_credit: bool, section: str
SummaryLine:  value: Decimal, sign: int (+1 | -1 | 0 for new_balance), section: str | None
StatementData:
              period_start: date, period_end: date, rows: list[Row],
              section_totals: dict[str, Decimal] | None   # printed, keyed by section
              summary: dict[str, SummaryLine] | None      # must contain "new_balance"
Mismatch:     check: str, section: str | None, expected: str, actual: str
```

- `SummaryLine.section` carries the summary-to-section pairing. The reader fills it from the profile, so the validator needs no profile knowledge.
- `post_date` is read but not stored. There is no column for it. That is fine for now.
- `Mismatch.expected` and `actual` are strings on the wire. Money is formatted with `str(Decimal)` and is never a float. FastAPI cannot serialise `Decimal` inside `HTTPException.detail` (confirmed on fastapi 0.135.3: `TypeError`, which would turn the 422 into a 500).

The 422 body is `{"detail": {"reason": "statement did not reconcile", "mismatches": [Mismatch, ...]}}`. **[confirm]** This is an API response shape.

**Validator checks, in order. All are reported, not just the first.**
1. `rows` is non-empty.
2. Each row has `amount > 0` and a non-null `trans_date`. Zero-amount rows depend on the real statement; see "Waiting on you" 4.
3. Every `trans_date` is in `[period_start - 7 days, period_end]`.
4. If `section_totals` or `summary` is `None`, reject with `cannot verify: no printed totals`.
5. For each section, the sum of `amount` equals the printed total, as exact `Decimal` (no rounding, no tolerance). A section that has rows but no printed total also fails, and so does a printed total with no rows unless it is zero.
6. `sum(sign × value) == summary["new_balance"].value`, and every `SummaryLine` with a `section` equals `section_totals[section]`. A summary line the profile does not know about fails closed in the reader (PR2).

**Tests**
- Validator: one test per check, plus a fully valid statement that passes.
  - Multiple mismatches are all reported.
  - Off by one cent fails.
  - 7 days before start passes, 8 days fails, and one day after end fails.
  - A `None` date and a zero amount fail as mismatches, not `TypeError`.
- Pipeline (`SessionLocal` mocked): on rejection no `add` or `commit` is called. A duplicate raises `DuplicateStatement`.
- Endpoint (FastAPI `TestClient`, pipeline mocked):
  - `StatementRejected` gives a 422 whose JSON body matches the contract, with money as strings.
  - `DuplicateStatement` gives 409.
  - An unrelated `ValueError` is not reported as 409.

**Done when**
- `python -m pytest` and `python -m pytest tests/` are both green without Ollama, Postgres or `.env`.
- A rejected upload leaves zero database rows and can be re-uploaded.
- The 422 body serialises.
- `CLAUDE.md` Tests and Layout are updated.
- `senior-reviewer` reports no blockers.

**Wiring (decided by "Waiting on you" 3).** Either:
- **(a) Wire now.** PR1 includes a legacy adapter from `parse_transactions_from_text` dicts to `StatementData`, with `section_totals=None`. Every upload is then rejected with `cannot verify: no printed totals` until PR2.
- **(b) Wire in PR2.** PR1 lands the validator, the exceptions and the 422 mapping, but `extract_from_pdf` keeps the legacy path. The guard goes live in PR2, and uploads stay on the unguarded path for one PR.

---

## PR2: synthetic fixtures, position-based reader, Wells Fargo profile, check script

Branch: `feat/profile-reader`

**Goal.** Read a Wells Fargo credit card PDF into `StatementData` using word positions and a YAML profile, with no Wells Fargo strings in the new modules. Prove it on synthetic statements, and give the user a script to prove it on real ones without Claude seeing them.

**Merge preconditions**
- "Waiting on you" 2 (spending semantics) is decided and implemented in this PR or merged before it. PR2 is the first change that stores fee and interest rows. Both sit in the Charges column, so without a read-path change they enter every "how much did I spend" answer. `app/query_router.py:11-25,129-131` forces `is_credit = 'false'` and does not show `section` to the model.
- "Waiting on you" 4 (exact strings and real-statement facts) is answered.

**Files**

| File | Change |
|---|---|
| `profiles/wells_fargo.credit_card.v1.yaml` (new) | All bank-specific strings and layout facts. |
| `app/profile.py` (new) | Load the YAML into a typed `Profile`. Reject unknown keys, missing keys and wrong types at load time (the lesson of #6 and #8). |
| `app/reader.py` (new) | `read_statement(pdf_path, profile) -> StatementData`. Owns PDF text access and period parsing, which move here from `extractor.py` / `app/pipeline.py:13-14`. Details below. |
| `app/pipeline.py` | Replace the `extract_raw_text`, `parse_statement_period`, `detect_structure` and state machine calls with `read_statement` and the single profile path. |
| `app/query_router.py` | Only if "Waiting on you" 2 is decided as part of this PR: add `section` to the schema text and the spending rule. |
| `requirements.txt` | Add `pyyaml`. It is a direct dependency now, no longer only pulled in by `transformers`. Left unpinned to match the file; pinning the whole file is a separate decision. |
| `tests/fixtures/__init__.py`, `tests/fixtures/make_wells_fargo.py` (new) | Synthetic PDF generator, below. |
| `tests/unit/test_profile.py`, `tests/unit/test_reader.py`, `tests/unit/test_check_profiles.py` (new) | See Tests. |
| `scripts/check_profiles.py` (new) | Below. |
| `.claude/CLAUDE.md` | Layout: add `profile.py`, `reader.py` and `scripts/`. Write path: `pipeline.py -> reader.py`. |

**Reader rules**
- Uses `page.get_text("words")`.
- Groups words into rows by y within `y_tolerance`.
- Finds every occurrence of the header row, which handles the continuation page.
- Column boundaries come from the header words' x positions.
- Tracks sections by heading. Reads section totals from total lines, and the summary from labelled values.
- Never uses page numbers.
- **Year resolution:** parse month and day as integers, never `strptime('%m/%d')`, which fails on 02/29 with no year. Then pick the year from `{period_start.year, period_end.year}` whose date falls in `[period_start - 7 days, period_end]`. If neither fits, or the day is invalid, it is a mismatch.
- **Cell errors:** an unparseable date or amount cell becomes a `Mismatch` naming the check and section. It never raises, and it never echoes the cell text.
- An unknown summary label fails closed.

**Fixture generator.** Uses PyMuPDF to write PDFs, so there is no new dependency. PDFs are generated into `tmp_path` at test time, so no binaries are committed. Each fixture has:
- the same 7-column header,
- the 5 sections with `TOTAL ... FOR THIS PERIOD` lines,
- an Account Summary,
- the header repeated on a continuation page,
- an unnumbered extra page, so PDF and printed page numbers differ,
- invented merchants and amounts.

**Check script (`scripts/check_profiles.py`).** Takes PDF paths or a directory. For each file it runs `read_statement` + `validate` inside a catch-all and prints exactly one line, `PASS <basename>` or `FAIL <basename>`.
- On failure, nothing from the document is printed: no amounts, merchants, dates, mismatch values or tracebacks.
- It exits non-zero if any file fails.
- No database, no Ollama, no writes.
- The user runs it; Claude never reads its input files.

**Proposed profile shape [confirm]** (a file format)

```yaml
profile: wells_fargo.credit_card
version: 1
period:
  pattern: '<regex with two date groups>'
  date_format: '%m/%d/%Y'
table:
  header: [Card Ending in, Trans Date, Post Date, Reference Number, Description, Credits, Charges]
  columns: {trans_date: Trans Date, post_date: Post Date, description: Description,
            credit: Credits, debit: Charges}
  y_tolerance: <points, tuned on fixtures>      # unverified
sections:
  - {key: payments,      heading: Payments}
  - {key: other_credits, heading: Other Credits}
  - {key: purchases,     heading: <exact heading>}
  - {key: fees,          heading: Fees Charged}
  - {key: interest,      heading: Interest Charged}
total_line: '<regex for TOTAL ... FOR THIS PERIOD>'
summary:
  previous_balance: {label: <exact label>, sign: +1}
  payments:         {label: <exact label>, sign: -1, section: payments}
  other_credits:    {label: <exact label>, sign: -1, section: other_credits}
  purchases:        {label: <exact label>, sign: +1, section: purchases}
  fees:             {label: <exact label>, sign: +1, section: fees}
  interest:         {label: <exact label>, sign: +1, section: interest}
  new_balance:      {label: <exact label>}
```

The summary equation is expressed as signs, so there is no expression parser.

**Tests**
- A valid synthetic statement reconciles. Every row's date, description, amount, side (credit or debit) and section match the generator's ground truth.
- A continuation page with a repeated header is read without duplicating or dropping rows.
- A multi-line description stays one row.
- An amount in the Credits column gives `is_credit=True`, and one in Charges gives `False`.
- A December-to-January statement assigns the correct years. A 02/29 row in a leap-year statement parses.
- Broken variants, each rejected with the right check:
  - a dropped row,
  - a printed total altered by 1 cent,
  - a missing total line,
  - a missing summary,
  - a summary equation off by 1 cent,
  - a date 8 days before the period,
  - zero rows,
  - an unparseable amount cell,
  - an unknown summary label.
- The profile loader rejects an unknown key, a missing section, and a string where a list is expected.
- `git grep` test: `statement.py`, `profile.py` and `reader.py` contain none of the profile's heading strings. `extractor.py` still does until PR3.
- `check_profiles.py` on fixtures:
  - each line matches `^(PASS|FAIL) .+$`,
  - a broken fixture's output contains none of its amounts or merchants,
  - the exit code is correct.

**Done when**
- `python -m pytest` is green.
- Upload of a synthetic statement through the pipeline (DB and Ollama mocked) stores reconciled rows with sections.
- The spending precondition is met.
- `scripts/check_profiles.py` runs on fixtures.
- `senior-reviewer` reports no blockers.
- The user has the command to run on their real statements.

---

## PR3: delete the legacy extractor

Branch: `chore/remove-legacy-extractor`

**Precondition.** The user confirms `scripts/check_profiles.py` prints only `PASS` on their real Wells Fargo statements. Do not start before that.

**Goal.** Remove the LLM-described structure and the line state machine, so there is one extraction path.

**Files**

| File | Change |
|---|---|
| `app/extractor.py` | Delete. Text access and period parsing already moved to `reader.py` in PR2. |
| `app/pipeline.py` | Remove any remaining `extractor` imports. |
| `tests/test_extractor.py` | Delete, and remove it from `collect_ignore`. |
| `.claude/CLAUDE.md` | Layout section: remove `extractor.py`. Also drop it from the "Ollama URL hardcoded in" line. |

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
- #12: statement identity across cards. Deferred in ADR 0002.
- Rejected PDFs stay in `uploads/` (`app/main.py:23-25`).

## Waiting on you

1. **`Transaction.section` migration mechanism.** `create_all()` does not add columns to an existing table. The choice is a hand-written `ALTER TABLE ... ADD COLUMN section TEXT NULL` script you run, or Alembic (a new dependency).
2. **Spending semantics. Must be decided before PR2 merges.** Once fees and interest are rows, should "how much did I spend" exclude them by default? Also: should fee and interest rows go through LLM categorisation and embedding (`app/pipeline.py:70-96`), or get a fixed category?
3. **PR1 wiring:** (a) or (b) above.
4. **Real-statement facts for the profile.** These come from your statement layout; Claude cannot read the real statement.
   - The exact Account Summary labels, the exact purchases heading (with or without the comma), and the exact total-line wording.
   - Do fee and interest rows print a Trans Date?
   - Can a zero-amount interest line appear?
   - Does the Account Summary have lines beyond the six listed (for example cash advances)?
5. **Contracts marked [confirm].** The `StatementData` shape, the 422 body, the nullable `section` column and the profile schema.
6. **Existing data.** Statements already stored were parsed by the unguarded legacy path, have no `section`, and have no fee or interest rows. The `uq_statement_period` constraint blocks re-uploading them. Should they be kept as legacy and unverified, or deleted and re-ingested after PR2 (SQL you run)?
7. **`check_profiles.py` output.** Bare `PASS`/`FAIL` gives you no debug loop when a real statement fails. Printing the failing check name and section key (no values) would help. This is your call, because the current rule is pass/fail only.
