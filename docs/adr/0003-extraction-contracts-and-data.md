# ADR 0003: Extraction contracts, spending semantics and legacy data

- **Status:** Accepted
- **Date:** 2026-11-10
- **Builds on:** [ADR 0002](0002-profile-based-extraction.md). It settles the questions ADR 0002 and the [extraction rebuild plan](../plans/extraction-rebuild.md) left open.

## Context

ADR 0002 chose a profile-driven, position-based reader that reconciles every statement against its printed totals and saves nothing on a mismatch. Before implementation, seven questions were open:
- a schema change with no migration tooling,
- how fees and interest affect "spending",
- when the guard goes live,
- layout facts only the real statement can answer,
- four contracts,
- what to do with already-stored data,
- what the real-statement check may print.

Layout facts supplied by the user from a real Wells Fargo credit card statement (layout only, no transaction data):

- **Period line.** `Statement Period MM/DD/YYYY to MM/DD/YYYY` is printed in the header of every numbered page.
- **Purchases heading.** `Purchases, Balance Transfers & Other Charges`, with the comma.
- **Section totals.** `TOTAL PAYMENTS FOR THIS PERIOD`, `TOTAL OTHER CREDITS FOR THIS PERIOD`, `TOTAL PURCHASES, BALANCE TRANSFERS & OTHER CHARGES FOR THIS PERIOD`, `TOTAL FEES CHARGED FOR THIS PERIOD`, `TOTAL INTEREST CHARGED FOR THIS PERIOD`.
- **The table header is stacked** over three lines: `Card Trans Post Reference Number Description Credits Charges`, then `Ending Date Date`, then `in`. The profile matches the top line as `table.header` and lists the two lines below as `table.header_continuation`; the reader skips them only directly under the header, in that order, and matches their text exactly. Found by `check_profiles` on the first real statement (2026-11-11).
- **A year-to-date trap.** A "Totals Year-to-Date" box prints `TOTAL FEES CHARGED IN <year>` and `TOTAL INTEREST CHARGED IN <year>`. These must not match a section total.
- **Account Summary, in order.** `Previous Balance`, − `Payments`, − `Other Credits`, + `Cash Advances`, + `Purchases, Balance Transfers & Other Charges` (wraps over two lines; the amount sits on the first line, `Other Charges` continues below it), + `Fees Charged`, + `Interest Charged`, = `New Balance`.
- **The summary is two columns.** The right column (`Total Credit Limit`, `Cash Advance Limit`, `Total Available Credit`, `Available for Cash Advances`) shares the same rows.
- **No Cash Advances section appeared,** so its table heading is unknown.
- **Interest rows print no Trans Date and no Post Date.** Fee rows are unknown, since none appeared.
- **Months with no interest.** Interest Charged prints both interest rows at 0.00 and a $0.00 total. Fees Charged with no fees prints only its $0.00 total line.
- **Payment rows** have a blank "Card Ending in" cell.

## Options considered and decisions

| # | Question | Options | Decision |
|---|---|---|---|
| 1 | How to add `transactions.section` (`create_all()` never alters existing tables) | A: numbered SQL scripts the user runs. B: Alembic. | **A.** One column on a single-user database. No new dependency, readable, one-line undo. Revisit with Alembic at the next schema change (statement identity, #12). |
| 2a | Do fees and interest count as spending? | A: include them. B: exclude by a prompt rule. C: exclude through a view. | **A, with separate `Fees` and `Interest` categories.** No filter on the read path. A breakdown by category shows them apart from purchases. |
| 2b | Do fee and interest rows go through LLM enrichment? | A: yes. B: no, category fixed from the section. | **B.** Deterministic labels, avoiding the inconsistency in #2. They are still embedded. Revisit when more banks are added. |
| 3 | When does the guard go live? | A: in PR1, via a throwaway legacy adapter, rejecting every upload until PR2. B: build it in PR1, wire it in PR2. | **B.** No throwaway code. Uploads pause by agreement, not by code, until PR2. |
| 4a | Date for rows with no printed date (interest, and fees if undated) | A: the statement closing date (`period_end`). B: NULL. | **A.** Keeps `trans_date` required, and keeps interest inside date-filtered questions. |
| 4b | Printed 0.00 interest rows | A: skip them. B: store them. | **B.** Stored as `0.00`. A zero row counts as a charge (`is_credit = 'false'`). |
| 5a | Parsed-statement contract | Dataclasses, or pydantic models. | **Dataclasses** (`StatementData`, `Row`, `SummaryLine`, `Mismatch`), as written in the plan. `StatementData.reader_mismatches` carries reader-found problems to the validator. One fixed list of check names (`CHECKS`) is the only vocabulary for failures. |
| 5b | 422 body | Structured, or a single string. | **Structured:** `{"detail": {"reason", "mismatches": [{check, section, expected, actual}]}}`, with money as strings. |
| 5c | Is `section` nullable? | Nullable, or required. | **Required in the end state.** Nullable only until the legacy rows are deleted (decision 6), then made `NOT NULL`. |
| 5d | Profile format (section keys are a canonical set: `payments`, `other_credits`, `purchases`, `cash_advances`, `fees`, `interest`) | YAML, TOML, JSON or Python. | **YAML** (ADR 0002 Decision 3 stands). TOML was considered: it is in the standard library and has no implicit typing. It was declined. The loader rejects unknown keys, missing keys and wrong types, which neutralises YAML's implicit typing. |
| 6 | Statements already stored by the legacy path | A: keep as they are. B: delete and re-ingest. C: keep and flag as unverified. | **B**, only after `check_profiles.py` passes on every real statement. The user runs the delete script and re-uploads. |
| 8 | Direction of money (found in the PR1 review): neither check looked at the credit flag, so a Charges amount read as a Credit would pass both | A: no check. B: a fixed side per section, plus a new check `credit_side`. C: signed totals. | **Signed amounts, stored signed in the database.** From the cardholder's side: payment, refund or credit is `+`; charge, fee or interest is `−`. `Row.is_credit` is dropped from the contract, so sign and flag cannot disagree. The `is_credit` column stays, derived from the sign at insert, for the SQL prompt. Each section's signed row sum must equal its printed total times minus that section's summary sign, so payments sum to `+total` and purchases to `−total`. A single wrong sign moves the sum by twice the amount and is always caught. A section mixing refunds and charges nets correctly; there, two wrong signs of equal size cancel out, as with any total-based check. No new check name. **Limit:** deriving direction from the summary sign holds for liability (credit card) accounts only. An asset account (checking, savings) reverses it, so its first profile must trigger a decision on an explicit per-section direction. |
| 7 | `check_profiles.py` output | A: `PASS`/`FAIL` only. B: on failure, also print check names and section keys. | **B.** One line per file: `PASS`/`FAIL` and the file name, plus on failure only the failing check names and section keys. Never amounts, dates, merchants or other statement text. A crash is reported as the check `unexpected_error`. A test enforces that every printed word after the file name is a check name or section key. |

## Consequences

- **This amends ADR 0002.**
  - ADR 0002's consequence "uploads may be blocked between PRs" becomes "uploads are paused by agreement" (decision 3).
  - ADR 0002 Decision 5's reason `cannot verify: no printed totals` is reported as the check `no_printed_totals`, under the fixed 422 reason "statement did not reconcile" (5b).
- **Stored 0.00 rows** add to transaction counts and can appear in semantic search results (decision 4b).
- **Signed amounts and the read path (decision 8).** With charges negative, an LLM writing SQL could drop the minus sign: negative spending, or `ORDER BY ... DESC` and `MAX` returning the *smallest* charge, a silent wrong ranking. The read path therefore moves to code-owned numbers (option C, ADR 0004). Until the data step deletes the legacy unsigned rows (`section IS NULL`), the read path must refuse to answer.
- **Spending totals rise from PR2 onward,** because they now include fees and interest. Old months stay comparable only because they are re-ingested (decision 6).
- **The category list gains `Fees` and `Interest`.** The SQL prompt in `app/query_router.py` must list them. `app/llm.py` does not, because those rows never reach it.
- **Uploads should pause** from PR1 until PR2 lands. Nothing in code enforces this.
- **Three SQL scripts, all run by the user:**
  - add `section` nullable (before PR1 code runs),
  - delete legacy statements and transactions (after all real statements PASS, and after a `pg_dump` backup),
  - make `section` `NOT NULL` (after re-upload).
- **Re-ingesting regenerates merchant names and categories,** which may differ from today's.
- **Fixed dates on undated rows.** Interest rows carry `period_end` as their date. That is a convention, not a printed date.
- **The profile grows** to cover:
  - undated sections,
  - the left-column bound of a two-column summary,
  - wrapped labels,
  - a Cash Advances summary line with no known section heading. A nonzero value fails closed.
- **The data-safety rule in `.claude/CLAUDE.md`** now allows check names and section keys in `check_profiles.py` output. That rule is the binding text for decision 7.
