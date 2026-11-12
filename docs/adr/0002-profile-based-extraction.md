# ADR 0002: Profile-based extraction with reconciliation

- **Status:** Accepted
- **Date:** 2026-11-10
- **Supersedes:** no recorded ADR. It replaces the zero-row guard proposal for #6 and #7 (options A to D), which was discussed but never written up. The number 0001 is unused.
- **Related issues:** #6, #7 (silent zero-row parses from `detect_structure`), #8 (`detect_structure` output `KeyError`s into a 500), #3 item 5 (`_parse_date` returns `None`), #12 (statement identity, deferred here).

## Context

`app/extractor.py` asks llama3.2 to describe the statement layout (`detect_structure`) and then walks text lines with a state machine. That design has three properties that conflict with the hard constraint "a parse failure must never silently become a wrong number":

1. The layout description is non-deterministic and only loosely validated. A malformed answer (string instead of list, reversed pages) parses zero rows and the upload returns HTTP 200 (#6, #7, #8).
2. Nothing checks the parsed rows against anything the statement itself prints. A synthetic probe showed one unmatched amount line merges two purchases into one row with the wrong amount, and, in a statement that spans a year boundary, a transaction dated before the period start is assigned the following year. Both succeed silently.
3. The parser reads text line by line, so it has no notion of columns.

Layout facts from one real Wells Fargo credit card statement (no transaction data recorded here):

- The transaction table header is `Card Ending in | Trans Date | Post Date | Reference Number | Description | Credits | Charges`. Money is in either the Credits or the Charges column. There is no single Amount column. (Verified later on a real statement: the header is stacked over three lines, top line `Card | Trans | Post | Reference Number | Description | Credits | Charges`, then `Ending Date Date`, then `in`. See ADR 0003, Context.)
- A statement has 5 tables. Only the 2 with this header hold transactions. The header repeats on the continuation page.
- Inside the table, sections are `Payments`, `Other Credits`, `Purchases, Balance Transfers & Other Charges`, `Fees Charged` and `Interest Charged`. Each ends in a line `TOTAL ... FOR THIS PERIOD`.
- PyMuPDF `get_text("words")`, grouped by vertical position, rebuilt every row. Per-section sums matched every printed total exactly.
- PyMuPDF `find_tables()` mangled the continuation table.
- The Account Summary gives a second, independent check: previous balance − payments − other credits + purchases + fees + interest = new balance.
- PDF page numbers differ from printed page numbers, because there is an unnumbered terms page. Page numbers are not a reliable locator.

The statement prints its own totals, so correctness can be checked against the document rather than trusted.

## Options considered

### Row reader

| Option | How it works | Pros | Cons | Breaks first |
|---|---|---|---|---|
| **Keep `detect_structure` + line state machine** | The LLM names headings and pages. A regex state machine walks lines. | Already written. | Non-deterministic. No column awareness. Silent zero or merged rows (#6, #7). Every call sends statement text to Ollama (local, so privacy holds). | Any model answer outside the expected shape. Already observed. |
| **PyMuPDF words grouped by vertical position + profile (chosen)** | Read word boxes, group them into rows by y, assign columns from the header's x positions, track sections by heading text. | Deterministic. No new heavy dependency (PyMuPDF already used). Rebuilt every row of the real statement. Column-aware, so Credits and Charges are distinct. Fast (unmeasured, but no model call). | Needs a per-bank profile. Has no answer for scanned PDFs without a text layer. The y-grouping tolerance needs tuning. | Statements where wrapped descriptions or skewed rows break y-grouping. Reconciliation turns that into a loud failure. |
| **PyMuPDF `find_tables()`** | Built-in table detection. | No grouping code to write. | Mangled the continuation table on the real statement. | Already failed on page 2. |
| **Docling** | ML layout model that outputs tables. | Handles irregular layouts and possibly scans. | Large new dependency and model weights. It runs locally, but downloads weights on first use. Slower (unmeasured). Less transparent when wrong. | Install size and latency. Its output still needs the same reconciliation. |

### Where bank knowledge lives

| Option | Pros | Cons |
|---|---|---|
| LLM describes the layout at upload time (current) | No per-bank work. | Non-deterministic. The cause of #6, #7 and #8. |
| A Python parser class per bank | Full flexibility. | Bank strings spread through `app/`. Every new bank is a code change and a review of parsing logic. |
| **Declarative YAML profile per bank and account type (chosen)** | The reader is bank-agnostic. A new bank is data plus a fixture. Profiles are versioned (`.v1`). | Only works for layouts the profile vocabulary can describe. The vocabulary will grow. |

### Validation

| Option | Catches | Misses |
|---|---|---|
| Zero-row guard only | Total parse failure. | Dropped, merged and misdated rows. Wrong amounts. |
| **Guard + date window + per-section totals + summary equation (chosen)** | Total failure, dropped and merged rows, out-of-window dates, wrong amounts, and a misread section total (cross-checked by the summary equation). | A wrong date inside the window, a wrong description, and two errors that cancel within one section. Statements that print no totals are rejected by Decision 5. |
| Save and flag (`statements.status`) | Keeps partial data. | Needs a schema change, and every read path must filter on status. A flagged statement leaks into answers until all of them do. |

## Decision

1. **Scope.** Wells Fargo credit card only for now.
2. **Row reader.** PyMuPDF `get_text("words")` grouped by vertical position. Docling is considered only if reconciliation fails on real statements.
3. **Profile.** One hand-written YAML, `profiles/wells_fargo.credit_card.v1.yaml`. The reader takes a profile as input and contains no Wells Fargo strings. There is no profile selection logic yet.
4. **Validation on every statement.**
   - At least 1 row.
   - Every transaction date lies inside the statement window, allowing up to 7 days before `period_start`.
   - Every per-section sum equals its printed total, compared as exact `Decimal`.
   - The Account Summary equation holds.

   On any failure, nothing is saved, and `/upload` returns 422 naming the specific mismatch.
5. **No printed totals.** Reject with 422, `cannot verify: no printed totals`.
6. **Fees and interest.** Store fees and interest rows as transactions, with their section.
7. **Concurrency.** Upload stays synchronous while the reader is fast.
8. **Statement identity.** Stays `(period_start, period_end)` for now.
9. **Replaces the earlier guard proposal for #6 and #7.** Its zero-row guard and reconciliation carry over into Decision 4.

## Consequences

- **Correctness becomes checkable.** A statement ingested after this change is either fully reconciled or not stored at all. This holds on two independent checks, the section totals and the summary equation. Statements already in the database were parsed by the unguarded legacy path and remain unverified.
- **"Save nothing" means the database.** `/upload` writes the PDF to `uploads/` before parsing (`app/main.py:23-25`), so a rejected statement's file stays on disk.
- **The LLM leaves the write path for structure.** It remains only for merchant and category enrichment (`app/llm.py`). Upload no longer sends layout prompts to Ollama.
- **Year inference gets a rule.** Dates are resolved against the statement window with a 7-day lookback. This fixes the observed wrong-year case, and a date outside the window becomes a rejection, not a guess.
- **Validation runs before any insert.** So "save nothing" holds for parse failures. It does not by itself make enrichment atomic with extraction (#3 item 4). That remains open.
- **Fees and interest become rows.** Spending answers change. A question like "how much did I spend" will include interest and fees unless the query excludes those sections. `transactions` needs a `section` column, which is a schema change.
- **422 replaces silent 200s.** Today `/upload` maps every `ValueError` to 409, so the parse rejection needs its own exception type and response shape.
- **Uploads may be blocked between PRs.** Once the guard is wired, the legacy extractor produces no printed totals, so every upload is rejected (Decision 5) until the profile-based reader lands.
- **New direct dependencies.** PyYAML moves into `requirements.txt` (today it is only present through `transformers`). pytest becomes a dev dependency.
- **Other banks are out of scope.** Chase, Discover and Fidelity, scanned PDFs, and any bank that prints no totals are rejected until a profile exists.
- **`app/extractor.py` is deleted** once `scripts/check_profiles.py` passes on the user's real statements.
- **Reversibility.** The profile and reader are additive until the legacy deletion. The `section` column is the only part that is costly to reverse.

## Deferred decisions

- **Statement identity.** `(period_start, period_end)` collides when a second bank or a second Wells Fargo card is added, because both share billing dates (#12). Revisit when either is added.
- **Profile selection.** How a PDF is matched to a profile. Not needed with one profile.
- **Docling fallback.** Revisit only if reconciliation fails on real statements.
- **Async upload.** Revisit if reader plus enrichment time makes synchronous upload unacceptable. Enrichment, not the reader, is the likely bottleneck (unmeasured).
- **Atomic extraction and enrichment.** Whether a failed enrichment should roll back the statement (#3 item 4).
