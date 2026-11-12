# ADR 0004: Code owns every number on the read path

- **Status:** Accepted
- **Date:** 2026-11-11
- **Builds on:** [ADR 0002](0002-profile-based-extraction.md) (the write path reconciles every stored number) and [ADR 0003](0003-extraction-contracts-and-data.md) decision 8 (amounts are stored signed).
- **Amends ADR 0003:**
  - Its consequence "the SQL prompt in `app/query_router.py` must list" Fees and Interest becomes "the known category list must contain them" (see Category list below).
  - Decision 8 kept `is_credit` "for the SQL prompt". After this ADR, no model-written SQL reads it; it remains for `GET /statements/{id}/transactions` (`app/main.py:88`). Removing it is a separate, later decision.
  - Its "the read path must refuse to answer" until the data step is implemented here as: all questions are refused while legacy rows exist.
  - Its consequence that stored 0.00 rows "add to transaction counts and can appear in semantic search" no longer holds on the read path: 0.00 rows are never counted and appear only in `list` (Definitions 1).
  - Its statement that `app/llm.py` does not list `Fees` and `Interest` is kept by splitting the category list (Category list below): enrichment sees only the subset without them.
- **Must land before:** PR2 of the [extraction rebuild plan](../plans/extraction-rebuild.md), the first change that stores signed rows.

## Context

After ADR 0002, every stored number is checked against the statement it came from. The read path gives no such guarantee. In `app/query_router.py`, llama3.2 makes three kinds of decision per question. A SQL question takes three model calls, or four with the retry at `:278`; a semantic question takes two.

1. **Which path the question takes** (`_classify`, `query_router.py:68-90`): SQL or semantic.
2. **The SQL** (`_generate_sql`, `:99-186`): which rows, which filters, `SUM` or `COUNT`, the sort direction. Postgres computes deterministically, but the *meaning* of the number is the model's choice. Guard rails exist only as prompt rules (`:126-141`) and a keyword blocklist (`:191-204`).
3. **The sentence the user reads** (`_format_answer`, `:231-261`). It restates the numbers. Its prompt tells the model the data is always correct and never to say it is missing (`:242`).

A wrong number on this path is silent: the wrong filter, the wrong sort, a misquoted figure. Signed amounts (ADR 0003, decision 8) make this sharper. With charges negative, a query that drops the minus sign:
- reports negative spending,
- ranks the *smallest* merchant first under `ORDER BY … DESC`,
- returns the smallest charge for `MAX`.

The hard constraint "a parse failure must never silently become a wrong number" applies as much to answering as to storing.

The questions in use today (`demo.py:15-46`):
- totals by category and month,
- a transaction count,
- a top-merchant ranking,
- two "looks like" searches.

The retrieval eval (`app/eval_search.py`) covers only searches. Trips (named date ranges) are an existing filter on the SQL path (`query_router.py:27-65`). Ollama 0.33.3 is installed. It supports constrained output against a JSON schema, so a model's answer can be forced into a fixed shape at generation time.

How this is done elsewhere, from general knowledge (no figures quoted):
- **Natural-language analytics products** route questions through a semantic or metrics layer of human-defined metrics, compiled deterministically, rather than free text-to-SQL. Examples are metrics layers and warehouse assistants built on a semantic model.
- **Customer-facing financial numbers** are rendered from the system of record.
- **Generated prose** that must not misstate numbers uses placeholders filled by code.
- **Groundedness checkers** are used for monitoring, not as hard gates.

## Options considered

| Option | How it works | Pros | Cons and cost | Breaks first |
|---|---|---|---|---|
| **A. Keep text-to-SQL, add prompt rules and evals** | Today's design. Add a `-amount` rule for spending, plus eval questions. | Answers any question. No rework. | Wrong numbers stay possible and invisible. Each new failure needs another prompt rule. It does not meet "numbers come from verifiable code". | The first question type no eval covers. |
| **B. Text-to-SQL over safe views** | The model only sees views with positive `spent` and `received` columns, never a signed `amount`. | Removes the sign class of errors. | The model still picks filters, grouping and sort order freely. A view is a schema change. | Filter and grouping errors. |
| **C. The LLM fills a form; code computes and states every number (chosen)** | The model turns a question into a small structured form, constrained by a JSON schema. Code validates it, fetches rows with a parameterised query, computes every number, and writes every number the user reads. | Every number is deterministic and unit-testable without Ollama. Sign handling lives in one place. No model-written SQL, so no injection surface. | Only question types we build are answered; others are refused. New modules and new contracts (the form, the response). Rewrites most of `query_router.py`, so today's SQL-path eval figures no longer apply. | Variety, not data size. Around 15 to 20 question types (unverified), the form becomes a small query language. |
| **D. No LLM: keyword rules** | Regex and keyword matching map questions to templates. | Fully deterministic. | Brittle to phrasing. The model is only needed for the one fuzzy step that C already isolates. | The first rephrased question. |

Privacy is the same for all four: everything runs locally.

## Decision

**Option C.** The language model's job on the read path is limited to two things:
- filling a form from the user's question,
- arranging code-written facts into a sentence (detail 3).

Code:
- validates the form,
- fetches rows with one parameterised query,
- computes every number,
- writes every fact the user reads.

A question that does not fit a supported type is refused, with what is supported, never guessed.

**Legacy rows.** While any legacy unsigned row exists (`transactions.section IS NULL`), **every** question is refused with "re-ingest pending". Signed direction is meaningless on those rows, for counts and lists as much as for totals (ADR 0003, decision 8). The check is removed in PR3, once `section` is `NOT NULL`.

**What would change this:** regular questions that no form can express. The fallback is a hybrid: forms first, then option B as a fallback whose answers are labelled "unverified".

## Details

### 1. Question types

Twelve types: eleven calculations plus `search` (detail 5). Each is one calculation with its own tests. Anything else is refused, with the list of what is supported.

| # | Type | Example | Computed by code |
|---|---|---|---|
| 1 | `total_spent` | "How much did I spend on groceries in December?" | Sum of spent rows, shown positive. Refunds under the same filters are stated beside it, never netted. |
| 2 | `total_received` | "How much did I pay off in December?" | Sum of rows in the chosen direction: `payments`, `refunds` or `received` |
| 3 | `count` | "How many purchases in December?" | Number of rows in the chosen direction |
| 4 | `ranking` | "Where did I spend the most / the least?", "Spending by category" | Merchants or categories ranked by spend, either order. "Least" only counts places with spend in the period. |
| 5 | `largest` | "Biggest purchases?" | Largest single spent rows |
| 6 | `list` | "Show my DoorDash orders in March" | Matching rows, capped at 50 ("showing 50 of N") |
| 7 | `compare` | "Did I spend more May 5 to July 5 than before?", "YTD vs last year?" | Two totals and the difference over any two ranges (Definitions 2) |
| 8 | `monthly_trend` | "How has my spending changed month to month?" | Spend per calendar month |
| 9 | `category_change` | "Which category saw improvement?" | Per category, between two ranges (as `compare`), sorted by largest decrease. Improvement means less spent, including lower fees and interest. |
| 10 | `savings_opportunities` | "Where can I cut down spending?" | Three fixed rules: recurring charges (same merchant in 3 of the last 4 full months, with monthly cost); categories above their own average of previous full months (with the excess); fees and interest paid. |
| 11 | `anomalies` | "Any anomaly categories?" | A category is flagged if the latest full month's spend is more than 1.5× its median over previous full months **and** at least $50 above it. Needs at least 3 previous full months; otherwise "not enough history (have N months)". |
| 12 | `search` | "Find charges that look like parking" | Detail 5 |

- The thresholds (3 of 4 months, 1.5×, $50, 3 months) are unverified starting values. They are named settings in one place, to be tuned once the full history is re-ingested.
- History-based types say so explicitly when history is too short, and never extrapolate.
- **Refused-question log.** A local append-only file holds only the text of questions refused as **unsupported**, or whose form could not be understood. That shows which types are missing.
  - Data-state refusals (legacy rows, no merchant match, no coverage) are not logged: they would fill the log without showing a missing type.
  - The path comes from `FINRAG_REFUSED_LOG`, defaulting to `~/.finrag/refused_questions.log` (expanded at runtime, so no username is committed). It is outside the repo and never committed.
  - A failed write is ignored with a warning and never fails the answer.
  - Claude never reads it, a data-safety rule added to `.claude/CLAUDE.md` in the implementing PR.
  - This stores refused questions only. Detail 7's decision not to store asked questions for evaluation stands.

### Definitions 1: directions on signed rows

Amounts are signed: `−` money out, `+` money back. Stored 0.00 rows (ADR 0003, 4b) are **never counted** in any total, count or ranking; they appear only in `list`.

| `direction` | Rows | Example |
|---|---|---|
| `spent` (default) | `amount < 0`: purchases, cash advances, fees, interest | "How much did I spend?" |
| `payments` | `amount > 0` in section `payments` | "How much did I pay off?" |
| `refunds` | `amount > 0` outside section `payments` (Other Credits, and any refund printed inside a charge section) | "How much came back in refunds?" |
| `received` | `amount > 0` | "How much came back to my card?" |

- Spending totals are gross. When `refunds` rows exist under the same filters, the answer states them beside the total, never netted.
- `payments` and `refunds` together are exactly `received`: every positive row is one or the other.
- `count` counts the chosen direction. The "Understood as" line names it, for example "charges (incl. fees and interest)".

### Definitions 2: coverage and periods

- **Coverage is computed per day** from statement periods (`statements.period_start` to `period_end`). A range is *full* if every day is covered, *partial* if some days are, and *none* otherwise. Rows are bucketed by their own date. A row dated up to 7 days before its statement's start (ADR 0002, decision 4) counts in its own date's period, and coverage reports that period honestly.
- **Totals, counts, rankings, largest, list and search** answer on any period and state partial or no coverage.
- **`compare` and `category_change`** work on any two date ranges:
  - with one range named, the other defaults to the same-length range immediately before it;
  - `ytd` (1 January to today) defaults to the same dates in the previous year;
  - both ranges must be fully covered, otherwise the answer refuses, states the coverage and suggests the largest comparable range;
  - when lengths differ, both totals and the per-day averages are shown, and up or down is decided on the per-day average, stated explicitly.
- **`monthly_trend`** uses calendar months. Partial months are shown, labelled "partial", and never used to say up or down.
- **`anomalies` and `savings_opportunities`** use only full calendar months, for the month under test and its history.
- **Default period when the question names none:**

| Type | Default |
|---|---|
| totals, count, ranking, largest, list, search | all data |
| `compare` | the latest two full months |
| `category_change` | the latest full month against the one before |
| `monthly_trend` | all months |
| `anomalies`, `savings_opportunities` | the latest full month, against the full months before it |

- Relative words the user *says* resolve against today (2a). Defaults the user *does not* say anchor to the latest full month. The "Understood as" line always shows the resolved dates.

### 2. Form fields, dates and matching

The model's output is constrained to this JSON schema (Ollama structured output). Code validation is the real guard: the schema shapes output, and code accepts nothing it has not checked. The model copies words into fields; it never does date arithmetic or name matching.

| Field | Values | Validated against |
|---|---|---|
| `type` | one of the 12 types, or `unsupported` | the fixed list. `unsupported` is refused and logged. |
| `direction` | `spent`, `payments`, `refunds`, `received` (default `spent`) | the fixed list |
| `group_by` | `merchant` or `category` (for `ranking`) | the fixed list |
| `order` | `most` or `least` | the fixed list |
| `limit` | 1 to 50 (default 5) | the range |
| `category` | optional free text | the known category list (case-insensitive exact). No match is refused, listing the categories. |
| `merchant` | optional free text | merchant names in the data (2b) |
| `trip` | optional free text | defined trips (2c) |
| `search_text` | free text (for `search`) | non-empty, length cap |
| `period`, `period_2` | a month (`YYYY-MM`, or `--MM` with no year), a date range (dates may omit the year: `--MM-DD`), a relative token (`this_month`, `last_month`, `last_3_months`, `this_year`, `last_year`, `ytd`), or `all` | resolved by code (2a, 2a′) |

- **Unusable model output** (malformed or truncated JSON, or out-of-range values) is refused with "couldn't understand the question" and logged, never guessed. If the model is unreachable or times out, the result is a 503.
- **2a. Relative dates are relative to today.** Coverage is stated as in Definitions 2. This replaces the current behaviour of anchoring to the latest transaction (`query_router.py:102-106`, `:138-139`), which silently reinterprets the user's words. Every token's range is fixed:
  - `this_month`: 1st of the current month to today,
  - `last_month`: the previous calendar month,
  - `last_3_months`: the three calendar months before the current month,
  - `this_year` and `ytd`: 1 January to today,
  - `last_year`: the previous calendar year,
  - `all`: all data.

  A range ending today is almost never fully covered, because the current billing cycle has no statement until it closes. So `compare` and `category_change` on `ytd`, `this_month` or `this_year` normally refuse, stating the coverage and offering the largest fully covered comparable range (for example 1 January to the last covered day, against the same dates last year). Totals and other single-range types answer with "partly covered".
- **2a′. Dates without a year.** The model never supplies a year the user did not say. A month or day without one ("in December", "May 5 to July 5") is written in the form without a year (`--12`, `--05-05`). Code resolves it to the **most recent occurrence that has started on or before today**: on 11 Nov 2026, "December" is Dec 2025 and "May 5 to July 5" is 5 May to 5 Jul 2026. A range whose resolved end is before its start rolls the start back a year. The resolved year always appears in the "Understood as" line. This follows the 2a principle: relative to today, stated, never inferred from data.
- **2b. Merchant matching is a combination.** Both sides are normalised first: lowercase, with punctuation and repeated spaces removed. A stored merchant matches if:
  - it contains the typed text, and the typed text is at least 3 characters (so "at" does not match "Target"), or
  - it is at least 90% similar to the whole name or to any word in it (`difflib.SequenceMatcher`, standard library).
  - Examples: "costco" finds "Costco Wholesale" by containment. "costcoo" (92%) and "dordash" (93%) match by similarity.
  - Every answer lists the merchants it counted.
  - No match gives "no merchant matching 'X'", with the closest names offered. It never silently picks one.
- **2c. Trip matching.**
  - A trip matches by case-insensitive exact name, or by containment of at least 3 normalised characters.
  - Several matching trips are refused, listing them. A trip whose end is before its start is refused.
  - A trip with an explicit period uses the intersection of the two, and an empty intersection is refused.
  - A trip's range gets the same coverage statement as any range.

### 3. Who writes the sentence

Template first, then a **placeholder rewrite**. The model arranges code-written facts; it never writes them.

1. Code renders each fact as one complete phrase behind an opaque token, **including its verb and direction**. For example, `[F1]` = "you spent $300.00 on Groceries", `[F2]` = "Dining came to $200.00, down $40.00 from November", `[F3]` = "you got $30.00 back in refunds". Code also writes a default sentence from the same tokens: that is `answer_exact`.
2. llama3.2 receives only the tokens (never their contents) and the question type. It writes a natural sentence using each token exactly once.
3. Code checks the result deterministically:
   - every token appears exactly once, in order (later facts refer back to earlier ones, so reordering could attach a fact to the wrong figure),
   - no digits outside tokens,
   - outside tokens, no word from a **closed, enumerated list** kept in `app/read_settings.py`. Every word on it is a test case. It covers:
     - comparison and ranking words (more, less, fewer, up, down, higher, lower, most, least, biggest, smallest, top, bottom),
     - negations (not, no, never, none, without, n't),
     - quantity and number words (zero to twenty, dozen, hundred, thousand, twice, double, triple, half, third, quarter, percent),
     - approximators (about, around, almost, nearly, roughly, approximately, over, under, above, below, just, only),
     - exclusions (except, excluding, besides, other than),
     - direction verbs (spent, spend, paid, received, earned, saved, refunded, charged, owe, owed, cost, costs).

     The list is closed: a word is allowed unless it is on it. Growing the list is a reviewed change.
   - a length cap.
4. If it passes, the tokens are substituted and the result is shown as `answer`. On any failure, or if Ollama is unavailable or exceeds its timeout, `answer` is `answer_exact`. Tables (rankings, lists, trends) are never rewritten.

A number, its subject, its direction and any comparison sit inside one code-written token, and the glue may not contain any word on the closed list, so the model cannot swap, negate, approximate, re-direct or spell out a fact. Formatting is fixed by code:
- money is `$1,234.50`, rounded half-up to cents for averages and medians,
- percentages have one decimal place,
- dates are `5 Dec 2025`.

- **Residual risk:** glue words that imply contrast or judgement ("while", "only", "nice") are tone, not facts, and `answer_exact` is always present. A banned-word list is the next step if needed.
- **Cost:** one rewrite call per answer with a sentence (a few seconds, unmeasured). The eval tracks the rejection rate.
- **Options considered:**
  - template only: kept as the fallback,
  - free-text rewrite with number, name and comparison-word checks: rejected in review, because the checks bind nothing to anything (swaps, negation and spelled-out numbers pass),
  - rewriting only single-fact answers: declined; placeholders make every type safe,
  - labelled commentary: not chosen.

### 4. Showing the interpretation

Every answer carries an "Understood as" line built by code from the validated form, never rewritten. It shows:
- the direction,
- the category, or the matched merchants with the typed text,
- the trip,
- the resolved dates,
- the coverage.

For example: *Understood as: spent · merchants Costco, Costco Wholesale (matched "costco") · last month = 1–31 Oct 2026 · no data for this period.*

- **Why always, despite evals.** Evals and few-shot prompting lower the misread *rate* over questions we wrote. They cannot say which of today's answers was computed for the wrong question. This line is the read path's runtime check, as reconciliation is the write path's. Only the user can check intent.
- **Planned switch:** show the line only when something was inferred (fuzzy match, relative date, default period, partial coverage) once the form-filling eval reaches **at least 98% over at least 100 questions taken from real usage**. That bar was set by the user.
- **Not reachable for now.** Detail 7 is synthetic-only, so the line stays always-on until that is revisited.
- Showing it only in the API response was declined: misreads would be invisible.

### 5. The "looks like" search

`search` fetches spent rows (or the chosen direction) within the similarity cutoff (0.75 cosine distance), with optional filters, capped at 50.
- The old cap of 5 silently dropped matches.
- Respecting direction fixes issue #4 item 4: refunds no longer appear as charges.

The answer lists the matches, with a count and total computed by code over exactly those rows. It says plainly that this is a similarity search that may miss some or include near-misses.
- The 0.75 cutoff was only ever exercised with a cap of 5 (`query_router.py:207-208`); `app/eval_search.py` uses a threshold of 1.0 and a limit of 10. At cap 50 it is unverified, so the search eval is re-run at the new cap.
- Rows without an embedding cannot match, and the answer says how many such rows fall in the period.
- Dropping the total is to be revisited once ADR 0005 categories exist. Dropping search entirely was declined: it loses cross-merchant exploration.

### Growth: new metrics and categories

- **New metrics** are never created at runtime: that would mean model-written SQL again. Refused questions land in the log, and a new metric is added as a reviewed template with tests.
- **New categories** are decided in **ADR 0005** (write path, to follow):
  - categories are the user's list,
  - the model must pick from it (output constrained and validated),
  - when nothing fits, the row is `Other` and flagged, with a suggested name,
  - the user approves a new category together with a deterministic rule that applies to all past and future rows.

  This addresses issues #2 and #4 item 1.

### Category list and enrichment limits

- **The known category list** moves to one module, `app/categories.py`, imported by both the write path (`app/llm.py`, today's home at `:6`) and the read path. That keeps the dependency direction: the read path never imports `llm.py`. It exports two lists:
  - `ALL_CATEGORIES`, used by the read path for validation,
  - `ENRICHMENT_CATEGORIES`, used by the enrichment prompt (`app/llm.py:29`).

  PR2 adds `Fees` and `Interest` to `ALL_CATEGORIES` only. Fee and interest rows never reach the LLM (ADR 0003, 2b), so the model is never offered those labels and cannot put a purchase under them.
- **Until ADR 0005, `merchant` and `category` are LLM enrichment output.** They are unvalidated and regenerated on re-ingest (`app/pipeline.py`). So:
  - merchant rankings can split one store across name variants,
  - recurring-charge detection can under-detect,
  - a category total omits rows the model labelled differently.

  Answers are computed exactly *over the stored labels*. They are not verified against what the labels should be. Every answer states how many rows in its period are flagged `needs_review` or carry a category outside the known list, so this limit is visible.

### Testing the numbers (review blocker B2)

**Pure core, thin I/O.**
- **The SQL is thin, and filters only by date.** The queries are:
  1. every row in the resolved date range or ranges, with its `amount`, `section`, `category`, `merchant`, `needs_review` and whether it has an embedding,
  2. the statement periods (for coverage),
  3. the distinct merchant names (for matching),
  4. the trips,
  5. the legacy-row check,
  6. the pgvector search.

  Direction, category and merchant filters are applied in Python on the rows from query 1. That is how refunds beside spending, the `needs_review` and off-list counts, and the no-embedding count come from the same fetch.
  - The one exception is `search`: direction and the date range are applied in SQL before the cap, because the cap must apply to the matches that count. Sign handling for it is a parameter supplied by the calculation module's direction definitions, so the definitions still live in one place.
- **All calculation is plain Python over `Decimal`:** totals, counts, rankings, month buckets, medians, averages, the anomaly and recurring rules, comparisons and coverage. **Unit tests** cover it exhaustively without Ollama or Postgres.
- **An integration suite** covers all six queries against `finrag_test`, a throwaway database on the local Postgres, with synthetic rows only. Queries take an injected session, so the suite binds them to `finrag_test` rather than to the engine `app/models.py` builds at import. The suite creates its tables from `Base.metadata`, not `create_tables()`.
  - It is marked `integration` and run on demand (`pytest -m integration`).
  - It connects only through `FINRAG_TEST_DATABASE_URL`, and refuses unless the database name is exactly `finrag_test`.
  - It drops its tables after each run, and never touches `finrag`.
  - The user approved this exception to the "no Postgres in tests" rule.
  - `finrag_test` was created on 2026-11-11, with pgvector 0.8.2.
- **Scale limit:** fetching rows into Python suits one person's statements (thousands of rows). It degrades somewhere around 10⁵ to 10⁶ rows (unverified), at which point aggregation moves back into SQL.
- **Options considered:**
  - all math in SQL with every template integration-tested: declined, because most tests would then need Postgres,
  - asserting SQL text or fake sessions: rejected in review, because it cannot prove numbers.

### 6. The `/query` response

A clean contract. `demo.py`, the only client, is updated in the same PR and shows `understood_as` and `answer`.

| Field | Content |
|---|---|
| `type` | one of the 12 types, or `refused` |
| `answer` | the accepted placeholder rewrite, else `answer_exact` |
| `answer_exact` | the code's template answer |
| `rewrite` | `accepted`, `rejected` or `skipped` (Ollama down or timed out, or a table answer) |
| `understood_as` | the interpretation line |
| `interpretation` | the validated form: direction, category, matched merchants, trip, resolved dates, coverage (`full`, `partial`, `none`) |
| `result` | the computed numbers; money as strings |
| `rows` | the rows used (capped at 50, with `rows_total`); dates and signed amounts as strings |
| `notes` | data-quality notes: rows flagged `needs_review` or off-list in the period, rows without embeddings (search) |
| `query` | the template name and its parameters, replacing `sql` |

| Outcome | HTTP |
|---|---|
| Answered | 200 |
| Refused (unsupported, not understood, unknown category, no merchant or trip match, ambiguous trip, coverage insufficient for a comparison, legacy rows present) | 200, `type: refused`, with the reason and what is supported |
| Local model unreachable or timed out at the form step | 503 (today this is a 200 with `type: "error"`) |
| A bug | 500 |

- **Timeouts:** 30 seconds for the form call and 15 seconds for the rewrite, inside `demo.py`'s 60 seconds. A rewrite timeout falls back to `answer_exact`.
- **The read path goes dark until re-ingest.** Every current row is legacy, so from this PR's merge until the data step, every question is refused with "re-ingest pending". That is the intended behaviour, not a regression.

### 7. Evaluating the form-filling step

A script, run by hand with Ollama up, not a unit test. It sends each eval question to the served llama3.2 and compares the form with the expected form, field by field. It reports:
- overall accuracy,
- accuracy per type,
- accuracy per field,
- the rate at which placeholder rewrites are rejected.

- **Questions: synthetic only.** Roughly 80 questions covering all 12 types, varied phrasing, and questions whose expected form is `unsupported`. Invented merchants and trips; committed to the repo.
- **Split:** few-shot examples come from a separate "teach" set and are never test questions, so the score cannot be inflated by the prompt containing its answers.
- **Model under test:** the served Q4 llama3.2 through Ollama. Earlier figures measured a bfloat16 HuggingFace build and do not transfer.
- A local set of real, user-labelled questions was declined, to avoid storing asked questions. Consequence: the detail 4 switch cannot be met, and synthetic scores say little about real phrasing.

## Consequences

- **Every fact the user reads is written by code:** totals, counts, rankings, comparisons and the words "more", "less" and "biggest", inside placeholders. Computed values are exact over the stored rows and labels. The labels themselves (merchant, category) remain LLM enrichment until ADR 0005, and answers say how many rows are flagged. The model's influence is limited to understanding the question, which is always shown and validated, and to glue words in the rewrite.
- **Twelve question types.** Others are refused honestly, and unsupported ones are logged locally to drive new templates.
- **`app/query_router.py` is largely rewritten.** `_classify`, `_generate_sql`, `FORBIDDEN_SQL`/`_validate_sql`, `_schema_for` and `_format_answer` go away. The SQL path's previous eval figures no longer apply.
- **LLM calls:** one for the form, plus one rewrite for answers with a sentence, instead of two to four today. The latency effect is unmeasured.
- **Relative dates mean calendar time,** and a missing year is resolved by code, never by the model. Comparisons work over any fully covered ranges. YTD compare usually refuses until the current cycle's statement exists, and offers the largest covered comparable range.
- **Signed amounts are safe on the read path:** the direction definitions live only in the calculation module, and the search query receives them as parameters.
- **New modules:** `app/question_form.py`, `app/answers.py`, `app/calculations.py` (pure), `app/read_queries.py` (the thin SQL), `app/categories.py` and `app/read_settings.py`.
- **New contracts:**
  - the form schema,
  - the `/query` response,
  - a 503 for an unavailable model,
  - the refused-question log (`FINRAG_REFUSED_LOG`),
  - `FINRAG_TEST_DATABASE_URL` for the integration suite.
- **Settings with unverified starting values,** in one place:
  - anomalies (1.5×, $50, 3 months),
  - recurring (3 of 4 months),
  - merchant similarity (0.90, minimum 3 characters),
  - search (cutoff 0.75, cap 50),
  - list cap (50), default limit (5),
  - timeouts (30 s, 15 s).
- **Dependency:** the eval's question files are YAML, so `pyyaml` becomes a direct dependency in this PR rather than in PR2.
- **Follow-up decision: ADR 0005,** user-owned categories with deterministic rules (write path).

## Implementation notes (read-path PR, 2026-11-11)

- **Few-shot examples** live in `app/query_router.py` (`TEACH_EXAMPLES`), because the app uses them at runtime and `app/` must not import `evals/`. `evals/read_path/run.py` refuses to run if any test question overlaps them, which keeps the teach/test split of detail 7.
- **A whole year** (`YYYY`) was added to the period forms, because users say "in 2026" and the model writes it that way.
- **Table answers are never rewritten,** including their summary line. Only the sentence types (`total_spent`, `total_received`, `count`, `compare`, `search`) get the placeholder rewrite.
- **History and comparison types fetch every row.** Their history reaches outside the asked period; this stays within the scale limit recorded under Testing.
- **The transitional `transactions.category` text column is dropped** (ADR 0005). An existing database was migrated once by hand (the SQL file was later removed); a fresh one gets the current schema from `scripts/reset_db.py`.
- **The search eval's cases moved out of the repo** to `~/.finrag/search_eval.yaml`, because they named real merchants. `evals/search_eval.example.yaml` shows the format with invented names.
- **Eval baseline,** 79 synthetic questions, served Q4 llama3.2 through Ollama, temperature 0:

  | Prompt | Form accuracy | Rewrites accepted |
  |---|---|---|
  | First version | 11/79 (13.9%) | 10/50 (20%) |
  | Stricter field rules, category vs merchant, `YYYY`, 17 few-shot examples; the rewrite prompt told that tokens are hidden complete clauses | 31/79 (39.2%) | 50/50 (100%) |

  The second row also scores unsupported questions on their type only, since the resolver ignores their other fields. The weakest types are `largest` (0/4), `search` (0/5), `savings_opportunities` (0/4) and `ranking` (1/8); the weakest field is `period` (43/69).
  - At 39%, misreads are common. The answer is still exact for what was understood, and the "Understood as" line shows the reading on every answer, which is why that line stays always-on (detail 4).
  - Raising form accuracy is prompt work for a follow-up, measured with the same eval.
  - The 100% rewrite rate comes from very similar output at temperature 0 ("Here is what I found: …"). It is natural but repetitive.
- **Code review (read-path PR) changes:**
  - **Fields a type does not apply are dropped at resolve time,** so the "Understood as" line never shows a filter that was not computed. For example, `anomalies` and `savings_opportunities` take only a period.
  - **Search applies category and merchant in SQL before its cap,** and says when the cap was hit.
  - **Monthly trends** treat a month clipped by the asked range as partial, and compare only whole, consecutive months.
  - **"No data" only when nothing was found:** coverage `none` with rows present still answers.
  - **`category_change` shows totals and per-day figures** when lengths differ.
  - **Notes cover every period an answer used.**
  - **A stated year is never rolled back.**
  - **The closed glue-word list grew** by rate words (per, each, every, daily, weekly, monthly, average, …), negation and doubt (cannot, neither, nor, hardly, …), conditionals and hedges (if, unless, maybe, might, would, …) and judgement words (good, bad, great, nice, well, …). The rewrite prompt no longer suggests "Good news" or "while".
  - **Re-run eval:** form accuracy 31/79 (39.2%), rewrites 50/50, unchanged.
- **Final review of #22 (fixed):**
  - a fact states every applied filter ("on Groceries at Costco");
  - a yearless range never spans more than a year, including ranges touching 29 Feb in or after a leap year;
  - years outside 1990-2100 are not understood;
  - overlapping comparison periods are refused rather than double-counted;
  - a refunds-only period is never reported as "No data";
  - the rewrite prompt names the question type, tokens must stay in order, and the glue may contain only plain ASCII words and punctuation;
  - the closed list adds or, instead, minus, plus, until, apart, lacking, supposedly, excluded, included;
  - `app/read_types.py` holds the plain data types, with no I/O imports.
