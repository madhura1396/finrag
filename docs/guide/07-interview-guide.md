# Interview guide: telling FinRAG's story

Status: drafted 2026-11-12. Drawn from `docs/guide/`, `docs/adr/0002`–`0005`, `docs/plans/extraction-rebuild.md` and `docs/findings-and-recommendations.md` on `main` at `2fdd28a`.

Use this for a system design or senior engineering interview. It covers the one-minute pitch, the hard constraints, the two big architectural migrations (LLM → deterministic), the label system, trade-offs, scale limits, and likely questions with answers.

---

## The one-minute pitch

FinRAG is a personal finance assistant that runs entirely on one laptop. You upload PDF credit card statements; every transaction is stored in Postgres with a category, a merchant name and a 384-dimension embedding (pgvector). You ask plain-English questions ("How much did I spend on travel in March?", "Any charges that look like parking?") and get exact answers.

Both models run locally: llama3.2 through Ollama, and all-MiniLM-L6-v2 through sentence-transformers. No statement text, transaction or embedding leaves the machine.

It is also a deliberate exercise in senior engineering: every architectural choice is recorded in `docs/adr/` with options, trade-offs and consequences.

---

## The three hard constraints (enforced by code, not convention)

| Constraint | What it means | Enforcement |
|---|---|---|
| **Privacy** | Statement content never leaves the machine | Local models only; Ollama reached at `localhost`; embeddings in-process; API refuses non-loopback clients with 403; real merchant text lives only in the DB or `~/.finrag/`, never the repo |
| **Money is `Decimal`, never float** | Exact cents; exact equality for reconciliation | `Numeric(10, 2)` storage; strict regex parse into `Decimal`; every number computed in `app/calculations.py`; money leaves the API as strings |
| **Fail loudly** | A parse failure never becomes a wrong number | Reconciliation rejects unreconciled statements (422, nothing saved); model can't label a row → `pending`, never `Other`; model misreads a question → refused with "Understood as" line; Ollama down → 503, not a made-up answer |

---

## The architecture in two migrations

FinRAG was rebuilt twice, and each migration is a good interview story.

### Migration 1: Write path — LLM-described layout → profile reader (+ reconciliation)

**Before (`app/extractor.py`, deleted):** llama3.2 described the statement's layout, then a regex state machine walked text lines. Failures: non-list section markers silently parsed zero rows (#6), `end_page < start_page` silently parsed zero rows (#7), `detect_structure` validated only 3 of 6 keys so the rest `KeyError` into a 500 (#8). Nothing checked the result against the statement, so merged rows and wrong years succeeded silently.

**After:** A bank-agnostic contract (`StatementData`) and validator, a word-position PyMuPDF reader driven by a declarative YAML profile, signed amounts, and reconciliation on two independent checks (per-section sums vs. printed totals; Account Summary equation vs. new balance). The LLM left the write path for structure entirely — it remains only for labels.

Lesson: the LLM was doing a job with a deterministic answer and a verifiable output. Moving it out turned "probably right" into "provably reconciled or rejected."

### Migration 2: Read path — LLM-generated SQL → code-owned numbers (ADR 0004, option C)

**Before:** llama3.2 classified SQL vs. semantic, wrote the SQL, and restated the result in prose. Wrong filters, sorts or misquoted figures reached the user as confident answers. Signed amounts made it worse (a dropped minus sign reports negative spending; `ORDER BY … DESC` ranks the smallest first). Guard rails were prompt rules and a keyword blocklist.

**After (option C):** The model fills a schema-constrained form (temperature 0, JSON schema). Code validates it, fetches rows with one thin date-filtered SELECT, computes every number in pure Python `Decimal`, writes every fact as a complete phrase behind an opaque token, and shows an "Understood as" line on every answer.

| Option | Verdict | Why |
|---|---|---|
| A. Keep text-to-SQL + more prompt rules | Rejected | Wrong numbers stay possible and invisible |
| B. Text-to-SQL over safe views | Rejected | Model still picks filters, grouping, sort |
| **C. LLM fills a form; code computes (chosen)** | **Adopted** | Every number deterministic and unit-testable; no model-written SQL → no injection surface |
| D. Keyword rules only | Rejected | Brittle to phrasing |

Trade-off: only the 12 built question types are answered; others are refused and logged. Variety, not volume, is the limit (~15–20 types estimated before the form becomes unwieldy).

---

## The label system (ADR 0005)

Every label comes from exactly one source, per field (category and merchant independently), strongest first:

```
section > pin > rule (by priority) > memory > model
```

- **Section** (fees, interest): fixed at insert, cannot be overridden or pinned.
- **Pin:** user-set on one row, one field; only the user removes it.
- **Rules:** user-written patterns (`contains` or regex) over the printed description; highest priority wins; equal-priority conflicts mean neither applies (fail loudly).
- **Memory:** remembered per normalised description key (digits dropped, so `GROCER MART #101` and `#202` share a key). Only real categories remembered — `none_fit` and failures never written.
- **Model:** llama3.2, temperature 0, JSON-schema-constrained output, batches of 15; constrained to real categories or `none_fit`.

### Changes go through preview → apply
Every multi-row change (create/rename/merge/delete category, create/edit/delete rule, set memory, undo) is previewed inside a savepoint and rolled back, then applied atomically under a `data_version` lock with a token binding the exact payload to the version. A later change or upload bumps the version → 409, preview again. Single-row `pin`/`unpin` apply directly, with no token.

### History is append-only SCD2
`transactions` holds current labels; `transaction_label_history` mirrors them exactly (`valid_from`/`valid_to`). Undo is a compensating change (`undoes = N`), never a history rewrite, allowed only when no later, not-yet-undone change touched the same rows, rules, memory or categories (so overlapping changes are undone newest first).

### Embeddings
`merchant + raw_description + category`, 384-dim MiniLM, pgvector. Staleness is derived (`is_stale`), not flagged, so a lost background task self-heals on the next refresh. Rows re-embedded in batches of 32 after label changes.

---

## Trade-off answers (likely interview questions)

**Q: Why not text-to-SQL?**
> A wrong filter, sort or misquoted figure reaches the user as a confident answer. Prompt rules and blocklists can't close that gap. Code-owned numbers mean every figure is deterministic, unit-testable, and there is no model-written SQL injection surface. The cost is question variety: unsupported types are refused honestly, and refusals are logged to drive new templates.

**Q: Why reconcile against the statement instead of trusting the parser?**
> The statement prints its own totals, so correctness is checked against the document, not trusted. Two independent checks (section sums + summary equation) must both pass, or nothing is saved (422 with every mismatch). A wrong sign moves a section sum by 2× and is always caught. Known limit: two equal errors cancelling in one section pass — shared by any total-based check.

**Q: How do you handle model mistakes?**
> Never silently. A row the model can't label stays `pending` and is flagged for review, never counted as `Other`. A `none_fit` answer opens a review item with the suggestion. A misread question is refused with an "Understood as" line so the user sees exactly how it was read. Model-chosen labels are final until the user corrects them via memory, rules or pins.

**Q: Why one database for relational + vector data?**
> So a similarity search can filter by date, sign, section, category and merchant in the same SQL as the vector order. Two stores (e.g. Postgres + Chroma/FAISS) mean keeping them consistent and splitting filters across them. Row locks, partial unique indexes and check constraints carry the ADR 0005 invariants in one place. Trade-off: no pgvector index yet — sequential scan, fine at thousands, HNSW around 10⁵ rows.

**Q: How does undo work without rewriting history?**
> Undo is a new change (`undoes = N`) that restores `before` snapshots and opens new history versions. Closed versions are never edited. It's allowed only if no later change touched the same rows, rules, memory or categories — otherwise refused, listing the blockers. After an undo, rows are re-resolved by normal precedence, so a formerly `model` row can come back as `memory`.

**Q: What's the privacy model?**
> Both models local; API loopback-only (403 otherwise); real data never in the repo. The label endpoints write real data with no authentication, so loopback is the whole defence — never expose beyond your machine. Refused questions and search-eval cases live in `~/.finrag/`, outside the repo, and are never read by tooling.

**Q: Why YAML profiles instead of a parser class per bank?**
> Bank strings (headings, totals, summary labels, column anchors) live in declarative, reviewable, versioned YAML. A new bank is data plus a fixture, not a parser — bank knowledge doesn't spread through `app/`. A strict loader neutralises YAML's implicit typing (`unquoted no` → `False` is refused). Trade-off: only layouts the profile vocabulary can describe; no profile selection yet (one bank at a time).

**Q: What breaks first at scale?**
| Axis | First limit | Rough number |
|---|---|---|
| Rows fetched into Python (read path) | History types fetch every row | 10⁵–10⁶ rows; then aggregation moves to SQL |
| Label change latency | `relabel_all` re-resolves every row under the lock | tens of thousands |
| Vector search | No pgvector index; sequential scan | fine at thousands; HNSW around 10⁵ |
| Embedding throughput | One row at a time, no lock | noticeable after a large rename/merge |
| Question variety | Form as a query language | ~15–20 types |
| Banks/accounts | Single profile; identity on period alone | second card or bank |
| Users | No auth; loopback only | second user or network exposure |

All unverified except code inspection — they're starting values to be tuned with real history.

---

## Open issues worth mentioning

Selected from the ranked list in `docs/findings-and-recommendations.md` (triaged 2026-11-11), most urgent first:
1. `/upload` uses the raw client filename — path traversal and same-name overwrite. Fix first.
2. Statements unique on period alone — second card with same period collides (needs an ADR).
3. `is_credit` is still a `String` column — Boolean change needs a schema change + ADR.
4. Embedder has no lock and encodes one row at a time.
5. llama3.2 form accuracy 37.8% — larger local model needs a new-model ADR.
6. No authentication — decide before exposing beyond this machine.

---

## Anti-patterns to call out (what we deliberately did not do)

- **No model-written SQL.** The model fills a form; code writes the query and every number.
- **No float for money.** `Numeric(10, 2)` and `Decimal` everywhere; exact reconciliation.
- **No silent `Other`.** `none_fit` and failures are flagged (`pending` / `needs_review`), never silently counted.
- **No history rewrites.** Undo is a compensating change; SCD2 is append-only.
- **No real data in the repo.** Statements, rules, memory and search-eval cases live outside the repo (in `uploads/`, the database or `~/.finrag/`); fixtures are synthetic with invented merchants.

---

## Source of truth

The code is the source of truth. `README.md` may lag — never quote it as fact without checking the code. Decisions live in `docs/adr/0002`–`0005`; plans in `docs/plans/extraction-rebuild.md`; wrap-up status in `docs/findings-and-recommendations.md`.
