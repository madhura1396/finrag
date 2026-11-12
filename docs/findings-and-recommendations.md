# FinRAG: findings and recommendations

Status: wrapped up, 2026-11-11. Totals below are after the final cleanup (598 unit, 60 integration tests). This records where the project stands, what the
measurements say, and what to do if work resumes. Decisions live in `docs/adr/`.

## What was built

| Area | State | Record |
|---|---|---|
| Write path | PDF → `reader.py` (word positions, driven by `profiles/*.yaml`) → `statement.validate` → save. A statement that does not reconcile against its printed totals and Account Summary is rejected and nothing is saved. | ADR 0002, 0003 |
| Categories | User-owned categories with history (SCD2). Labels follow a precedence: section, pin, rule, memory, model. Changes go through preview/apply with a change log and undo. | ADR 0005 |
| Read path | llama3.2 fills a validated form (trips removed as out of scope). SQL only filters by date; every number is computed in Python `Decimal`. Answers are exact and always carry an "Understood as" line. | ADR 0004 |
| Safety | Every database carries an environment marker, `reset_db` is guarded, the API only accepts connections from this machine, and `check_profiles` prints only pass/fail. | CLAUDE.md |

All of it runs locally: Postgres with pgvector, Ollama llama3.2, and MiniLM embeddings.

## Findings

### 1. The write path works on a real statement
- **First run failed.** The first real Wells Fargo statement failed `check_profiles`. The table header is printed over three lines, and the Account Summary prints the purchases amount beside the first line of its wrapped label.
- **Fixed in the profile (PR #24).** The profile now matches the real layout, and the reader skips the header's continuation lines only in exact order.
- **Diagnosed without exposing data.** The diagnosis used scripts that print profile words and yes/no only, never statement text.
- **Result.** The statement passes. All 38 rows were saved and reconcile to the printed totals.
- **Lesson.** Synthetic fixtures encode guesses. One real statement found two layout facts that 650 unit tests could not. Run `check_profiles` on each new statement before trusting it.

### 2. The read path is safe but llama3.2 reads questions poorly
The form eval (`python -m evals.read_path.run`) uses 82 synthetic questions and scores the model's form *before* code runs.

| Measure | Result |
|---|---|
| Whole form correct | 31/82 (37.8%); baseline 31/79 (39.2%) |
| Question type correct | 57–63 of 82 |
| Weakest kinds | search 0/7, largest 0/4, savings 0/4, ranking 1/8 |
| Weakest field | period, 46–48 of 72 (invented years, unfilled `--MM`, extra second period) |
| Latency | median ≈ 1.0 s per question |

What a miss means:
- **Usually a refusal** ("couldn't understand"). Sometimes the wrong question is answered, and the "Understood as" line shows it.
- **Never an invented number.** That is the guarantee the design buys.

### 3. Prompt-side fixes have hit their limit on this model
| Experiment | Effect | Kept? |
|---|---|---|
| 4 few-shot examples of "spend on <brand>" → merchant | 39.0%, no change; merchants still land in `category` | No |
| `category` restricted to real names (schema enum) | Merchants and trips were silently mapped to a real category (e.g. a store → Groceries). A loud failure became a silent wrong answer. | No (reverted) |
| Code-side validation (PR #26) | Placeholder `"null"` values ignored; an unknown category becomes a merchant only by whole words and only when not near a category name; search words on a non-search type → "not understood" | Yes |

Lesson: constraining a small model's output can turn a refusal into a confident wrong answer. Measure before keeping any model-side change.

### 4. Categorisation needs the user's input to get good
- **First upload.** The model labelled 36 of 38 rows, but 20 rows in 13 merchant groups came back `none_fit`, and grocery stores were labelled Shopping.
- **The fix is data, not code.** A `set_memory` change per merchant group fixes the rows and every future statement from that merchant.

## Recommendations (in order)

0. **Fix the `/upload` filename issue** (see GitHub issues below). It is small and safe to do first.
1. **Clear the review queue.** Use `set_memory` for the 13 merchant groups in `GET /review`. This is the biggest quality gain for the least work.
2. **Evaluate a larger local model for the read path.** Run the same form eval on 2–3 local 7–8B models and compare accuracy, latency and memory against llama3.2. Adopt one only if it clearly improves type and period accuracy. This needs a new-model decision (ADR); privacy is unchanged because it stays local.
3. **Make `eval_search` aggregate-only.** Add a `--summary` mode that prints only scores, so the semantic search eval can be run without printing real merchant text.
4. **Friendly reply to greetings and off-topic messages.** Add a `greeting`/`off_topic` kind that returns a fixed help message. It was approved in principle and never built.
5. **Before adding another bank:** write its profile, run `check_profiles` on real statements, then decide whether `profile.py` needs selection logic (ADR 0002 deliberately has none).
6. **Before exposing the API beyond this machine:** decide on authentication. The label endpoints write data with no auth.

## GitHub issues (triaged 2026-11-11)

Closed as fixed or obsolete after the rebuild: #6, #7, #8, #9, #11. The trips feature was removed as out of scope, so the `/trips` findings (#3.1, #4.2) no longer apply.

Still open, ranked:

| Rank | Issue | Problem | Suggested fix |
|---|---|---|---|
| 1 | #2.2 (#4.7) | `/upload` uses the raw filename: path traversal, and same-name files overwrite | Keep the basename only, check the path stays in `UPLOAD_DIR`, use a unique stored name |
| 2 | #12 | Statements are unique on period alone, so a second card with the same period collides | Add an account identifier or file hash to the key (architectural: needs an ADR) |
| 3 | #13 | `quant_gap_eval` compares 1B base against 3B instruct; the README claims rest on it | Fix the eval or retract the README claim |
| 4 | #14 | Fresh clone: no `.env.example`; requirements list the unused `ollama` and pin no versions | Add `.env.example`, clean up requirements, add `requirements-experiments.txt` |
| 5 | #4.3 | `/statements/{id}/transactions` gives the same 404 for a missing statement and an empty one | Check that the statement exists first |
| 6 | #4.8 | `is_credit` is a String column | Boolean (schema change: ADR) |
| 7 | #5.1, #5.2 | Embedder has no lock and encodes one row at a time | Add a lock and batch the encoding |
| 8 | #15 | README embedding table contradicts the shipped (documented) format | Fix the README row, then close |

All of these are low risk while the API is loopback-only. Fix 1 before any wider exposure.

## Known limits
- One bank and one statement layout are verified (Wells Fargo credit card). Fee rows have not been seen on a real statement.
- The eval questions are synthetic. Real phrasing may score differently, and refused real questions are logged locally for review.
- A merchant whose name is a word in a category name (e.g. a category containing "DoorDash") cannot be reached through the category fallback. Such a question is refused loudly; ask about the category instead.
