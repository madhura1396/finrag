# FinRAG

Personal finance assistant I use for real. PDF bank statements in, reconciled transactions to Postgres + pgvector, plain-English questions answered by a model-filled form, with every number computed by code (ADR 0004). Ollama (llama3.2) and all-MiniLM-L6-v2 run locally.

I am also using this project to learn senior engineering, so every design choice must be explained and defensible.

## Scope

- Today: Wells Fargo credit card only (ADR 0002, Decision 1). Target: Wells Fargo, Chase, Discover, Fidelity. PDF statements only.
- Trips removed as out of scope (2026-11-11).

## Hard constraints

Privacy, `Decimal` money and fail-loudly are in `.claude/rules/` (`privacy.md`, `money-decimal.md`, `fail-loudly.md`). They always apply, everywhere in the repo, including `experiments/`.

## Code ownership

- Claude writes the code in app/ and tests/. I decide architecture through the decision protocol and review every PR.
- When a change is ready, run senior-reviewer on the diff and fix every blocker, then ask me to confirm the commit.

## Architectural decisions: evaluate with me first

An architectural decision is any of:
- a new dependency, model or external service
- a schema change or migration
- a new module, or a change to the boundary between modules
- a contract: API request or response shape, a function signature used across modules, a file or data format
- the concurrency, transaction, error handling or security model
- anything expensive to reverse

Before making one, follow the `write-adr` skill: options, recommendation, then stop and wait for my choice. Write no code for it until I choose.

Not architectural, so just do it inside the agreed task: naming, refactors inside one function, formatting, test cases for behaviour we already agreed.

## Source of truth

- The code is the source of truth. `README.md` may lag; never quote it as fact without checking the code.
- Decisions live in `docs/adr/`.

## Layout

```
app/            production code
profiles/       bank layout profiles, YAML
scripts/        reset_db.py (guarded schema reset); check_profiles.py (pass/fail on your real statements)
evals/          read_path/ form eval (synthetic questions; needs Ollama); search_eval.example.yaml
tests/          unit/ = pytest suite; integration/ = finrag_test suite; top-level test_*.py = manual scripts
experiments/    model-internals research, not part of the app
docs/adr/       architecture decision records
docs/plans/     implementation plans
```

Folder rules load when you work in that folder: `app/`, `profiles/`, `scripts/`, `evals/`, `tests/`, `experiments/` and `docs/adr/` each have a `CLAUDE.md`. File-level rules for `app/` live in `.claude/rules/`: `write-path.md`, `read-path.md`, `database.md`.

## Git

- Never commit to `main`. One branch and one PR per change: `fix/<name>`, `feat/<name>`, `chore/<name>`, `adr/<name>`.
- Commit messages: imperative subject under 72 characters, then a body that says why.
- Before every commit, show me git status, the files to be staged and the commit message, and wait for me to confirm. After I confirm, commit and push the branch with git push -u origin <branch>. Never push to main, never force push, never rebase or delete branches.
- Run `git status` before and after any change and tell me what changed.
- Stage files by name. Never use git add -A or git add .

## Data safety

- Never read `.env` or anything in `uploads/`. They hold credentials and real bank statements.
- Never read the refused-question log (`FINRAG_REFUSED_LOG`, default `~/.finrag/refused_questions.log`) or `~/.finrag/search_eval.yaml`. They hold my questions and real merchant names.
- Never run SQL that writes to the database (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER) unless I ask. Exception: the integration suite may write to `finrag_test` (never `finrag`), and `scripts/reset_db.py` may target `finrag_test` in tests. Running `reset_db` against `finrag` needs my explicit ask.
- Rules, label memory, pins and the change log hold real merchant text and live only in the database. Never copy them into the repo, fixtures or docs.
- Never paste real transaction data into commit messages, PRs, ADRs or docs.
- To check profiles against my real statements, write a script that prints only pass/fail per file, plus on failure the failing check names and section keys (fixed words from code and the profile, never amounts, dates, merchants or other statement text; ADR 0003). I run it; you never read its input files.

## Working style

- One task per session. Before starting, restate: goal, files in scope, out of scope, how we know it is done.
- Use a subagent for broad searches ("where is X called") so only the answer comes back.
- When unsure, ask one question rather than guessing.

## Review

After any change to files in app/ or tests/, run the senior-reviewer subagent on the diff before committing. Report its verdict and top issues to me.
