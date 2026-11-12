---
paths:
  - "app/query_router.py"
  - "app/question_form.py"
  - "app/read_queries.py"
  - "app/calculations.py"
  - "app/answers.py"
  - "app/read_config.py"
---

# Read path (ADR 0004)

- The model fills a schema-constrained form. It never sees transaction data and never writes SQL.
- Code resolves dates, merchants and defaults; SQL is thin, parameterised and date-filtered.
- Every number is computed in `calculations.py` over `Decimal`. Code writes every fact; every answer shows an "Understood as" line.
- The optional rewrite may only arrange fact tokens; a rewrite that adds digits or changes a fact falls back to the exact answer.
- A misread or unsupported question is refused, never guessed.
- `TEACH_EXAMPLES` must never overlap the form eval questions (see `evals/CLAUDE.md`).
