# scripts/

## reset_db.py (guarded schema reset)

- Every database carries an environment marker (`db_environment`): `finrag` is `production`, `finrag_test` is `test`. `reset_db` never re-marks a database.
- Fresh database: `python -m scripts.reset_db --env production --yes --confirm-name finrag` (create, mark production, seed). A database that holds data or is marked production also needs `--allow-production`, which destroys it.
- Running `reset_db` against `finrag` needs my explicit ask. Tests may target `finrag_test`.

## check_profiles.py (pass/fail on real statements)

- Prints only `PASS <name>` or `FAIL <name> <check>[:<section>]` per file: fixed check names from `app.statement.CHECKS` and section keys, never amounts, dates, merchants or other statement text (ADR 0003, decision 7).
- Imports only `app.profile`, `app.reader` and `app.statement`, never `app.models`, so it never loads `.env` or opens a database.
- I run it on my statements; you never read its input files.
