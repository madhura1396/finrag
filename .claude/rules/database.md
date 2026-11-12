---
paths:
  - "app/models.py"
  - "app/db_guard.py"
  - "app/db_setup.py"
  - "tests/integration/**"
---

# Database environments

- Every database carries an environment marker (`db_environment`): `finrag` is `production`, `finrag_test` is `test`. Connections compare it with `FINRAG_ENV` (default `production`) and refuse on a mismatch or a missing marker.
- Schema resets go through `scripts/reset_db.py` (see `scripts/CLAUDE.md`).
