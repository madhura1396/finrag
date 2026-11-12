# Tests

- Unit tests: `pip install -r requirements-dev.txt`, then `python -m pytest` from the repo root. They live in `tests/unit/` and need no Ollama, Postgres or `.env` (`tests/conftest.py` disables `load_dotenv` and sets a dummy `DATABASE_URL`).
- Integration tests (user-approved exception to the no-Postgres rule): `FINRAG_TEST_DATABASE_URL=<finrag_test URL> python -m pytest -m integration`. They live in `tests/integration/`, refuse any database not named `finrag_test`, connect with the environment guard (`FINRAG_ENV=test`), and roll back each test's transaction.
- A manual script remains in `tests/` and is excluded from pytest: `python -m tests.test_llm`. It needs live Ollama.
- New tests must not call Ollama or Postgres unless I ask for an integration test. Mock the HTTP calls.
- Never commit real statements as test fixtures. Use synthetic statements.
