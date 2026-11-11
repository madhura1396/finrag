"""Shared pytest setup. Runs before any test module imports `app`.

Unit tests must never touch Ollama, Postgres, `.env` or `uploads/`.
"""

import os

import dotenv

# Manual scripts, not pytest tests: test_extractor.py's functions take a
# `filepath` argument and need a real PDF, and test_llm.py calls live Ollama.
# Listing them here keeps them out of `pytest tests/` as well as bare `pytest`.
collect_ignore = ["test_extractor.py", "test_llm.py"]

# app/models.py calls load_dotenv() and create_engine(DATABASE_URL) at import.
# Neutralise load_dotenv before that import so tests never read .env, and give
# the engine a URL that is never connected to (create_engine is lazy).
dotenv.load_dotenv = lambda *args, **kwargs: False
os.environ["DATABASE_URL"] = "postgresql://unit-test:unit-test@127.0.0.1:1/unit-test"
