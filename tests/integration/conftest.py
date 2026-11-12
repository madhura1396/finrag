"""Integration tests against the throwaway finrag_test database (user-approved;
ADR 0004 and ADR 0005).

Safety rails, all before any connection:
- runs only with FINRAG_TEST_DATABASE_URL set, otherwise every test is skipped;
- refuses unless that URL's database is exactly `finrag_test`;
- before dropping anything, reads the database's marker: only `test`, or
  unmarked and empty, may be dropped;
- its engine carries the environment guard, and FINRAG_ENV=test (set in
  tests/conftest.py), so a database marked `production` refuses on connect;
- every test runs inside a transaction that is rolled back afterwards.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

TEST_DB_NAME = "finrag_test"


def _test_url():
    raw = os.getenv("FINRAG_TEST_DATABASE_URL")
    if not raw:
        pytest.skip("FINRAG_TEST_DATABASE_URL is not set")
    url = make_url(raw)
    if url.database != TEST_DB_NAME:
        pytest.fail(f"FINRAG_TEST_DATABASE_URL must point at {TEST_DB_NAME!r}, not {url.database!r}")
    return url


@pytest.fixture(scope="session")
def test_url():
    return _test_url()


@pytest.fixture(scope="session")
def engine(test_url):
    from app.db_guard import install_guard
    from app.db_setup import create_tables, drop_tables

    # Build the schema fresh with an unguarded engine (setup writes the marker),
    # then hand tests a guarded one.
    setup_engine = create_engine(test_url)
    # Before anything destructive, read the database's own marker: drop only a
    # database marked `test`, or an unmarked one holding no user data.
    from scripts.reset_db import inspect_target
    marker, rows = inspect_target(setup_engine)
    if marker not in (None, "test") or (marker is None and rows):
        setup_engine.dispose()
        pytest.fail(f"{TEST_DB_NAME} is marked {marker!r} with {rows} rows of user data; refusing to drop it")
    drop_tables(setup_engine)
    create_tables(setup_engine, "test")
    setup_engine.dispose()

    guarded = install_guard(create_engine(test_url))
    yield guarded
    guarded.dispose()


@pytest.fixture
def db(engine):
    """A session whose every commit is a savepoint inside one outer transaction
    that is rolled back after the test. Code under test gets sessions from the
    same connection through app.models."""
    from app.models import set_session_factory

    connection = engine.connect()
    outer = connection.begin()
    factory = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")
    set_session_factory(factory)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()
