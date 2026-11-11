"""Refuse to talk to a database whose environment marker does not match ours.

Each database records what it is in a one-row `db_environment` table
(`production` for finrag, `test` for finrag_test), the way Rails keeps its
environment inside the database. Every new connection checks that marker
against FINRAG_ENV (default `production`). A test run pointed at the real
database therefore fails on connect and writes nothing.

Setup code (app.db_setup, scripts/reset_db) uses unguarded engines because it
is what writes the marker.
"""

import os

from sqlalchemy import event
from sqlalchemy.engine import Engine

ENV_VAR = "FINRAG_ENV"
ENVIRONMENTS = ("production", "test")


class EnvironmentMismatch(RuntimeError):
    """Connected to a database marked for a different environment."""


def expected_environment() -> str:
    env = os.getenv(ENV_VAR, "production")
    if env not in ENVIRONMENTS:
        raise EnvironmentMismatch(f"{ENV_VAR}={env!r} is not one of {ENVIRONMENTS}")
    return env


def read_marker(dbapi_connection) -> str | None:
    """The database's own marker, or None if it has none (never set up)."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(
            "SELECT to_regclass('public.db_environment') IS NOT NULL"
        )
        if not cursor.fetchone()[0]:
            return None
        cursor.execute("SELECT environment FROM db_environment WHERE id = 1")
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        # The SELECTs opened a transaction; leave the connection clean.
        dbapi_connection.rollback()


def _check(dbapi_connection, _record) -> None:
    expected = expected_environment()
    marker = read_marker(dbapi_connection)
    if marker is None:
        raise EnvironmentMismatch(
            "database has no environment marker; set it up with "
            "`python -m scripts.reset_db` (see .claude/CLAUDE.md)"
        )
    if marker != expected:
        raise EnvironmentMismatch(
            f"database is marked {marker!r} but {ENV_VAR} expects {expected!r}; refusing to connect"
        )


def install_guard(engine: Engine) -> Engine:
    if not event.contains(engine, "connect", _check):
        event.listen(engine, "connect", _check)
    return engine
