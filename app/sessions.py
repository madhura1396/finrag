"""Where write-path code gets its database session (ADR 0005 fixes, option B).

One process-wide factory, set once by the app and swapped only by tests.
Its safety comes from the environment guard on every engine
(app/db_guard.py), not from discipline: a test that forgets to swap the
factory hits a `production` marker and fails on connect.
"""

from collections.abc import Callable, Iterator

from sqlalchemy.orm import Session

from app.models import SessionLocal

_factory: Callable[[], Session] = SessionLocal


def new_session() -> Session:
    return _factory()


def set_session_factory(factory: Callable[[], Session]) -> None:
    global _factory
    _factory = factory


def reset_session_factory() -> None:
    set_session_factory(SessionLocal)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request. Tests may override it."""
    session = new_session()
    try:
        yield session
    finally:
        session.close()
