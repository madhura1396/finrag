"""Create and seed the schema on an explicit engine (ADR 0005).

Every function takes the engine or session it acts on; nothing here uses the
import-time engine in app/models.py. These engines are unguarded on purpose:
this module is what writes the environment marker the guard checks.

It also holds the category seed data. Nothing reads that at runtime except
seed(), which runs only on an empty database. System categories are identified
by a stable `system_key`, not a fixed numeric ID, so seeding never fights the
ID sequence.
"""

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db_guard import ENVIRONMENTS
from app.models import Base, CategoryId, CategoryVersion, DataVersion, DbEnvironment

# (system_key, name). These cannot be renamed, merged, deleted, targeted by
# rules or pins, or offered to the model.
SYSTEM_CATEGORIES = [
    ("fees",     "Fees"),       # fixed by statement section (ADR 0003, 2b)
    ("interest", "Interest"),   # fixed by statement section (ADR 0003, 2b)
    ("other",    "Other"),      # the model's none_fit fallback
]

# User-owned defaults. After seeding they belong to the user: renamed, merged
# or archived defaults are never re-created.
DEFAULT_CATEGORIES = [
    "Groceries",
    "Dining & DoorDash",
    "Travel & Transport",
    "Shopping",
    "Telephone & Internet",
    "Entertainment",
    "Health & Fitness",
]


def create_tables(engine: Engine, environment: str) -> None:
    """Create every table, mark the database's environment, seed categories.
    Safe on an existing database: it only adds what is missing."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}, got {environment!r}")

    with engine.begin() as conn:
        # pgvector must exist before Vector(384) columns are created.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

    with Session(bind=engine) as session:
        marker = session.get(DbEnvironment, 1)
        if marker is None:
            session.add(DbEnvironment(id=1, environment=environment))
        elif marker.environment != environment:
            raise ValueError(
                f"database is already marked {marker.environment!r}; refusing to re-mark it {environment!r}"
            )
        seed(session)
        session.commit()


def seed(session: Session) -> None:
    """Seed categories and the data_version row, only into an empty dimension.

    Keyed on "category_ids is empty", never on names, so a default the user
    renamed, merged or archived is never re-created.
    """
    if session.get(DataVersion, 1) is None:
        session.add(DataVersion(id=1, version=0))

    if session.scalar(select(CategoryId.category_id).limit(1)) is not None:
        return

    for system_key, name in SYSTEM_CATEGORIES:
        _add_category(session, name, is_system=True, system_key=system_key)
    for name in DEFAULT_CATEGORIES:
        _add_category(session, name, is_system=False, system_key=None)


def _add_category(session: Session, name: str, is_system: bool, system_key: str | None) -> None:
    identity = CategoryId(is_system=is_system, system_key=system_key)
    session.add(identity)
    session.flush()
    session.add(CategoryVersion(category_id=identity.category_id, name=name))


def drop_tables(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)
