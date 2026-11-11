"""Read-only access to categories and the shared text normaliser (ADR 0005).

The only place that resolves "current" (valid_to IS NULL), so no caller can
read an old version by accident. Imports only app.models; safe for the read
path, the write path and scripts.
"""

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CategoryId, CategoryVersion

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def normalise(text: str) -> str:
    """Upper-case, punctuation removed, spaces collapsed. Used by rules,
    memory keys and merchant matching alike."""
    return _SPACES.sub(" ", _PUNCT.sub(" ", (text or "").upper())).strip()


def description_key(text: str) -> str:
    """The memory key: normalised, digits dropped too, so store numbers do not
    split one store. Falls back to keeping digits, then to the raw text, so a
    key is never empty."""
    normalised = normalise(text)
    without_digits = _SPACES.sub(" ", _DIGITS.sub(" ", normalised)).strip()
    return without_digits or normalised or (text or "").strip().upper() or "(BLANK)"


@dataclass(frozen=True)
class Category:
    category_id: int
    name: str
    archived: bool
    is_system: bool
    system_key: str | None


def current_categories(session: Session) -> list[Category]:
    rows = session.execute(
        select(CategoryVersion.category_id, CategoryVersion.name, CategoryVersion.archived,
               CategoryId.is_system, CategoryId.system_key)
        .join(CategoryId, CategoryId.category_id == CategoryVersion.category_id)
        .where(CategoryVersion.valid_to.is_(None))
        .order_by(CategoryVersion.category_id)
    ).all()
    return [Category(*row) for row in rows]


def names_by_id(session: Session) -> dict[int, str]:
    """Current names for every category, archived included, in one query."""
    return {c.category_id: c.name for c in current_categories(session)}


def enrichment_categories(session: Session) -> list[Category]:
    """What the model may choose from: current, not archived, not system."""
    return [c for c in current_categories(session) if not c.archived and not c.is_system]


def system_category_id(session: Session, system_key: str) -> int:
    category_id = session.scalar(select(CategoryId.category_id).where(CategoryId.system_key == system_key))
    if category_id is None:
        raise LookupError(f"system category {system_key!r} is not seeded")
    return category_id


def name_at(session: Session, category_id: int, when: datetime) -> str | None:
    """The category's name as it was at `when` (point in time)."""
    return session.scalar(
        select(CategoryVersion.name).where(
            CategoryVersion.category_id == category_id,
            CategoryVersion.valid_from <= when,
            (CategoryVersion.valid_to.is_(None)) | (CategoryVersion.valid_to > when),
        )
    )
