"""The read path's only SQL (ADR 0004, "Testing the numbers"; ADR 0005 removed
the legacy-row check). Thin and parameterised: rows are filtered by date
here, and by direction, category and merchant in app/calculations.py. The
one exception is search, which applies date and direction before its cap.
Every function takes the session it reads with.
"""

from datetime import date

from sqlalchemy import and_, not_, select
from sqlalchemy.orm import Session

from app import category_store
from app.calculations import RowView, direction_params
from app.models import Statement, Transaction, Trip as TripModel
from app.question_form import Trip

_PRESENT = object()     # stands in for an embedding: only its presence matters downstream


# Every column the read path needs, and only whether an embedding exists: the
# 384-float vectors themselves are never loaded for calculation.
_COLUMNS = (
    Transaction.id, Transaction.trans_date, Transaction.amount, Transaction.section, Transaction.category_id,
    Transaction.category_source, Transaction.merchant, Transaction.raw_description, Transaction.needs_review,
    Transaction.embedded_text, Transaction.embedding.is_not(None).label("has_embedding"),
)


def _view(row, names: dict[int, str]) -> RowView:
    return RowView(
        id=row.id, date=row.trans_date, amount=row.amount, section=row.section,
        category=names.get(row.category_id) if row.category_id is not None else None,
        category_source=row.category_source, merchant=row.merchant, raw_description=row.raw_description,
        needs_review=row.needs_review, embedded_text=row.embedded_text,
        embedding=_PRESENT if row.has_embedding else None,
    )


def fetch_rows(session: Session, start: date | None = None, end: date | None = None) -> list[RowView]:
    """1. Every row in the date range (all rows when start is None)."""
    query = select(*_COLUMNS).order_by(Transaction.trans_date, Transaction.id)
    if start is not None:
        query = query.where(Transaction.trans_date >= start, Transaction.trans_date <= end)
    names = category_store.names_by_id(session)
    return [_view(row, names) for row in session.execute(query).all()]


def statement_periods(session: Session) -> list[tuple[date, date]]:
    """2. Statement periods, for coverage."""
    return [(s, e) for s, e in session.execute(
        select(Statement.period_start, Statement.period_end).order_by(Statement.period_start)).all()]


def merchant_names(session: Session) -> list[str]:
    """3. Distinct stored merchant names, for matching."""
    return list(session.scalars(select(Transaction.merchant).distinct().order_by(Transaction.merchant)).all())


def trips(session: Session) -> list[Trip]:
    """4. Trips, for matching."""
    return [Trip(t.name, t.start_date, t.end_date)
            for t in session.scalars(select(TripModel).order_by(TripModel.start_date)).all()]


def search(session: Session, vector: list[float], direction: str, start: date | None, end: date | None,
           cutoff: float, cap: int, category: str | None = None, merchants: tuple[str, ...] = ()) -> list[RowView]:
    """5. Nearest rows by cosine distance, within the cutoff, closest first.
    Date, direction, category and merchants are applied before the cap so the
    cap counts only matches that can be part of the answer."""
    params = direction_params(direction)
    distance = Transaction.embedding.cosine_distance(vector)
    conditions = [Transaction.embedding.is_not(None), distance < cutoff]
    conditions.append(Transaction.amount < 0 if params["sign"] == "negative" else Transaction.amount > 0)
    if params["only_sections"]:
        conditions.append(Transaction.section.in_(params["only_sections"]))
    if params["except_sections"]:
        conditions.append(not_(Transaction.section.in_(params["except_sections"])))
    if start is not None:
        conditions.append(and_(Transaction.trans_date >= start, Transaction.trans_date <= end))
    names = category_store.names_by_id(session)
    if category is not None:
        ids = [cid for cid, name in names.items() if name == category]
        conditions.append(Transaction.category_id.in_(ids))
    if merchants:
        conditions.append(Transaction.merchant.in_(list(merchants)))
    rows = session.execute(select(*_COLUMNS).where(*conditions).order_by(distance, Transaction.id).limit(cap)).all()
    return [_view(row, names) for row in rows]
