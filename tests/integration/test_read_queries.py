"""The read path's five queries against a real Postgres (finrag_test). Invented data."""

from datetime import date
from decimal import Decimal

import pytest

from app import category_store, read_queries
from app.models import Statement, Transaction

pytestmark = pytest.mark.integration


def unit(i: int) -> list[float]:
    v = [0.0] * 384
    v[i] = 1.0
    return v


@pytest.fixture
def data(db):
    groceries = next(c.category_id for c in category_store.current_categories(db) if c.name == "Groceries")
    s1 = Statement(filename="a.pdf", period_start=date(2026, 10, 1), period_end=date(2026, 10, 31))
    s2 = Statement(filename="b.pdf", period_start=date(2026, 11, 1), period_end=date(2026, 11, 30))
    db.add_all([s1, s2])
    db.flush()

    def tx(statement, d, amount, merchant, section="purchases", category=groceries, vector=None):
        row = Transaction(statement_id=statement.id, trans_date=d, raw_description=merchant.upper(),
                          amount=Decimal(amount), section=section, merchant=merchant,
                          category_id=category, category_source="model" if category else "pending",
                          merchant_source="model", embedding=vector)
        db.add(row)
        return row

    tx(s1, date(2026, 10, 1), "-10.00", "Grocer Mart", vector=unit(0))          # boundary: first day
    tx(s1, date(2026, 10, 31), "-20.00", "Grocer Mart", vector=unit(0))         # boundary: last day
    tx(s1, date(2026, 10, 15), "5.00", "Grocer Mart", section="other_credits", vector=unit(0))
    tx(s1, date(2026, 10, 16), "50.00", "Payment", section="payments", vector=unit(0))
    tx(s2, date(2026, 11, 2), "-30.00", "Taco Town", category=None, vector=unit(1))
    tx(s2, date(2026, 11, 3), "-40.00", "Taco Town", vector=None)
    db.flush()
    return db


def test_fetch_rows_respects_inclusive_date_bounds(data):
    rows = read_queries.fetch_rows(data, date(2026, 10, 1), date(2026, 10, 31))
    assert [r.amount for r in rows] == [Decimal("-10.00"), Decimal("5.00"), Decimal("50.00"), Decimal("-20.00")]
    assert len(read_queries.fetch_rows(data)) == 6


def test_fetch_rows_joins_current_names_and_marks_pending(data):
    taco_rows = read_queries.fetch_rows(data, date(2026, 11, 1), date(2026, 11, 30))
    assert {r.category for r in taco_rows} == {None, "Groceries"}
    assert {r.embedding is None for r in taco_rows} == {True, False}


def test_statement_periods_and_merchants(data):
    assert read_queries.statement_periods(data) == [(date(2026, 10, 1), date(2026, 10, 31)),
                                                    (date(2026, 11, 1), date(2026, 11, 30))]
    assert read_queries.merchant_names(data) == ["Grocer Mart", "Payment", "Taco Town"]


@pytest.mark.parametrize("direction, amounts", [
    ("spent", {Decimal("-10.00"), Decimal("-20.00")}),     # credits and far rows excluded
    ("refunds", {Decimal("5.00")}),
    ("payments", {Decimal("50.00")}),
])
def test_search_applies_cutoff_and_direction(data, direction, amounts):
    found = read_queries.search(data, unit(0), direction, None, None, cutoff=0.5, cap=50)
    assert {r.amount for r in found} == amounts


def test_search_applies_dates_and_cap_before_returning(data):
    capped = read_queries.search(data, unit(0), "spent", None, None, cutoff=0.5, cap=1)
    assert len(capped) == 1
    dated = read_queries.search(data, unit(0), "spent", date(2026, 10, 31), date(2026, 10, 31), cutoff=0.5, cap=50)
    assert [r.amount for r in dated] == [Decimal("-20.00")]


def test_search_never_returns_rows_without_embeddings(data):
    rows = read_queries.search(data, unit(1), "spent", None, None, cutoff=2.0, cap=50)
    assert all(r.embedding is not None for r in rows)


def test_search_filters_category_and_merchant_before_the_cap(data):
    # Taco Town's pending row (vector 1) is excluded by a category filter; the cap of 1
    # then applies only to rows that can count.
    groceries = read_queries.search(data, unit(0), "spent", None, None, cutoff=0.5, cap=1, category="Groceries")
    assert len(groceries) == 1 and groceries[0].category == "Groceries"
    by_merchant = read_queries.search(data, unit(1), "spent", None, None, cutoff=2.0, cap=50, merchants=("Taco Town",))
    assert {r.merchant for r in by_merchant} == {"Taco Town"}
