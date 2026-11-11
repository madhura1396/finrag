"""persist_statement and the duplicate guard, with the database mocked."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app import pipeline
from app.models import Statement, Transaction
from app.statement import DuplicateStatement, StatementRejected
from tests.unit.factories import valid_statement, with_summary


def fake_session(existing=None):
    """A mocked session that records add() calls and gives the statement an id on flush."""
    session = MagicMock()
    session.query.return_value.filter_by.return_value.first.return_value = existing
    added = []
    session.add.side_effect = added.append

    def flush():
        for obj in added:
            if isinstance(obj, Statement):
                obj.id = 42

    session.flush.side_effect = flush
    session.added = added
    return session


@pytest.fixture
def session(monkeypatch):
    s = fake_session()
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: s)
    return s


def test_rejected_statement_never_opens_a_session(monkeypatch):
    opened = MagicMock()
    monkeypatch.setattr(pipeline, "SessionLocal", opened)
    with pytest.raises(StatementRejected) as exc:
        pipeline.persist_statement(with_summary(valid_statement(), new_balance="0.00"), "s.pdf")
    assert [m.check for m in exc.value.mismatches] == ["summary_equation"]
    opened.assert_not_called()


def test_valid_statement_is_stored_in_one_transaction(session):
    data = valid_statement()
    result = pipeline.persist_statement(data, "s.pdf")

    statements = [o for o in session.added if isinstance(o, Statement)]
    txs = [o for o in session.added if isinstance(o, Transaction)]
    assert len(statements) == 1 and statements[0].filename == "s.pdf"
    assert len(txs) == len(data.rows)
    assert all(tx.statement_id == 42 for tx in txs)
    session.commit.assert_called_once()
    session.close.assert_called_once()

    assert result == {
        "statement_id":       42,
        "period_start":       data.period_start,
        "period_end":         data.period_end,
        "total_transactions": len(data.rows),
    }


def test_amounts_are_stored_signed_and_is_credit_follows_the_sign(session):
    pipeline.persist_statement(valid_statement(), "s.pdf")
    txs = [o for o in session.added if isinstance(o, Transaction)]
    payment = next(tx for tx in txs if tx.section == "payments")
    purchase = next(tx for tx in txs if tx.section == "purchases")
    zero_interest = next(tx for tx in txs if tx.section == "interest" and tx.amount == 0)
    assert (payment.amount > 0, payment.is_credit) == (True, "true")
    assert (purchase.amount < 0, purchase.is_credit) == (True, "false")
    assert zero_interest.is_credit == "false"


def test_fee_and_interest_rows_get_fixed_category_at_insert(session):
    pipeline.persist_statement(valid_statement(), "s.pdf")
    txs = [o for o in session.added if isinstance(o, Transaction)]

    for section, category in (("fees", "Fees"), ("interest", "Interest")):
        rows = [tx for tx in txs if tx.section == section]
        assert rows and all(tx.category == category for tx in rows)
        assert all(tx.merchant == tx.raw_description for tx in rows)

    # Other rows are left for LLM enrichment: never given a fixed category.
    assert all(tx.category not in ("Fees", "Interest") for tx in txs if tx.section == "purchases")


def test_existing_period_raises_duplicate_and_adds_nothing(monkeypatch):
    s = fake_session(existing=object())
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: s)
    with pytest.raises(DuplicateStatement):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    assert s.added == []
    s.commit.assert_not_called()


def test_unique_constraint_race_is_reported_as_duplicate(session):
    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name="uq_statement_period"))
    session.commit.side_effect = IntegrityError("INSERT", {}, orig)
    with pytest.raises(DuplicateStatement):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    session.rollback.assert_called_once()


def test_other_integrity_errors_are_not_disguised(session):
    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name="some_other_constraint"))
    session.commit.side_effect = IntegrityError("INSERT", {}, orig)
    with pytest.raises(IntegrityError):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    session.rollback.assert_called_once()


def test_legacy_path_raises_duplicate_not_value_error(monkeypatch):
    s = fake_session(existing=object())
    monkeypatch.setattr(pipeline, "SessionLocal", lambda: s)
    monkeypatch.setattr(pipeline, "extract_raw_text", lambda path: [])
    monkeypatch.setattr(
        pipeline, "parse_statement_period", lambda pages: (date(2025, 12, 3), date(2026, 1, 2))
    )
    with pytest.raises(DuplicateStatement):
        pipeline.extract_from_pdf("unused.pdf", "s.pdf")


def test_unique_constraint_at_flush_is_reported_as_duplicate(session):
    # psycopg2 raises the unique violation when the INSERT is flushed, before commit.
    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name="uq_statement_period"))
    session.flush.side_effect = IntegrityError("INSERT", {}, orig)
    with pytest.raises(DuplicateStatement):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    session.rollback.assert_called_once()
    session.commit.assert_not_called()


def test_statement_id_is_read_before_commit_expires_it(session):
    def expire_on_commit():
        for obj in session.added:
            if isinstance(obj, Statement):
                obj.id = None   # stands in for an expired attribute that would reload
    session.commit.side_effect = expire_on_commit
    assert pipeline.persist_statement(valid_statement(), "s.pdf")["statement_id"] == 42
