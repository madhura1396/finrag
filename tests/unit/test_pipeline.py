"""persist_statement and the duplicate guard, with the database mocked.

The label helpers (version lock, history, category lookups) are stubbed here;
they are exercised against a real database in tests/integration.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app import categorization, category_store, pipeline
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
        next_tx_id = 100
        for obj in added:
            if isinstance(obj, Statement):
                obj.id = 42
            elif isinstance(obj, Transaction) and obj.id is None:
                next_tx_id += 1
                obj.id = next_tx_id

    session.flush.side_effect = flush
    session.added = added
    return session


SECTION_IDS = {"fees": 101, "interest": 102}
NAMES = {101: "Fees", 102: "Interest", 103: "Other", 104: "Groceries"}


@pytest.fixture(autouse=True)
def stub_label_helpers(monkeypatch):
    calls = {"lock": 0, "bump": 0, "history": []}
    monkeypatch.setattr(categorization, "lock_version", lambda db: calls.__setitem__("lock", calls["lock"] + 1))
    monkeypatch.setattr(categorization, "bump_version", lambda db: calls.__setitem__("bump", calls["bump"] + 1))
    monkeypatch.setattr(categorization, "open_history", lambda db, tx, change_id: calls["history"].append(tx.id))
    monkeypatch.setattr(category_store, "names_by_id", lambda db: NAMES)
    monkeypatch.setattr(category_store, "system_category_id", lambda db, key: SECTION_IDS[key])
    return calls


@pytest.fixture
def session(monkeypatch):
    s = fake_session()
    monkeypatch.setattr(pipeline, "new_session", lambda: s)
    return s


def test_rejected_statement_never_opens_a_session(monkeypatch):
    opened = MagicMock()
    monkeypatch.setattr(pipeline, "new_session", opened)
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


def test_rows_are_stored_signed_and_labelled_by_section(session):
    pipeline.persist_statement(valid_statement(), "s.pdf")
    txs = [o for o in session.added if isinstance(o, Transaction)]
    payment = next(tx for tx in txs if tx.section == "payments")
    zero_interest = next(tx for tx in txs if tx.section == "interest" and tx.amount == 0)
    assert (payment.amount > 0, payment.is_credit, zero_interest.is_credit) == (True, "true", "false")
    for tx in txs:
        if tx.section in SECTION_IDS:       # fees and interest: fixed category at insert
            assert (tx.category_id, tx.category_source) == (SECTION_IDS[tx.section], "section")
            assert (tx.merchant, tx.merchant_source) == (tx.raw_description, "section")
        elif tx.section == "purchases":     # everything else waits for labelling
            assert (tx.amount < 0, tx.is_credit) == (True, "false")
            assert (tx.category_id, tx.category_source, tx.merchant_source) == (None, "pending", "pending")


def test_upload_locks_and_bumps_data_version_and_opens_history(session, stub_label_helpers):
    data = valid_statement()
    pipeline.persist_statement(data, "s.pdf")
    assert stub_label_helpers["lock"] == 1 and stub_label_helpers["bump"] == 1
    assert len(stub_label_helpers["history"]) == len(data.rows)


def test_existing_period_raises_duplicate_and_adds_nothing(monkeypatch):
    s = fake_session(existing=object())
    monkeypatch.setattr(pipeline, "new_session", lambda: s)
    with pytest.raises(DuplicateStatement):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    assert s.added == []
    s.commit.assert_not_called()


@pytest.mark.parametrize("at, constraint, raised", [
    ("commit", "uq_statement_period", DuplicateStatement),
    ("flush", "uq_statement_period", DuplicateStatement),     # psycopg2 raises at flush, before commit
    ("commit", "some_other_constraint", IntegrityError),      # never disguised as a duplicate
])
def test_unique_constraint_race(session, at, constraint, raised):
    orig = SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint))
    getattr(session, at).side_effect = IntegrityError("INSERT", {}, orig)
    with pytest.raises(raised):
        pipeline.persist_statement(valid_statement(), "s.pdf")
    session.rollback.assert_called_once()


def test_extract_reads_with_the_single_profile_then_persists(monkeypatch):
    calls = {}
    monkeypatch.setattr(pipeline, "load_single", lambda: "PROFILE")
    monkeypatch.setattr(pipeline, "read_statement", lambda path, profile: calls.setdefault("read", (path, profile)))
    monkeypatch.setattr(pipeline, "persist_statement", lambda data, name: {"persisted": (data, name)})
    out = pipeline.extract_from_pdf("s.pdf", "s.pdf")
    assert calls["read"] == ("s.pdf", "PROFILE") and out == {"persisted": (("s.pdf", "PROFILE"), "s.pdf")}


def test_statement_id_is_read_before_commit_expires_it(session):
    def expire_on_commit():
        for obj in session.added:
            if isinstance(obj, Statement):
                obj.id = None   # stands in for an expired attribute that would reload
    session.commit.side_effect = expire_on_commit
    assert pipeline.persist_statement(valid_statement(), "s.pdf")["statement_id"] == 42
