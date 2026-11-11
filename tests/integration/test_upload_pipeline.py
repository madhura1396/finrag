"""The whole write path on finrag_test: synthetic PDF -> reader -> validate ->
store -> enrich (model mocked). Invented data only."""

from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app import categorization, category_store, pipeline
from app.llm import ModelLabel
from app.models import Statement, Transaction
from app.statement import DuplicateStatement, StatementRejected
from tests.fixtures.make_wells_fargo import make_statement, standard_spec

pytestmark = pytest.mark.integration


def count(db, model):
    return db.scalar(select(func.count()).select_from(model))


def test_a_reconciling_statement_is_stored_with_sections_and_signed_amounts(db, tmp_path):
    path = tmp_path / "s.pdf"
    truth = make_statement(path, standard_spec())
    result = pipeline.extract_from_pdf(str(path), "s.pdf")
    assert result["total_transactions"] == len(truth)

    rows = db.scalars(select(Transaction).order_by(Transaction.id)).all()
    assert sorted((r.raw_description, r.amount, r.section) for r in rows) == \
        sorted((d, a, s) for _, d, a, s in truth)
    names = category_store.names_by_id(db)
    interest = [r for r in rows if r.section == "interest"]
    assert {names[r.category_id] for r in interest} == {"Interest"} and all(r.category_source == "section" for r in interest)
    assert all(r.category_source == "pending" for r in rows if r.section == "purchases")


def test_fee_and_interest_rows_never_reach_the_model(db, tmp_path):
    path = tmp_path / "s.pdf"
    make_statement(path, standard_spec())
    pipeline.extract_from_pdf(str(path), "s.pdf")
    db.commit()     # harness: release this session's savepoint before code that rolls back its read
    asked = []

    def labeller(descriptions, names):
        asked.extend(descriptions.values())
        return {k: ModelLabel("Shopping", "Some Shop", None) for k in descriptions}

    categorization.enrich_pending(db, labeller=labeller)
    assert asked and not any("INTEREST" in d for d in asked)


def test_a_broken_statement_is_rejected_and_stores_nothing(db, tmp_path):
    spec = standard_spec()
    spec = replace(spec, totals={"purchases": Decimal("1.00")})
    path = tmp_path / "bad.pdf"
    make_statement(path, spec)
    before = (count(db, Statement), count(db, Transaction))
    with pytest.raises(StatementRejected) as e:
        pipeline.extract_from_pdf(str(path), "bad.pdf")
    assert ("section_total", "purchases") in {(m.check, m.section) for m in e.value.mismatches}
    assert (count(db, Statement), count(db, Transaction)) == before


def test_the_same_statement_twice_is_a_duplicate(db, tmp_path):
    path = tmp_path / "s.pdf"
    make_statement(path, standard_spec())
    pipeline.extract_from_pdf(str(path), "s.pdf")
    with pytest.raises(DuplicateStatement):
        pipeline.extract_from_pdf(str(path), "s-again.pdf")
