"""/query end to end with the database queries and the model mocked (ADR 0004, detail 6)."""

from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app import main, query_router, read_queries
from app import category_store
from app.calculations import RowView
from app.question_form import ModelUnavailable, NotUnderstood, RawForm
from app.sessions import get_session

TODAY = date(2026, 12, 20)
ROWS = [
    RowView(1, date(2026, 11, 3), Decimal("-40.00"), "purchases", "Groceries", "model", "Grocer Mart",
            "GROCER MART #101", False, None, None),
    RowView(2, date(2026, 11, 9), Decimal("-10.00"), "purchases", None, "pending", "Taco Town",
            "TACO TOWN", True, None, None),
]


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(read_queries, "statement_periods", lambda s: [(date(2026, 11, 1), date(2026, 11, 30))])
    monkeypatch.setattr(read_queries, "merchant_names", lambda s: ["Grocer Mart", "Taco Town"])
    monkeypatch.setattr(read_queries, "trips", lambda s: [])
    monkeypatch.setattr(read_queries, "fetch_rows", lambda s, start=None, end=None: ROWS)
    monkeypatch.setattr(category_store, "current_categories",
                        lambda s: [category_store.Category(4, "Groceries", False, False, None)])


def ask_returns(form):
    def ask(question, examples):
        if isinstance(form, Exception):
            raise form
        return form
    return ask


def route(form, call=lambda p: "Here you go: [F1], and [F2].", question="q"):
    return query_router.route(question, session=None, today=TODAY, ask=ask_returns(form), call=call)


def test_answered_response_has_every_field_and_strings_for_money(stubbed):
    out = route(RawForm(type="total_spent", period="2026-11"))
    assert set(out) == {"type", "answer", "answer_exact", "rewrite", "understood_as", "interpretation", "result",
                        "rows", "rows_total", "notes", "query"}
    assert out["type"] == "total_spent" and out["rewrite"] == "accepted"
    assert out["result"]["total"] == "50.00"
    assert out["rows"][0]["amount"] == "-40.00" and out["rows"][0]["date"] == "2026-11-03"
    assert out["notes"] == {"needs_review": 1, "pending": 1}
    assert out["understood_as"].startswith("Understood as: total spent")
    assert out["answer"].startswith("Here you go: you spent $50.00")


def test_rewrite_rejection_shows_the_exact_answer(stubbed):
    out = route(RawForm(type="total_spent"), call=lambda p: "Almost [F1] and [F2].")
    assert (out["rewrite"], out["answer"]) == ("rejected", out["answer_exact"])


def test_unsupported_is_a_refusal_and_is_logged(stubbed, tmp_path, monkeypatch):
    out = route(RawForm(type="unsupported"), question="what's the weather")
    assert out["type"] == "refused" and out["result"]["kind"] == "unsupported"
    log = tmp_path / "refused_questions.log"
    assert log.read_text() == "what's the weather\n"


def test_not_understood_is_refused_and_logged(stubbed, tmp_path):
    out = route(NotUnderstood("garbage"), question="asdf qwer")
    assert out["result"]["kind"] == "not_understood"
    assert (tmp_path / "refused_questions.log").read_text() == "asdf qwer\n"


def test_data_refusals_are_not_logged(stubbed, tmp_path):
    out = route(RawForm(type="total_spent", category="Petrol"))
    assert out["type"] == "refused" and out["result"]["kind"] == "data"
    assert not (tmp_path / "refused_questions.log").exists()


def test_log_write_failure_never_fails_the_answer(stubbed, monkeypatch, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("FINRAG_REFUSED_LOG", str(blocker / "refused.log"))     # parent is a file: cannot write
    assert route(RawForm(type="unsupported"))["type"] == "refused"


def test_table_answer_through_route_is_never_rewritten(stubbed):
    out = route(RawForm(type="ranking", group_by="merchant"), call=lambda p: pytest.fail("no rewrite for tables"))
    assert out["rewrite"] == "skipped" and out["result"]["table"][0]["name"] == "Grocer Mart"


def test_search_through_route_passes_filters_to_sql(stubbed, monkeypatch):
    seen = {}

    def fake_search(session, vector, direction, start, end, cutoff, cap, category=None, merchants=()):
        seen.update(direction=direction, category=category, merchants=merchants)
        return [ROWS[0]]
    monkeypatch.setattr(read_queries, "search", fake_search)
    out = query_router.route("q", session=None, today=TODAY, embed=lambda text: [0.0] * 384,
                             ask=ask_returns(RawForm(type="search", search_text="groceries", category="groceries")),
                             call=lambda p: "Here is what I found: [F1], and [F2].")
    assert seen == {"direction": "spent", "category": "Groceries", "merchants": ()}
    assert out["answer"].endswith("may miss some or include near-misses.")


def test_no_data_period_says_so_with_coverage(stubbed):
    out = route(RawForm(type="total_spent", period="2026-08"))
    assert out["answer"].startswith("No data:") and "1 Nov 2026 to 30 Nov 2026" in out["answer"]


# --- HTTP mapping ---------------------------------------------------------------------

@pytest.fixture
def client():
    main.app.dependency_overrides[get_session] = lambda: None
    yield TestClient(main.app, raise_server_exceptions=False, client=("127.0.0.1", 50000))
    main.app.dependency_overrides.clear()


def test_refusal_is_a_200(client, monkeypatch):
    monkeypatch.setattr(main, "route", lambda q, db: {"type": "refused"})
    assert client.post("/query", json={"question": "x"}).status_code == 200


def test_model_down_is_a_503(client, monkeypatch):
    def down(q, db):
        raise ModelUnavailable("no ollama")
    monkeypatch.setattr(main, "route", down)
    assert client.post("/query", json={"question": "x"}).status_code == 503


def test_empty_question_is_a_400(client):
    assert client.post("/query", json={"question": "  "}).status_code == 400


def test_rows_found_in_an_uncovered_period_are_still_answered(stubbed, monkeypatch):
    # A row dated before its statement starts sits in a period no statement covers.
    monkeypatch.setattr(read_queries, "fetch_rows", lambda s, start=None, end=None: [ROWS[0]])
    monkeypatch.setattr(read_queries, "statement_periods", lambda s: [(date(2026, 11, 5), date(2026, 11, 30))])
    out = route(RawForm(type="total_spent", period="2026-11-01..2026-11-04"))
    assert not out["answer"].startswith("No data") and out["result"]["total"] == "40.00"
    assert "no data for this period" in out["understood_as"]


def test_refund_only_period_is_not_reported_as_no_data(stubbed, monkeypatch):
    refund = RowView(9, date(2026, 11, 2), Decimal("25.00"), "other_credits", "Groceries", "model", "Grocer Mart",
                     "GROCER MART", False, None, None)
    monkeypatch.setattr(read_queries, "fetch_rows", lambda s, start=None, end=None: [refund])
    monkeypatch.setattr(read_queries, "statement_periods", lambda s: [(date(2026, 11, 5), date(2026, 11, 30))])
    out = route(RawForm(type="total_spent", period="2026-11-01..2026-11-04"))
    assert not out["answer"].startswith("No data") and out["result"]["refunds"] == "25.00"
