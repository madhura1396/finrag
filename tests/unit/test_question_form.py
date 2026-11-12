"""Form validation and resolution (ADR 0004, details 1 and 2). Pure; a fixed today."""

from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from app import question_form as qf
from app.question_form import DataContext, NotUnderstood, RawForm
from app.read_config import Refusal

TODAY = date(2026, 11, 11)
LEAP_DAY = date(2028, 2, 29)


def ctx(**kw):
    base = dict(today=TODAY, categories=["Groceries", "Dining & DoorDash", "Shopping"],
                merchants=["Costco", "Costco Wholesale", "Target", "Trader Joe's", "GROCER MART S #101"],
                full_months=[(2026, 8), (2026, 9), (2026, 10)])
    return DataContext(**{**base, **kw})


def resolve(**kw):
    return qf.resolve(RawForm(**kw), ctx())


def span(r):
    return r.start, r.end


def model_reply(text):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": text}
    return MagicMock(return_value=response)


# --- raw validation and the model call ------------------------------------------

@pytest.mark.parametrize("data", [
    None, {"type": "drop_tables"}, {"type": "count", "limit": 51}, {"type": "count", "limit": True},
    {"type": "count", "direction": "sideways"}, {"type": "count", "category": 5},
])
def test_out_of_shape_output_is_not_understood(data):
    with pytest.raises(NotUnderstood):
        qf.parse_raw(data)


@pytest.mark.parametrize("value", ["null", " None ", "n/a", ""])
def test_spelled_out_null_is_absent(value):
    raw = qf.parse_raw({"type": "ranking", "order": "least", "limit": 3, "merchant": value, "period": value})
    assert (raw.order, raw.limit, raw.merchant, raw.period) == ("least", 3, None, None)


def test_ask_sends_schema_and_temperature_zero():
    post = model_reply('{"type": "count"}')
    assert qf.ask_model("how many", [], post=post).type == "count"
    sent = post.call_args.kwargs["json"]
    assert sent["options"]["temperature"] == 0 and sent["format"]["properties"]["type"]["enum"][-1] == "unsupported"


def test_model_unreachable_is_unavailable_and_garbage_is_not_understood():
    with pytest.raises(qf.ModelUnavailable):
        qf.ask_model("q", [], post=MagicMock(side_effect=httpx.ConnectError("down")))
    with pytest.raises(NotUnderstood):
        qf.ask_model("q", [], post=model_reply("not json"))


# --- categories and merchants --------------------------------------------------

def test_category_matches_ignoring_case_and_unknown_is_refused_with_the_list():
    assert resolve(type="total_spent", category="groceries").category == "Groceries"
    out = resolve(type="total_spent", category="Petrol")
    assert isinstance(out, Refusal) and "Groceries" in out.detail["categories"]


@pytest.mark.parametrize("typed, merchants", [
    ("costco", ("Costco", "Costco Wholesale")),       # containment finds longer names
    ("trader joes", ("Trader Joe's",)),               # punctuation is normalised
])
def test_merchant_matching(typed, merchants):
    assert resolve(type="total_spent", merchant=typed).merchants == merchants


def test_similarity_catches_typos():
    assert "Costco" in resolve(type="total_spent", merchant="costcoo").merchants


@pytest.mark.parametrize("typed", ["at", "grocer marts"])   # too short to contain; known plural limit, pinned
def test_unmatched_merchant_is_refused_with_closest_names(typed):
    out = resolve(type="total_spent", merchant=typed)
    assert isinstance(out, Refusal)
    if typed == "grocer marts":
        assert "GROCER MART S #101" in out.detail["closest"]


def fallback(category, merchants):
    return qf.resolve(RawForm(type="total_spent", category=category), ctx(merchants=merchants))


def test_unknown_category_naming_a_merchant_by_whole_word_becomes_the_merchant():
    form = fallback("uber", ["UBER RIDES", "TUBERS CAFE", "Taco Town"])     # substring-only neighbour excluded
    assert (form.category, form.merchants, form.merchant_text) == (None, ("UBER RIDES",), "uber")


@pytest.mark.parametrize("word, merchant", [("gas", "Vegas Buffet"), ("grocery", "Grocery Outlet")])
def test_category_fallback_needs_a_whole_word_and_a_word_not_near_a_category(word, merchant):
    out = fallback(word, [merchant])
    assert isinstance(out, Refusal) and "Groceries" in out.detail["categories"]


# --- periods ----------------------------------------------------------------------

@pytest.mark.parametrize("token,start,end", [
    ("this_month", date(2026, 11, 1), TODAY),
    ("last_3_months", date(2026, 8, 1), date(2026, 10, 31)),
    ("ytd", date(2026, 1, 1), TODAY),
    ("2025", date(2025, 1, 1), date(2025, 12, 31)),
])
def test_period_tokens_resolve_against_today(token, start, end):
    assert span(qf.parse_period(token, TODAY)) == (start, end)


@pytest.mark.parametrize("text,start,end", [
    ("--12", date(2025, 12, 1), date(2025, 12, 31)),                 # most recent started month
    ("--11", date(2026, 11, 1), date(2026, 11, 30)),
    ("--05-05..--07-05", date(2026, 5, 5), date(2026, 7, 5)),
    ("--10-01..--12-31", date(2025, 10, 1), date(2025, 12, 31)),     # not yet ended: rolls back
])
def test_yearless_periods(text, start, end):
    assert span(qf.parse_period(text, TODAY)) == (start, end)


@pytest.mark.parametrize("today,text,start,end", [
    (TODAY, "--02-29..--03-01", date(2024, 2, 29), date(2024, 3, 1)),
    (TODAY, "--03-01..--02-29", date(2023, 3, 1), date(2024, 2, 29)),
    (LEAP_DAY, "--02-29..--03-01", date(2024, 2, 29), date(2024, 3, 1)),     # was a 3-year span
    (LEAP_DAY, "--02-20..--02-29", date(2028, 2, 20), date(2028, 2, 29)),
])
def test_yearless_ranges_never_span_more_than_a_year(today, text, start, end):
    r = qf.parse_period(text, today)
    assert span(r) == (start, end) and r.days <= 367


@pytest.mark.parametrize("bad", [
    "2026-13", "2026-02-30..2026-03-01", "soon",
    "2026-12-01..--03-01",                  # a stated year is never rolled back
])
def test_bad_periods_are_not_understood(bad):
    with pytest.raises(NotUnderstood):
        qf.parse_period(bad, TODAY)


@pytest.mark.parametrize("bad", ["0000-05", "1500", "2999-01-01..2999-02-01", "2026-10-01..2026-09-01"])
def test_implausible_years_and_reversed_ranges_are_not_understood(bad):
    with pytest.raises(NotUnderstood):
        qf.parse_period(bad, TODAY)


def test_default_periods():
    for type_ in ("total_spent", "ranking", "list"):
        r = resolve(type=type_).period
        assert r.is_all and not r.said
    assert qf.parse_period("all", TODAY).is_all
    assert resolve(type="anomalies").period.label == "Oct 2026"          # latest full month


# --- comparisons ------------------------------------------------------------------

@pytest.mark.parametrize("period, start, end", [
    ("2026-10-05..2026-10-14", date(2026, 9, 25), date(2026, 10, 4)),    # same length before
    ("ytd", date(2025, 1, 1), date(2025, 11, 11)),                       # same dates last year
    ("this_year", date(2025, 1, 1), date(2025, 11, 11)),
])
def test_compare_second_period_default(period, start, end):
    assert span(resolve(type="compare", period=period).period_2) == (start, end)


def test_compare_defaults_to_the_latest_two_full_months():
    form = resolve(type="compare")
    assert (form.period.label, form.period_2.label) == ("Oct 2026", "Sep 2026")
    assert isinstance(qf.resolve(RawForm(type="category_change"), ctx(full_months=[(2026, 10)])), Refusal)


@pytest.mark.parametrize("period, period_2", [("all", "2026-10"), ("2026-10", "all"), ("2026", "2026-03")])
def test_all_data_or_overlapping_comparison_is_refused(period, period_2):
    assert isinstance(resolve(type="compare", period=period, period_2=period_2), Refusal)


# --- type rules -------------------------------------------------------------------

def test_unsupported_is_refused_and_total_received_means_received():
    out = resolve(type="unsupported")
    assert isinstance(out, Refusal) and out.kind == "unsupported"
    assert resolve(type="total_received").direction == "received"


def test_fields_a_type_does_not_apply_are_dropped():
    form = resolve(type="anomalies", direction="payments", category="groceries", merchant="costco")
    assert (form.direction, form.category, form.merchants) == ("spent", None, ())
    assert resolve(type="category_change", category="groceries").category is None


@pytest.mark.parametrize("kw", [
    dict(type="search", search_text="  "),                 # search without words
    dict(type="anomalies", search_text="parking"),         # search words on another type
    dict(type="compare", period_2="2026-10"),              # second period without a first
    dict(type="compare", period="--MM"),                   # unfilled period template
])
def test_misread_forms_are_not_understood(kw):
    with pytest.raises(NotUnderstood):
        resolve(**kw)
