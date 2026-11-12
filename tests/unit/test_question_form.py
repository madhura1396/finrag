"""Form validation and resolution (ADR 0004, details 1 and 2). Pure; a fixed today."""

from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from app import question_form as qf
from app.question_form import DataContext, NotUnderstood, RawForm, Refusal, Trip

TODAY = date(2026, 11, 11)


def ctx(**kw):
    base = dict(today=TODAY, categories=["Groceries", "Dining & DoorDash", "Shopping"],
                merchants=["Costco", "Costco Wholesale", "Target", "Trader Joe's", "GROCER MART S #101"],
                trips=[Trip("Lakeside Trip", date(2026, 7, 1), date(2026, 7, 10)),
                       Trip("Lake Weekend", date(2026, 8, 1), date(2026, 8, 3)),
                       Trip("Backwards", date(2026, 9, 10), date(2026, 9, 1))],
                full_months=[(2026, 8), (2026, 9), (2026, 10)])
    return DataContext(**{**base, **kw})


def resolve(**kw):
    return qf.resolve(RawForm(**kw), ctx())


# --- raw validation -----------------------------------------------------------

@pytest.mark.parametrize("data", [
    None, {}, {"type": "drop_tables"}, {"type": "count", "limit": 0}, {"type": "count", "limit": 51},
    {"type": "count", "limit": True}, {"type": "count", "direction": "sideways"}, {"type": "count", "order": "biggest"},
    {"type": "count", "category": 5},
])
def test_out_of_shape_output_is_not_understood(data):
    with pytest.raises(NotUnderstood):
        qf.parse_raw(data)


def test_valid_raw_form():
    raw = qf.parse_raw({"type": "ranking", "order": "least", "limit": 3, "period": "last_month", "category": ""})
    assert (raw.type, raw.order, raw.limit, raw.category) == ("ranking", "least", 3, None)


def test_unsupported_is_refused():
    out = resolve(type="unsupported")
    assert isinstance(out, Refusal) and out.kind == "unsupported"


def test_model_unreachable_raises_unavailable():
    post = MagicMock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(qf.ModelUnavailable):
        qf.ask_model("q", [], post=post)


def test_model_garbage_is_not_understood():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "not json"}
    with pytest.raises(NotUnderstood):
        qf.ask_model("q", [], post=MagicMock(return_value=response))


def test_ask_sends_schema_and_temperature_zero():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": '{"type": "count"}'}
    post = MagicMock(return_value=response)
    assert qf.ask_model("how many", [], post=post).type == "count"
    sent = post.call_args.kwargs["json"]
    assert sent["options"]["temperature"] == 0 and sent["format"]["properties"]["type"]["enum"][-1] == "unsupported"


# --- categories and merchants --------------------------------------------------

def test_unknown_category_is_refused_with_the_list():
    out = resolve(type="total_spent", category="Petrol")
    assert isinstance(out, Refusal) and "Groceries" in out.detail["categories"]


def test_category_matches_ignoring_case():
    assert resolve(type="total_spent", category="groceries").category == "Groceries"


def test_contains_finds_longer_names():
    assert resolve(type="total_spent", merchant="costco").merchants == ("Costco", "Costco Wholesale")


def test_similarity_catches_typos():
    assert "Costco" in resolve(type="total_spent", merchant="costcoo").merchants


def test_short_text_never_matches_by_containment():
    out = resolve(type="total_spent", merchant="at")
    assert isinstance(out, Refusal)


def test_punctuation_is_normalised():
    assert resolve(type="total_spent", merchant="trader joes").merchants == ("Trader Joe's",)


def test_no_match_lists_closest_names():
    out = resolve(type="total_spent", merchant="Costko Wholesail")
    assert isinstance(out, Refusal) and out.detail["closest"]


def test_known_limit_plural_typed_name_does_not_match():
    # Pinned on purpose: changing it should be a decision, not an accident.
    out = resolve(type="total_spent", merchant="grocer marts")
    assert isinstance(out, Refusal) and "GROCER MART S #101" in out.detail["closest"]


# --- trips -----------------------------------------------------------------------------

def test_trip_exact_and_contains():
    assert resolve(type="total_spent", trip="lakeside trip").period.start == date(2026, 7, 1)
    assert resolve(type="total_spent", trip="weekend").period.end == date(2026, 8, 3)


def test_ambiguous_trip_is_refused():
    out = resolve(type="total_spent", trip="lake")
    assert isinstance(out, Refusal) and len(out.detail["trips"]) == 2


def test_trip_ending_before_it_starts_is_refused():
    assert isinstance(resolve(type="total_spent", trip="backwards"), Refusal)


def test_trip_intersects_a_period_and_empty_intersection_is_refused():
    r = resolve(type="total_spent", trip="lakeside", period="2026-07-05..2026-07-31").period
    assert (r.start, r.end) == (date(2026, 7, 5), date(2026, 7, 10))
    assert isinstance(resolve(type="total_spent", trip="lakeside", period="2026-10"), Refusal)


# --- periods ----------------------------------------------------------------------

@pytest.mark.parametrize("token,start,end", [
    ("this_month", date(2026, 11, 1), TODAY),
    ("last_month", date(2026, 10, 1), date(2026, 10, 31)),
    ("last_3_months", date(2026, 8, 1), date(2026, 10, 31)),
    ("this_year", date(2026, 1, 1), TODAY),
    ("ytd", date(2026, 1, 1), TODAY),
    ("last_year", date(2025, 1, 1), date(2025, 12, 31)),
    ("2025-12", date(2025, 12, 1), date(2025, 12, 31)),
])
def test_relative_tokens_resolve_against_today(token, start, end):
    r = qf.parse_period(token, TODAY)
    assert (r.start, r.end) == (start, end)


def test_all_means_all_data():
    assert qf.parse_period("all", TODAY).is_all


def test_yearless_month_is_the_most_recent_started_one():
    r = qf.parse_period("--12", TODAY)
    assert (r.start, r.end) == (date(2025, 12, 1), date(2025, 12, 31))
    assert qf.parse_period("--11", TODAY).start == date(2026, 11, 1)


def test_yearless_range_resolves_to_this_year_and_wrapping_rolls_back():
    r = qf.parse_period("--05-05..--07-05", TODAY)
    assert (r.start, r.end) == (date(2026, 5, 5), date(2026, 7, 5))
    r = qf.parse_period("--10-01..--12-31", TODAY)
    assert (r.start, r.end) == (date(2025, 10, 1), date(2025, 12, 31))


@pytest.mark.parametrize("bad", ["2026-13", "--00", "2026-02-30..2026-03-01", "soon", "--13-01..--14-01"])
def test_bad_periods_are_not_understood(bad):
    with pytest.raises(NotUnderstood):
        qf.parse_period(bad, TODAY)


@pytest.mark.parametrize("type_", ["total_spent", "count", "ranking", "largest", "list", "monthly_trend"])
def test_default_period_is_all_data(type_):
    r = resolve(type=type_).period
    assert r.is_all and not r.said


def test_compare_defaults_to_the_latest_two_full_months():
    form = resolve(type="compare")
    assert (form.period.label, form.period_2.label) == ("Oct 2026", "Sep 2026")


def test_compare_with_one_period_uses_the_same_length_before():
    form = resolve(type="compare", period="2026-10-05..2026-10-14")
    assert (form.period_2.start, form.period_2.end) == (date(2026, 9, 25), date(2026, 10, 4))


def test_ytd_compare_defaults_to_same_dates_last_year():
    form = resolve(type="compare", period="ytd")
    assert (form.period_2.start, form.period_2.end) == (date(2025, 1, 1), date(2025, 11, 11))


def test_month_types_default_to_the_latest_full_month():
    assert resolve(type="anomalies").period.label == "Oct 2026"


def test_comparison_needs_two_full_months():
    out = qf.resolve(RawForm(type="category_change"), ctx(full_months=[(2026, 10)]))
    assert isinstance(out, Refusal)


def test_total_received_defaults_direction_to_received():
    assert resolve(type="total_received").direction == "received"


def test_search_needs_search_words():
    with pytest.raises(NotUnderstood):
        resolve(type="search", search_text="  ")


# --- review round ---------------------------------------------------------------------

def test_fields_a_type_does_not_apply_are_dropped():
    form = resolve(type="anomalies", direction="payments", category="groceries", merchant="costco")
    assert (form.direction, form.category, form.merchants) == ("spent", None, ())
    form = resolve(type="category_change", category="groceries")
    assert form.category is None


def test_all_data_in_a_comparison_is_refused_not_a_crash():
    assert isinstance(resolve(type="compare", period="all", period_2="2026-10"), Refusal)
    assert isinstance(resolve(type="compare", period="2026-10", period_2="all"), Refusal)


@pytest.mark.parametrize("bad", ["0000", "0000-05", "0000-01-01..0000-02-01"])
def test_year_zero_is_not_understood(bad):
    with pytest.raises(NotUnderstood):
        qf.parse_period(bad, TODAY)


def test_a_stated_year_is_never_rolled_back():
    with pytest.raises(NotUnderstood):
        qf.parse_period("2026-10-01..2026-09-01", TODAY)
    with pytest.raises(NotUnderstood):                      # end resolves to Mar 2026, before the stated start
        qf.parse_period("2026-12-01..--03-01", TODAY)


def test_whole_year_and_yearless_leap_day():
    r = qf.parse_period("2025", TODAY)
    assert (r.start, r.end) == (date(2025, 1, 1), date(2025, 12, 31))
    r = qf.parse_period("--02-29..--03-01", TODAY)
    assert (r.start, r.end) == (date(2024, 2, 29), date(2024, 3, 1))


def test_a_second_period_without_a_first_is_not_understood():
    with pytest.raises(NotUnderstood):
        resolve(type="compare", period_2="2026-10")


def test_this_year_compares_against_the_same_dates_last_year():
    form = resolve(type="compare", period="this_year")
    assert (form.period_2.start, form.period_2.end) == (date(2025, 1, 1), date(2025, 11, 11))


# --- final review of #22 ----------------------------------------------------------------

LEAP_DAY = date(2028, 2, 29)


@pytest.mark.parametrize("today,text,start,end", [
    (TODAY, "--02-29..--03-01", date(2024, 2, 29), date(2024, 3, 1)),
    (TODAY, "--02-20..--02-29", date(2024, 2, 20), date(2024, 2, 29)),
    (TODAY, "--03-01..--02-29", date(2023, 3, 1), date(2024, 2, 29)),
    (LEAP_DAY, "--02-29..--03-01", date(2024, 2, 29), date(2024, 3, 1)),     # was a 3-year span
    (LEAP_DAY, "--02-20..--02-29", date(2028, 2, 20), date(2028, 2, 29)),
])
def test_yearless_ranges_never_span_more_than_a_year(today, text, start, end):
    r = qf.parse_period(text, today)
    assert (r.start, r.end) == (start, end) and r.days <= 367


@pytest.mark.parametrize("bad", ["0001-01", "1500", "2999-01-01..2999-02-01"])
def test_implausible_years_are_not_understood(bad):
    with pytest.raises(NotUnderstood):
        qf.parse_period(bad, TODAY)


def test_overlapping_comparison_periods_are_refused():
    out = resolve(type="compare", period="2026", period_2="2026-03")
    assert isinstance(out, Refusal) and "overlap" in out.reason



# --- demo fixes ------------------------------------------------------------------

@pytest.mark.parametrize("value", ["null", " None ", "n/a"])
def test_spelled_out_null_is_absent(value):
    raw = qf.parse_raw({"type": "ranking", "merchant": value, "period": value, "direction": value})
    assert (raw.merchant, raw.period, raw.direction) == (None, None, None)


@pytest.mark.parametrize("type_", ["total_spent", "compare"])
def test_unfilled_period_template_is_not_understood(type_):
    with pytest.raises(NotUnderstood):
        resolve(type=type_, period="--MM")


def fallback(category, merchants):
    return qf.resolve(RawForm(type="total_spent", category=category), ctx(merchants=merchants))


def test_unknown_category_naming_a_merchant_by_whole_word_becomes_the_merchant():
    form = fallback("uber", ["UBER TRIP", "TUBERS CAFE", "Taco Town"])     # substring-only neighbour excluded
    assert (form.category, form.merchants, form.merchant_text) == (None, ("UBER TRIP",), "uber")


@pytest.mark.parametrize("word, merchant", [("gas", "Vegas Buffet"), ("car", "Caribou Coffee"),
                                            ("grocery", "Grocery Outlet")])
def test_category_fallback_needs_a_whole_word_and_a_word_not_near_a_category(word, merchant):
    out = fallback(word, [merchant])
    assert isinstance(out, Refusal) and "Groceries" in out.detail["categories"]


@pytest.mark.parametrize("type_", ["unsupported", "anomalies"])
def test_search_words_on_a_non_search_type_are_not_understood(type_):
    with pytest.raises(NotUnderstood):
        resolve(type=type_, search_text="parking")

