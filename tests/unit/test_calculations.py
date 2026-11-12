"""Every read-path number, on synthetic signed rows (ADR 0004). No DB, no model."""

from datetime import date
from decimal import Decimal

import pytest

from app import calculations as calc
from app import read_config as rs
from app.calculations import RowView
from app.read_config import Form, Range, Refusal, month_range

D = Decimal


def row(i, d, amount, section="purchases", category="Groceries", merchant="Grocer Mart", needs_review=False):
    return RowView(i, d, D(amount), section, category, "model" if category else "pending", merchant,
                   merchant.upper(), needs_review, None, None)


# Statement covering all of Oct and Nov 2026, and 1-14 Dec 2026.
PERIODS = [(date(2026, 10, 1), date(2026, 10, 31)), (date(2026, 11, 1), date(2026, 11, 30)),
           (date(2026, 12, 1), date(2026, 12, 14))]
OCT, NOV = month_range(2026, 10), month_range(2026, 11)
ALL = Range(None, None, "all data", said=False)

ROWS = [
    row(1, date(2026, 10, 3), "-40.00"),
    row(2, date(2026, 10, 9), "-10.00", category="Dining & DoorDash", merchant="Taco Town"),
    row(3, date(2026, 10, 20), "100.00", section="payments", category="Other", merchant="Payment"),
    row(4, date(2026, 10, 22), "5.00", section="other_credits", merchant="Grocer Mart"),        # refund
    row(5, date(2026, 10, 31), "0.00", section="interest", category="Interest", merchant="Interest"),
    row(6, date(2026, 10, 31), "-1.50", section="interest", category="Interest", merchant="Interest"),
    row(7, date(2026, 11, 2), "-25.00", category=None, merchant="Mystery Shop"),                  # pending
    row(8, date(2026, 11, 5), "-60.00"),
    row(9, date(2026, 11, 7), "-2.00", section="purchases", category="Groceries", merchant="Grocer Mart",
        needs_review=True),
]


def form(**kw):
    return Form(**{"type": "total_spent", **kw})


# --- directions ---------------------------------------------------------------------

def test_spent_excludes_zero_and_credits_and_states_refunds_beside_never_netted():
    result = calc.compute(form(period=OCT), ROWS, PERIODS)
    assert (result.facts["total"], result.facts["count"], result.facts["refunds"]) == (D("51.50"), 3, D("5.00"))
    assert calc.compute(form(type="count", period=OCT), ROWS, PERIODS).facts["count"] == 3


@pytest.mark.parametrize("direction,expected", [("payments", "100.00"), ("refunds", "5.00"), ("received", "105.00")])
def test_received_directions(direction, expected):
    result = calc.compute(form(type="total_received", direction=direction, period=OCT), ROWS, PERIODS)
    assert result.facts["total"] == D(expected)


# --- pending rows ---------------------------------------------------------------------

@pytest.mark.parametrize("category, total", [(None, "87.00"), ("Groceries", "62.00")])
def test_pending_row_counts_only_in_an_unfiltered_total(category, total):
    assert calc.compute(form(period=NOV, category=category), ROWS, PERIODS).facts["total"] == D(total)


def test_pending_row_is_its_own_bucket_in_a_category_ranking():
    result = calc.compute(form(type="ranking", group_by="category", period=NOV), ROWS, PERIODS)
    assert [r["name"] for r in result.table] == ["Groceries", rs.PENDING_BUCKET]


def test_notes_count_pending_and_needs_review_in_the_period():
    assert calc.compute(form(period=NOV), ROWS, PERIODS).notes == {"needs_review": 1, "pending": 1}


# --- ranking, largest, list -----------------------------------------------------------

def test_ranking_both_orders_with_signed_rows():
    most = calc.compute(form(type="ranking", period=ALL, limit=2), ROWS, PERIODS)
    least = calc.compute(form(type="ranking", period=ALL, order="least", limit=2), ROWS, PERIODS)
    assert [r["name"] for r in most.table] == ["Grocer Mart", "Mystery Shop"]
    assert [r["name"] for r in least.table] == ["Interest", "Taco Town"]


def test_largest_takes_the_biggest_charges():
    result = calc.compute(form(type="largest", period=ALL, limit=2), ROWS, PERIODS)
    assert [r["amount"] for r in result.table] == [D("60.00"), D("40.00")]


def test_list_includes_zero_rows_and_caps():
    result = calc.compute(form(type="list", period=OCT, merchant_text="interest", merchants=("Interest",)),
                          ROWS, PERIODS)
    assert [r.amount for r in result.rows] == [D("0.00"), D("-1.50")]


# --- coverage ---------------------------------------------------------------------------

def test_coverage_full_partial_none_and_full_months():
    assert calc.coverage(OCT, PERIODS).status == "full"
    assert calc.coverage(month_range(2026, 12), PERIODS).status == "partial"
    assert calc.coverage(month_range(2026, 1), PERIODS).status == "none"
    assert calc.full_months(PERIODS) == [(2026, 10), (2026, 11)]


# --- compare and category_change -----------------------------------------------------

def test_compare_equal_lengths_uses_totals():
    a, b = Range(date(2026, 11, 1), date(2026, 11, 10), "a"), Range(date(2026, 10, 1), date(2026, 10, 10), "b")
    result = calc.compute(form(type="compare", period=a, period_2=b), ROWS, PERIODS)
    assert (result.facts["basis"], result.facts["change"], result.facts["difference"]) == ("total", "up", D("37.00"))


def test_compare_unequal_lengths_decides_on_per_day_average():
    # Nov (30 days, 87.00 -> 2.90/day) against Oct (31 days, 51.50 -> 1.66/day)
    result = calc.compute(form(type="compare", period=NOV, period_2=OCT), ROWS, PERIODS)
    assert (result.facts["basis"], result.facts["a_per_day"], result.facts["b_per_day"]) == \
        ("per_day", D("2.90"), D("1.66"))
    assert result.facts["change"] == "up"


def test_compare_refuses_a_partly_covered_range():
    dec = month_range(2026, 12)
    out = calc.compute(form(type="compare", period=dec, period_2=NOV), ROWS, PERIODS)
    assert isinstance(out, Refusal) and out.detail["coverage"] == "partial"


def test_category_change_sorts_by_largest_decrease_and_shows_totals_and_per_day():
    result = calc.compute(form(type="category_change", period=NOV, period_2=OCT), ROWS, PERIODS)
    assert result.facts["basis"] == "per_day"
    first = result.table[0]
    # Oct dining 10.00/31 = 0.32/day and Oct interest 1.50/31 = 0.05/day both fell to zero: dining fell most.
    assert (first["category"], first["now"], first["before"], first["before_per_day"]) == \
        ("Dining & DoorDash", D("0"), D("10.00"), D("0.32"))
    assert first["change"] == D("-0.32")


# --- monthly trend ---------------------------------------------------------------------

def test_monthly_trend_labels_partial_months_and_never_compares_them():
    result = calc.compute(form(type="monthly_trend"), ROWS, PERIODS)
    by_month = {r["month"]: r for r in result.table}
    assert by_month["Dec 2026"]["partial"] and by_month["Dec 2026"]["change"] is None
    assert by_month["Nov 2026"]["change"] == "up"


# --- anomalies and savings ---------------------------------------------------------------

def history_rows(dining_by_month, start_month=6):
    rows, i = [], 100
    for offset, amount in enumerate(dining_by_month):
        m = start_month + offset
        if amount:
            rows.append(row(i, date(2026, m, 10), f"-{amount}", category="Dining", merchant=f"Diner {m}"))
            i += 1
    return rows


SIX_MONTHS = [(date(2026, m, 1), month_range(2026, m).end) for m in range(6, 12)]


@pytest.mark.parametrize("history, latest, excess", [
    ("100.00", "160.00", "60.00"),
    ("100.00", "150.00", None),         # exactly 1.5x is not "more than" 1.5x
    ("90.00", "140.00", "50.00"),       # 1.56x and exactly +50 meets "at least $50"
    ("10.00", "40.00", None),           # 4x but only +30
])
def test_anomaly_needs_ratio_and_minimum_excess(history, latest, excess):
    table = calc.compute(form(type="anomalies", period=NOV), history_rows([history] * 5 + [latest]), SIX_MONTHS).table
    if excess is None:
        assert table == []
    else:
        assert (table[0]["category"], table[0]["median"], table[0]["excess"]) == ("Dining", D(history), D(excess))


def test_anomalies_need_three_months_of_history():
    periods = [(date(2026, m, 1), month_range(2026, m).end) for m in (9, 10, 11)]
    out = calc.compute(form(type="anomalies", period=NOV), history_rows(["1", "1", "1"], 9), periods)
    assert isinstance(out, Refusal) and "have 2" in out.reason


def test_recurring_needs_three_of_the_last_four_months():
    rows = [row(200 + m, date(2026, m, 3), "-9.99", category="Entertainment", merchant="StreamCo")
            for m in (8, 9, 11)]
    result = calc.compute(form(type="savings_opportunities", period=NOV), rows, SIX_MONTHS)
    assert result.facts["recurring"][0]["merchant"] == "StreamCo"
    rows = [r for r in rows if r.date.month != 9]
    result = calc.compute(form(type="savings_opportunities", period=NOV), rows, SIX_MONTHS)
    assert result.facts["recurring"] == []


def test_savings_lists_categories_above_their_average_and_fees():
    rows = history_rows(["50.00", "50.00", "50.00", "50.00", "50.00", "80.00"])
    rows.append(row(300, date(2026, 11, 30), "-3.00", section="fees", category="Fees", merchant="Late fee"))
    result = calc.compute(form(type="savings_opportunities", period=NOV), rows, SIX_MONTHS)
    assert result.facts["above_average"][0]["excess"] == D("30.00")
    assert result.facts["fees_and_interest"] == D("3.00")


def test_month_types_refuse_a_non_month_period():
    out = calc.compute(form(type="anomalies", period=Range(date(2026, 11, 1), date(2026, 11, 15), "x")), [], SIX_MONTHS)
    assert isinstance(out, Refusal)


# --- search ------------------------------------------------------------------------------

def test_search_counts_and_totals_only_the_matches():
    matches = [ROWS[0], ROWS[7]]
    result = calc.compute(form(type="search", search_text="grocer", period=ALL), ROWS, PERIODS, matches)
    assert (result.facts["count"], result.facts["total"]) == (2, D("100.00"))
    assert "no_embedding" in result.notes


def test_direction_params_drive_the_search_query():
    assert calc.direction_params("refunds") == {"sign": "positive", "only_sections": None,
                                                "except_sections": ["payments"]}


# --- review fixes ------------------------------------------------------------------------

def test_trend_month_clipped_by_the_asked_range_is_partial_and_not_compared():
    periods = [(date(2026, 5, 1), date(2026, 7, 31))]
    rows = [row(1, date(2026, 5, 5), "-10.00"), row(2, date(2026, 5, 20), "-500.00"),
            row(3, date(2026, 6, 10), "-100.00"), row(4, date(2026, 7, 4), "-5.00"), row(5, date(2026, 7, 20), "-900.00")]
    r = Range(date(2026, 5, 5), date(2026, 7, 5), "x")
    table = {t["month"]: t for t in calc.compute(form(type="monthly_trend", period=r), rows, periods).table}
    assert table["May 2026"]["partial"] and table["Jul 2026"]["partial"]
    assert table["Jun 2026"]["change"] is None              # its neighbours are partial: nothing to compare
    assert table["Jul 2026"]["change"] is None


def test_trend_never_compares_across_a_gap():
    periods = [(date(2026, 7, 1), date(2026, 7, 31)), (date(2026, 9, 1), date(2026, 9, 30))]
    rows = [row(1, date(2026, 7, 5), "-10.00"), row(2, date(2026, 9, 5), "-20.00")]
    table = {t["month"]: t for t in calc.compute(form(type="monthly_trend"), rows, periods).table}
    assert table["Sep 2026"]["change"] is None


def test_trend_includes_a_month_with_rows_but_no_statement():
    rows = [row(1, date(2026, 9, 28), "-30.00")]          # dated before its statement started
    table = calc.compute(form(type="monthly_trend"), rows, PERIODS).table
    assert {"month": "Sep 2026", "amount": D("30.00"), "partial": True, "change": None} in table


def test_compare_same_and_zero_rows_in_rankings():
    a, b = Range(date(2026, 10, 1), date(2026, 10, 2), "a"), Range(date(2026, 10, 3), date(2026, 10, 4), "b")
    rows = [row(1, date(2026, 10, 1), "-5.00"), row(2, date(2026, 10, 3), "-5.00"),
            row(3, date(2026, 10, 3), "0.00", merchant="Zero Co")]
    assert calc.compute(form(type="compare", period=a, period_2=b), rows, PERIODS).facts["change"] == "same"
    ranking = calc.compute(form(type="ranking", period=ALL), rows, PERIODS)
    assert "Zero Co" not in [t["name"] for t in ranking.table]


def test_notes_cover_both_compared_periods():
    out = calc.compute(form(type="compare", period=NOV, period_2=OCT), ROWS, PERIODS)
    assert out.notes == {"needs_review": 1, "pending": 1}
    rows = ROWS + [row(50, date(2026, 10, 2), "-1.00", category=None, merchant="Odd", needs_review=True)]
    out = calc.compute(form(type="compare", period=NOV, period_2=OCT), rows, PERIODS)
    assert out.notes == {"needs_review": 2, "pending": 2}


def test_search_reports_when_the_cap_was_hit():
    matches = [row(i, date(2026, 10, 1), "-1.00") for i in range(rs.SEARCH_CAP)]
    result = calc.compute(form(type="search", search_text="x", period=ALL), ROWS, PERIODS, matches)
    assert result.facts["capped"] is True and result.facts["count"] == rs.SEARCH_CAP
