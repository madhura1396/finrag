import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from app.statement import Mismatch, Row, StatementRejected, SummaryLine, validate
from tests.unit.factories import (
    PERIOD_END,
    PERIOD_START,
    D,
    valid_statement,
    with_row,
    with_summary,
)


def checks(mismatches):
    return [m.check for m in mismatches]


def check_pairs(mismatches):
    return {(m.check, m.section) for m in mismatches}


def summary_edit(**lines):
    data = valid_statement()
    return replace(data, summary=dict(data.summary, **lines))


# --- the happy path ---------------------------------------------------------

def test_valid_statement_with_zero_row_and_zero_total_has_no_mismatches():
    # The fixture holds a 0.00 interest row (ADR 0003, 4b) and a printed $0.00
    # Other Credits total with no rows.
    data = valid_statement()
    assert any(r.amount == 0 for r in data.rows)
    assert data.section_totals["other_credits"] == 0
    assert validate(data) == []
    assert validate(replace(data, reader_mismatches=None)) == []


def test_refund_inside_a_charge_section_nets_correctly():
    # A bank that lists a refund among purchases prints the net total.
    data = valid_statement()
    rows = list(data.rows) + [Row(date(2025, 12, 22), None, "BOOKSTORE C REFUND", D("5.00"), "purchases")]
    totals = dict(data.section_totals, purchases=D("30.50"))
    summary = dict(data.summary)
    summary["purchases"] = replace(summary["purchases"], value=D("30.50"))
    summary["new_balance"] = replace(summary["new_balance"], value=D("433.73"))
    assert validate(replace(data, rows=rows, section_totals=totals, summary=summary)) == []


# --- rows -------------------------------------------------------------------

def test_no_rows():
    assert "no_rows" in checks(validate(replace(valid_statement(), rows=[])))


@pytest.mark.parametrize("index, amount, section", [
    (0, "10.00", "purchases"),      # a charge read as a credit: same size, wrong sign
    (2, "-100.00", "payments"),     # a payment read as a charge
])
def test_wrong_sign_is_caught(index, amount, section):
    assert check_pairs(validate(with_row(valid_statement(), index, amount=D(amount)))) == {("section_total", section)}


@pytest.mark.parametrize("amount", [Decimal("NaN"), D("-10.004"), D("-100000000.00"), -10.0])
def test_unstorable_amount_is_row_invalid(amount):
    assert ("row_invalid", "purchases") in check_pairs(validate(with_row(valid_statement(), 0, amount=amount)))


@pytest.mark.parametrize("changes", [
    {"trans_date": None}, {"trans_date": datetime(2025, 12, 5, 9, 30)}, {"description": "   "}, {"section": "groceries"},
])
def test_malformed_row_is_row_invalid_not_raised(changes):
    assert "row_invalid" in checks(validate(with_row(valid_statement(), 0, **changes)))


def test_more_than_two_decimals_cannot_reconcile_by_rounding():
    # 10.004 + 25.496 = 35.50 would pass a sum check, but Numeric(10,2) would
    # store 10.00 and 25.50: the validated and stored values must not differ.
    data = with_row(with_row(valid_statement(), 0, amount=D("-10.004")), 1, amount=D("-25.496"))
    assert "row_invalid" in checks(validate(data))


@pytest.mark.parametrize("days, ok", [(-7, True), (-8, False)])
def test_date_window_before_start(days, ok):
    data = with_row(valid_statement(), 0, trans_date=PERIOD_START + timedelta(days=days))
    assert check_pairs(validate(data)) == (set() if ok else {("date_window", "purchases")})


def test_one_day_after_end_fails():
    data = with_row(valid_statement(), 0, trans_date=PERIOD_END + timedelta(days=1))
    assert check_pairs(validate(data)) == {("date_window", "purchases")}


# --- period -----------------------------------------------------------------

@pytest.mark.parametrize("changes", [
    {"period_start": None}, {"period_end": None}, {"period_start": PERIOD_END, "period_end": PERIOD_START},
])
def test_missing_or_reversed_period_is_reported(changes):
    assert checks(validate(replace(valid_statement(), **changes))) == ["period_not_found"]


# --- printed totals ---------------------------------------------------------

@pytest.mark.parametrize("field", ["section_totals", "summary"])
def test_missing_printed_totals_is_rejected(field):
    assert checks(validate(replace(valid_statement(), **{field: None}))) == ["no_printed_totals"]


def test_section_total_off_by_one_cent():
    found = validate(with_row(valid_statement(), 1, amount=D("-25.51")))
    mismatch = next(m for m in found if m.check == "section_total")
    assert (mismatch.section, mismatch.expected, mismatch.actual) == ("purchases", "-35.50", "-35.51")


@pytest.mark.parametrize("section, drop", [("purchases", "total"), ("payments", "rows")])
def test_rows_and_printed_total_must_both_exist(section, drop):
    data = valid_statement()
    if drop == "total":
        data = replace(data, section_totals={k: v for k, v in data.section_totals.items() if k != section})
    else:
        data = replace(data, rows=[r for r in data.rows if r.section != section])
    assert ("section_total", section) in check_pairs(validate(data))


@pytest.mark.parametrize("bad_total", [Decimal("sNaN"), 35.5, None])
def test_unreadable_printed_total_is_reported_not_raised(bad_total):
    data = valid_statement()
    found = validate(replace(data, section_totals=dict(data.section_totals, purchases=bad_total)))
    assert ("section_total", "purchases") in check_pairs(found)


# --- account summary --------------------------------------------------------

def test_summary_equation_off_by_one_cent():
    data = with_summary(valid_statement(), new_balance="438.74")
    assert check_pairs(validate(data)) == {("summary_equation", None)}


@pytest.mark.parametrize("values, section", [
    ({"purchases": "35.51", "new_balance": "438.74"}, "purchases"),         # disagrees with its section total
    ({"cash_advances": "5.00", "new_balance": "443.73"}, "cash_advances"),  # nonzero with no section: fail closed
])
def test_summary_line_must_match_its_section(values, section):
    assert check_pairs(validate(with_summary(valid_statement(), **values))) == {("summary_section", section)}


def test_flipped_section_with_negative_printed_values_fails_on_the_equation():
    # Purchases flipped to positive and the printed values made negative to
    # match: internally consistent, but the equation misses the new balance.
    data = valid_statement()
    data = with_row(with_row(data, 0, amount=D("10.00")), 1, amount=D("25.50"))
    data = replace(data, section_totals=dict(data.section_totals, purchases=D("-35.50")))
    data = with_summary(data, purchases="-35.50")
    assert "summary_equation" in checks(validate(data))


S = valid_statement().summary


@pytest.mark.parametrize("summary", [
    {k: v for k, v in S.items() if k != "new_balance"},                     # no new balance
    dict(S, purchases=replace(S["purchases"], value=35.5)),                 # float value
    dict(S, purchases=D("35.50")),                                          # not a SummaryLine
    {"new_balance": SummaryLine(D("0.00"), 0, None)},                       # only new balance checks nothing
])
def test_incomplete_summary_is_reported(summary):
    assert "summary_incomplete" in checks(validate(replace(valid_statement(), summary=summary)))


@pytest.mark.parametrize("key, sign", [("purchases", 0), ("purchases", 5), ("new_balance", 1), ("payments", True)])
def test_bad_summary_sign_is_reported(key, sign):
    assert "summary_incomplete" in checks(validate(summary_edit(**{key: replace(S[key], sign=sign)})))


def test_summary_direction_per_section_must_be_exactly_one_line():
    no_direction = {k: v for k, v in S.items() if k != "purchases"}
    no_direction["new_balance"] = replace(S["new_balance"], value=D("474.23"))
    found = validate(replace(valid_statement(), summary=no_direction))
    assert ("summary_incomplete", "purchases") in check_pairs(found)
    # An extra opposite-sign line for payments must not get to decide direction.
    found = validate(summary_edit(payments_again=SummaryLine(D("100.00"), +1, "payments")))
    assert ("summary_incomplete", "payments") in check_pairs(found)


# --- reporting --------------------------------------------------------------

def test_all_mismatches_are_reported_reader_ones_first():
    reader_found = Mismatch("unknown_summary_label", None, "a profile label", "unmatched row")
    data = with_row(valid_statement(), 0, trans_date=PERIOD_END + timedelta(days=1))
    data = with_row(data, 1, amount=D("-25.51"))
    data = replace(with_summary(data, new_balance="1.00"), reader_mismatches=[reader_found])
    found = validate(data)
    assert found[0] is reader_found
    assert {"date_window", "section_total", "summary_equation"} <= set(checks(found))


def test_unknown_check_name_is_refused():
    with pytest.raises(ValueError):
        Mismatch("made_up_check", None, "x", "y")


def test_mismatches_serialise_money_as_strings_and_the_message_names_no_amounts():
    found = validate(with_row(valid_statement(), 1, amount=D("-25.51")))
    body = json.loads(json.dumps([m.to_dict() for m in found]))
    assert body[0]["expected"] == "-35.50" and isinstance(body[0]["actual"], str)
    message = str(StatementRejected(found))
    assert "section_total" in message and "35.5" not in message


# --- malformed input becomes a mismatch, never an exception (review M2) ------

@pytest.mark.parametrize("changes, expected_check", [
    ({"rows": "not a list"},                 "no_rows"),
    ({"rows": [None]},                       "row_invalid"),
    ({"reader_mismatches": [None]},          "unexpected_error"),
    ({"summary": "previous 500"},            "no_printed_totals"),
])
def test_malformed_containers_are_reported_not_raised(changes, expected_check):
    assert expected_check in checks(validate(replace(valid_statement(), **changes)))


GARBAGE = [None, "", "x", 0, -1, 1.5, Decimal("NaN"), Decimal("sNaN"), [], {}, object()]


@pytest.mark.parametrize("field", ["period_start", "period_end", "rows", "section_totals", "summary", "reader_mismatches"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_statement_fields(field, value):
    assert all(isinstance(m, Mismatch) for m in validate(replace(valid_statement(), **{field: value})))


@pytest.mark.parametrize("field", ["trans_date", "post_date", "description", "amount", "section"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_row_fields(field, value):
    assert all(isinstance(m, Mismatch) for m in validate(with_row(valid_statement(), 0, **{field: value})))


@pytest.mark.parametrize("field", ["value", "sign", "section"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_summary_fields(field, value):
    found = validate(summary_edit(purchases=replace(S["purchases"], **{field: value})))
    assert all(isinstance(m, Mismatch) for m in found)
