import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from app.statement import CHECKS, Mismatch, Row, StatementRejected, SummaryLine, validate
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


# --- the happy path ---------------------------------------------------------

def test_valid_statement_has_no_mismatches():
    assert validate(valid_statement()) == []


def test_zero_amount_row_is_valid():
    # The fixture already holds a 0.00 interest row (ADR 0003, 4b); make it explicit.
    data = valid_statement()
    assert any(r.amount == 0 for r in data.rows)
    assert validate(data) == []


def test_printed_zero_total_with_no_rows_is_valid():
    data = valid_statement()
    assert data.section_totals["other_credits"] == 0
    assert not any(r.section == "other_credits" for r in data.rows)
    assert validate(data) == []


# --- rows -------------------------------------------------------------------

def test_no_rows():
    assert "no_rows" in checks(validate(replace(valid_statement(), rows=[])))


def test_charge_recorded_as_credit_is_caught():
    # A Charges amount read as a credit: same size, wrong sign. The signed sum
    # moves by twice the amount, so the section cannot reconcile.
    data = with_row(valid_statement(), 0, amount=D("10.00"))
    assert check_pairs(validate(data)) == {("section_total", "purchases")}


def test_payment_recorded_as_charge_is_caught():
    data = with_row(valid_statement(), 2, amount=D("-100.00"))
    found = validate(data)
    assert check_pairs(found) == {("section_total", "payments")}
    mismatch = found[0]
    assert (mismatch.expected, mismatch.actual) == ("100.00", "-100.00")


def test_refund_inside_a_charge_section_nets_correctly():
    # A bank that lists a refund among purchases prints the net total.
    data = valid_statement()
    rows = list(data.rows) + [Row(date(2025, 12, 22), None, "BOOKSTORE C REFUND", D("5.00"), "purchases")]
    totals = dict(data.section_totals, purchases=D("30.50"))
    summary = dict(data.summary)
    summary["purchases"] = replace(summary["purchases"], value=D("30.50"))
    summary["new_balance"] = replace(summary["new_balance"], value=D("433.73"))
    assert validate(replace(data, rows=rows, section_totals=totals, summary=summary)) == []


def test_section_without_summary_direction_is_incomplete_not_a_crash():
    data = valid_statement()
    summary = {k: v for k, v in data.summary.items() if k != "purchases"}
    summary["new_balance"] = replace(summary["new_balance"], value=D("474.23"))
    found = validate(replace(data, summary=summary))
    assert ("summary_incomplete", "purchases") in check_pairs(found)


def test_nan_amount_is_row_invalid_not_an_exception():
    data = with_row(valid_statement(), 0, amount=Decimal("NaN"))
    assert ("row_invalid", "purchases") in check_pairs(validate(data))


def test_missing_date_is_row_invalid_not_type_error():
    data = with_row(valid_statement(), 0, trans_date=None)
    assert ("row_invalid", "purchases") in check_pairs(validate(data))


def test_unknown_section_is_row_invalid():
    data = with_row(valid_statement(), 0, section="groceries")
    assert "row_invalid" in checks(validate(data))


# --- date window ------------------------------------------------------------

def test_seven_days_before_start_passes():
    data = with_row(valid_statement(), 0, trans_date=PERIOD_START - timedelta(days=7))
    assert validate(data) == []


def test_eight_days_before_start_fails():
    data = with_row(valid_statement(), 0, trans_date=PERIOD_START - timedelta(days=8))
    assert check_pairs(validate(data)) == {("date_window", "purchases")}


def test_one_day_after_end_fails():
    data = with_row(valid_statement(), 0, trans_date=PERIOD_END + timedelta(days=1))
    assert check_pairs(validate(data)) == {("date_window", "purchases")}


# --- printed totals ---------------------------------------------------------

@pytest.mark.parametrize("field", ["section_totals", "summary"])
def test_missing_printed_totals_is_rejected(field):
    data = replace(valid_statement(), **{field: None})
    assert checks(validate(data)) == ["no_printed_totals"]


def test_section_total_off_by_one_cent():
    data = with_row(valid_statement(), 1, amount=D("-25.51"))
    found = validate(data)
    assert ("section_total", "purchases") in check_pairs(found)
    mismatch = next(m for m in found if m.check == "section_total")
    assert (mismatch.expected, mismatch.actual) == ("-35.50", "-35.51")


def test_rows_without_a_printed_total_fail():
    data = valid_statement()
    totals = {k: v for k, v in data.section_totals.items() if k != "purchases"}
    found = validate(replace(data, section_totals=totals))
    assert ("section_total", "purchases") in check_pairs(found)


def test_nonzero_printed_total_without_rows_fails():
    data = valid_statement()
    rows = [r for r in data.rows if r.section != "payments"]
    found = validate(replace(data, rows=rows))
    assert ("section_total", "payments") in check_pairs(found)


# --- account summary --------------------------------------------------------

def test_summary_equation_off_by_one_cent():
    data = with_summary(valid_statement(), new_balance="438.74")
    assert check_pairs(validate(data)) == {("summary_equation", None)}


def test_summary_line_disagreeing_with_its_section_total():
    # Equation still balances, but the summary's purchases differ from the table's.
    data = with_summary(valid_statement(), purchases="35.51", new_balance="438.74")
    assert check_pairs(validate(data)) == {("summary_section", "purchases")}


def test_nonzero_cash_advances_without_a_section_fails_closed():
    data = with_summary(valid_statement(), cash_advances="5.00", new_balance="443.73")
    assert check_pairs(validate(data)) == {("summary_section", "cash_advances")}


def test_missing_new_balance_is_summary_incomplete_not_key_error():
    data = valid_statement()
    summary = {k: v for k, v in data.summary.items() if k != "new_balance"}
    assert "summary_incomplete" in checks(validate(replace(data, summary=summary)))


# --- reporting --------------------------------------------------------------

def test_all_mismatches_are_reported_not_just_the_first():
    data = with_row(valid_statement(), 0, trans_date=PERIOD_END + timedelta(days=1))
    data = with_row(data, 1, amount=D("-25.51"))
    data = with_summary(data, new_balance="1.00")
    assert {"date_window", "section_total", "summary_equation"} <= set(checks(validate(data)))


def test_reader_mismatches_come_first():
    reader_found = Mismatch("unknown_summary_label", None, "a profile label", "unmatched row")
    data = replace(with_summary(valid_statement(), new_balance="1.00"), reader_mismatches=[reader_found])
    found = validate(data)
    assert found[0] is reader_found
    assert "summary_equation" in checks(found)


def test_unknown_check_name_is_refused():
    with pytest.raises(ValueError):
        Mismatch("made_up_check", None, "x", "y")


def test_mismatch_serialises_with_money_as_strings():
    found = validate(with_row(valid_statement(), 1, amount=D("-25.51")))
    body = json.loads(json.dumps([m.to_dict() for m in found]))
    assert body[0]["expected"] == "-35.50"
    assert isinstance(body[0]["actual"], str)


def test_rejection_message_names_checks_but_no_amounts():
    found = validate(with_row(valid_statement(), 1, amount=D("-25.51")))
    message = str(StatementRejected(found))
    assert "section_total" in message
    assert "35.5" not in message


# --- malformed input becomes a mismatch, never an exception (review M2) ------

@pytest.mark.parametrize("changes, expected", [
    ({"section": None},                              ("row_invalid", None)),
    ({"section": ["purchases"]},                     ("row_invalid", None)),
    ({"amount": D("-10.004")},                       ("row_invalid", "purchases")),
    ({"amount": D("-100000000.00")},                 ("row_invalid", "purchases")),
    ({"amount": -10.0},                              ("row_invalid", "purchases")),
    ({"trans_date": datetime(2025, 12, 5, 9, 30)},   ("row_invalid", "purchases")),
])
def test_malformed_row_is_reported_not_raised(changes, expected):
    found = validate(with_row(valid_statement(), 0, **changes))
    assert expected in check_pairs(found)


def test_more_than_two_decimals_cannot_reconcile_by_rounding():
    # 10.004 + 25.496 = 35.50 would pass a sum check, but Numeric(10,2) would
    # store 10.00 and 25.50: the validated and stored values must not differ.
    data = with_row(with_row(valid_statement(), 0, amount=D("-10.004")), 1, amount=D("-25.496"))
    assert "row_invalid" in checks(validate(data))


@pytest.mark.parametrize("field", ["period_start", "period_end"])
def test_missing_period_is_reported_not_raised(field):
    assert checks(validate(replace(valid_statement(), **{field: None}))) == ["period_not_found"]


def test_reversed_period_is_reported():
    data = replace(valid_statement(), period_start=PERIOD_END, period_end=PERIOD_START)
    assert checks(validate(data)) == ["period_not_found"]


@pytest.mark.parametrize("bad_total", [Decimal("sNaN"), 35.5, None])
def test_unreadable_printed_total_is_reported_not_raised(bad_total):
    data = valid_statement()
    totals = dict(data.section_totals, purchases=bad_total)
    found = validate(replace(data, section_totals=totals))
    assert ("section_total", "purchases") in check_pairs(found)


def test_float_summary_value_is_reported_not_raised():
    data = valid_statement()
    summary = dict(data.summary, purchases=replace(data.summary["purchases"], value=35.5))
    assert "summary_incomplete" in checks(validate(replace(data, summary=summary)))


@pytest.mark.parametrize("key, sign", [("purchases", 0), ("purchases", 5), ("new_balance", 1), ("payments", True)])
def test_bad_summary_sign_is_reported(key, sign):
    data = valid_statement()
    summary = dict(data.summary, **{key: replace(data.summary[key], sign=sign)})
    assert "summary_incomplete" in checks(validate(replace(data, summary=summary)))


def test_summary_of_only_new_balance_does_not_pass():
    # Without a line per printed section the second check would check nothing.
    data = valid_statement()
    summary = {"new_balance": SummaryLine(D("0.00"), 0, None)}
    found = validate(replace(data, summary=summary))
    assert ("summary_incomplete", "purchases") in check_pairs(found)


def test_every_check_name_used_by_validate_is_in_checks():
    # Construction already refuses unknown names; this pins that the validator
    # only ever reaches for names that exist, across a spread of failures.
    broken = [
        replace(valid_statement(), rows=[]),
        with_row(valid_statement(), 0, trans_date=None, amount=Decimal("NaN")),
        with_summary(valid_statement(), new_balance="0"),
        replace(valid_statement(), summary=None),
        replace(valid_statement(), period_start=None),
    ]
    produced = {m.check for data in broken for m in validate(data)}
    assert produced <= CHECKS
    assert {"no_rows", "row_invalid", "summary_equation", "no_printed_totals", "period_not_found"} <= produced


# --- container shapes, descriptions, duplicate links (second review) --------

@pytest.mark.parametrize("changes, expected_check", [
    ({"rows": None},                                        "no_rows"),
    ({"rows": "not a list"},                                "no_rows"),
    ({"reader_mismatches": "oops"},                         "unexpected_error"),
    ({"reader_mismatches": [None]},                         "unexpected_error"),
    ({"section_totals": ["purchases"]},                     "no_printed_totals"),
    ({"summary": "previous 500"},                           "no_printed_totals"),
])
def test_malformed_containers_are_reported_not_raised(changes, expected_check):
    assert expected_check in checks(validate(replace(valid_statement(), **changes)))


def test_reader_mismatches_none_means_none_found():
    assert validate(replace(valid_statement(), reader_mismatches=None)) == []


def test_non_row_element_is_row_invalid():
    data = valid_statement()
    found = validate(replace(data, rows=list(data.rows) + [None]))
    assert ("row_invalid", None) in check_pairs(found)


def test_non_summary_line_value_is_summary_incomplete():
    data = valid_statement()
    summary = dict(data.summary, purchases=D("35.50"))
    assert "summary_incomplete" in checks(validate(replace(data, summary=summary)))


@pytest.mark.parametrize("description", [None, "", "   ", 42])
def test_missing_description_is_row_invalid(description):
    found = validate(with_row(valid_statement(), 0, description=description))
    assert ("row_invalid", "purchases") in check_pairs(found)


def test_two_summary_lines_for_one_section_are_rejected():
    # An extra opposite-sign line for payments must not get to decide direction.
    data = valid_statement()
    summary = dict(data.summary, payments_again=SummaryLine(D("100.00"), +1, "payments"))
    found = validate(replace(data, summary=summary))
    assert ("summary_incomplete", "payments") in check_pairs(found)


def test_flipped_section_with_negative_printed_values_fails_on_the_equation():
    # Purchases flipped to positive and the printed total and summary value made
    # negative to match: internally consistent, but the equation no longer
    # reaches the printed new balance.
    data = valid_statement()
    data = with_row(with_row(data, 0, amount=D("10.00")), 1, amount=D("25.50"))
    data = replace(data, section_totals=dict(data.section_totals, purchases=D("-35.50")))
    data = with_summary(data, purchases="-35.50")
    assert "summary_equation" in checks(validate(data))


GARBAGE = [None, "", "x", 0, -1, 1.5, Decimal("NaN"), Decimal("sNaN"), [], {}, object()]


@pytest.mark.parametrize("field", ["period_start", "period_end", "rows", "section_totals", "summary", "reader_mismatches"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_statement_fields(field, value):
    found = validate(replace(valid_statement(), **{field: value}))
    assert all(isinstance(m, Mismatch) for m in found)


@pytest.mark.parametrize("field", ["trans_date", "post_date", "description", "amount", "section"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_row_fields(field, value):
    found = validate(with_row(valid_statement(), 0, **{field: value}))
    assert all(isinstance(m, Mismatch) for m in found)


@pytest.mark.parametrize("field", ["value", "sign", "section"])
@pytest.mark.parametrize("value", GARBAGE)
def test_validate_never_raises_on_garbage_summary_fields(field, value):
    data = valid_statement()
    summary = dict(data.summary, purchases=replace(data.summary["purchases"], **{field: value}))
    found = validate(replace(data, summary=summary))
    assert all(isinstance(m, Mismatch) for m in found)
