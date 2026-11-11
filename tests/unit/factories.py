"""Synthetic StatementData for unit tests. Invented values, no real data."""

from dataclasses import replace
from datetime import date
from decimal import Decimal

from app.statement import Row, StatementData, SummaryLine

PERIOD_START = date(2025, 12, 3)
PERIOD_END = date(2026, 1, 2)


def D(value: str) -> Decimal:
    return Decimal(value)


def valid_statement() -> StatementData:
    """A statement that reconciles: every section sum matches its printed total
    and the summary equation holds.

    previous 500.00 - payments 100.00 - other credits 0.00 + cash advances 0.00
    + purchases 35.50 + fees 2.00 + interest 1.23 = new balance 438.73

    Amounts are signed from the cardholder's side: payments positive, charges
    negative. Printed totals and summary values are positive, as printed.
    Other Credits prints a $0.00 total with no rows; interest includes a 0.00 row.
    """
    rows = [
        Row(date(2025, 12, 5),  date(2025, 12, 6),  "COFFEE SHOP A",                    D("-10.00"),  "purchases"),
        Row(date(2025, 12, 20), date(2025, 12, 21), "BOOKSTORE C",                      D("-25.50"),  "purchases"),
        Row(date(2025, 12, 10), date(2025, 12, 10), "ONLINE PAYMENT THANK YOU",         D("100.00"),  "payments"),
        Row(PERIOD_END,         None,                "LATE FEE",                         D("-2.00"),   "fees"),
        Row(PERIOD_END,         None,                "INTEREST CHARGE ON PURCHASES",     D("-1.23"),   "interest"),
        Row(PERIOD_END,         None,                "INTEREST CHARGE ON CASH ADVANCES", D("0.00"),    "interest"),
    ]
    section_totals = {
        "payments":      D("100.00"),
        "other_credits": D("0.00"),
        "purchases":     D("35.50"),
        "fees":          D("2.00"),
        "interest":      D("1.23"),
    }
    summary = {
        "previous_balance": SummaryLine(D("500.00"), +1, None),
        "payments":         SummaryLine(D("100.00"), -1, "payments"),
        "other_credits":    SummaryLine(D("0.00"),   -1, "other_credits"),
        "cash_advances":    SummaryLine(D("0.00"),   +1, "cash_advances"),
        "purchases":        SummaryLine(D("35.50"),  +1, "purchases"),
        "fees":             SummaryLine(D("2.00"),   +1, "fees"),
        "interest":         SummaryLine(D("1.23"),   +1, "interest"),
        "new_balance":      SummaryLine(D("438.73"),  0, None),
    }
    return StatementData(PERIOD_START, PERIOD_END, rows, section_totals, summary)


def with_row(data: StatementData, index: int, **changes) -> StatementData:
    rows = list(data.rows)
    rows[index] = replace(rows[index], **changes)
    return replace(data, rows=rows)


def with_summary(data: StatementData, **values: str) -> StatementData:
    summary = dict(data.summary)
    for key, value in values.items():
        summary[key] = replace(summary[key], value=D(value))
    return replace(data, summary=summary)
