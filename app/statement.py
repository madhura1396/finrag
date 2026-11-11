"""Bank-agnostic description of a parsed statement, and the checks it must pass
before anything is stored.

ADR 0002 Decisions 4 and 5, ADR 0003 decisions 4b, 5a and 5b.

This module knows nothing about PDFs, banks, profiles or the database. A reader
turns a PDF into a StatementData. validate() decides whether it can be trusted.
Nothing is written unless validate() returns no mismatches.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

# Transactions may be dated up to this long before the statement period starts:
# a purchase made just before the cycle closes can post into the next one.
DATE_LOOKBACK = timedelta(days=7)

# Canonical section keys. Profiles must use these, and the pipeline relies on
# them across banks (for example the fixed Fees/Interest categories).
SECTIONS = frozenset({
    "payments",
    "other_credits",
    "purchases",
    "cash_advances",
    "fees",
    "interest",
})

# The only words allowed to describe a failure. check_profiles.py prints these
# and nothing else, so a new failure mode must be added here first.
CHECKS = frozenset({
    "no_text_layer",
    "period_not_found",
    "period_mismatch",
    "header_not_found",
    "summary_not_found",
    "unknown_summary_label",
    "summary_incomplete",
    "row_invalid",
    "no_rows",
    "date_window",
    "no_printed_totals",
    "section_total",
    "summary_equation",
    "summary_section",
    "unexpected_error",
})

NEW_BALANCE = "new_balance"
REJECT_REASON = "statement did not reconcile"

# transactions.amount is Numeric(10, 2). An amount outside this range or with
# more than 2 decimal places would be rounded or refused at insert, so the value
# stored would differ from the value validated.
MAX_AMOUNT = Decimal("99999999.99")
MAX_DECIMAL_PLACES = 2


@dataclass(frozen=True)
class Mismatch:
    """One reason a statement cannot be trusted.

    expected and actual are strings so the 422 body serialises without a float
    ever touching money: str(Decimal) is exact.
    """

    check: str
    section: str | None
    expected: str
    actual: str

    def __post_init__(self) -> None:
        if self.check not in CHECKS:
            raise ValueError(f"unknown check name: {self.check!r}")

    def to_dict(self) -> dict:
        return {
            "check":    self.check,
            "section":  self.section,
            "expected": self.expected,
            "actual":   self.actual,
        }


@dataclass(frozen=True)
class Row:
    # Optional in the type so that validate(), not the constructor, rejects a
    # missing date: a reader bug becomes a 422 naming the check, not a crash.
    trans_date: date | None
    post_date: date | None
    description: str
    # Signed, from the cardholder's side: money back to you (payment, refund,
    # credit) is positive; money out (charge, fee, interest) is negative.
    # The sign is the only record of direction, so it cannot disagree with a flag.
    amount: Decimal
    section: str


@dataclass(frozen=True)
class SummaryLine:
    value: Decimal
    sign: int               # +1 or -1 for terms of the equation, 0 for new_balance
    section: str | None     # the section total this line must equal, if any


@dataclass(frozen=True)
class StatementData:
    period_start: date
    period_end: date
    rows: list[Row]
    section_totals: dict[str, Decimal] | None     # as printed, keyed by section
    summary: dict[str, SummaryLine] | None        # must contain "new_balance"
    reader_mismatches: list[Mismatch] = field(default_factory=list)


class StatementRejected(Exception):
    """The statement did not reconcile. Nothing was saved."""

    def __init__(self, mismatches: list[Mismatch]):
        self.mismatches = list(mismatches)
        # Check names only: this message can end up in logs, amounts must not.
        names = ", ".join(sorted({m.check for m in self.mismatches}))
        super().__init__(f"{REJECT_REASON}: {names}")


class DuplicateStatement(Exception):
    """A statement for the same period is already stored."""


def validate(data: StatementData) -> list[Mismatch]:
    """Return every reason the statement cannot be trusted. Empty means valid.

    All mismatches are reported, not just the first, so one upload attempt
    shows everything that is wrong.

    Malformed input of any shape (wrong types, None, NaN, unknown section keys,
    wrong container types) becomes a mismatch, never an exception: a reader bug
    must surface as a 422 naming the check, not as a 500.
    """
    mismatches = _reader_mismatches(data.reader_mismatches)
    rows = list(data.rows) if isinstance(data.rows, (list, tuple)) else []

    if not (_is_date(data.period_start) and _is_date(data.period_end)) \
            or data.period_start > data.period_end:
        mismatches.append(Mismatch("period_not_found", None, "a statement period", "none"))
        return mismatches

    mismatches += _check_rows(rows, data.period_start, data.period_end)

    if not isinstance(data.section_totals, dict) or not isinstance(data.summary, dict):
        mismatches.append(Mismatch("no_printed_totals", None, "printed totals", "none"))
        return mismatches

    mismatches += _check_summary(data.summary, data.section_totals)
    mismatches += _check_section_totals(rows, data.section_totals, _section_signs(data.summary))
    return mismatches


def _reader_mismatches(value) -> list[Mismatch]:
    """The reader's own findings, kept first. Anything that is not a Mismatch is
    a reader bug and must not be dropped silently, so it is reported as one."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return [Mismatch("unexpected_error", None, "a list of mismatches", type(value).__name__)]
    return [
        m if isinstance(m, Mismatch)
        else Mismatch("unexpected_error", None, "a mismatch", type(m).__name__)
        for m in value
    ]


def _is_date(value) -> bool:
    # Exact type: datetime is a date subclass, and comparing one with a date raises.
    return type(value) is date


def _is_section(value) -> bool:
    return isinstance(value, str) and value in SECTIONS


def _is_money(value) -> bool:
    """A finite Decimal that fits Numeric(10, 2) exactly. Sign is not checked here."""
    return (
        isinstance(value, Decimal)
        and value.is_finite()                       # before any comparison: NaN raises
        and value.as_tuple().exponent >= -MAX_DECIMAL_PLACES
        and abs(value) <= MAX_AMOUNT
    )


def _counts_toward_totals(row) -> bool:
    return isinstance(row, Row) and _is_section(row.section) and _is_money(row.amount)


def _check_rows(rows: list, period_start: date, period_end: date) -> list[Mismatch]:
    if not rows:
        return [Mismatch("no_rows", None, "at least 1 row", "0")]

    earliest = period_start - DATE_LOOKBACK
    window = f"{earliest}..{period_end}"
    out = []

    for row in rows:
        if not isinstance(row, Row):
            out.append(Mismatch("row_invalid", None, "a row", type(row).__name__))
            continue

        if not _is_section(row.section):
            out.append(Mismatch("row_invalid", None, "known section", str(row.section)))
            continue

        if not _is_money(row.amount):
            out.append(Mismatch("row_invalid", row.section, "an amount with 2 decimal places", str(row.amount)))

        # description becomes raw_description and merchant, both NOT NULL. An
        # empty one would be stored as a blank merchant.
        if not isinstance(row.description, str) or not row.description.strip():
            out.append(Mismatch("row_invalid", row.section, "a description", "none"))

        if row.trans_date is None:
            out.append(Mismatch("row_invalid", row.section, "a date", "none"))
        elif not _is_date(row.trans_date):
            out.append(Mismatch("row_invalid", row.section, "a date", str(row.trans_date)))
        elif not earliest <= row.trans_date <= period_end:
            out.append(Mismatch("date_window", row.section, window, str(row.trans_date)))

    return out


def _summary_lines(summary: dict) -> list[SummaryLine]:
    return [line for line in summary.values() if isinstance(line, SummaryLine)]


def _duplicate_links(summary: dict) -> set[str]:
    seen, duplicates = set(), set()
    for line in _summary_lines(summary):
        if _is_section(line.section):
            (duplicates if line.section in seen else seen).add(line.section)
    return duplicates


def _section_signs(summary: dict[str, SummaryLine]) -> dict[str, int]:
    """Which way each section's rows should point, read from the Account Summary.

    The summary is written from the balance's side ("- Payments", "+ Purchases"),
    rows from the cardholder's side, so a section's rows sum to minus its summary
    sign times its printed total: payments to +total, purchases to -total.
    """
    duplicates = _duplicate_links(summary)
    return {
        line.section: -line.sign
        for line in _summary_lines(summary)
        if _is_section(line.section)
        and line.section not in duplicates     # ambiguous direction: reported elsewhere
        and type(line.sign) is int and line.sign in (1, -1)
    }


def _check_section_totals(
    rows: list[Row], printed: dict[str, Decimal], signs: dict[str, int]
) -> list[Mismatch]:
    """Signed comparison: each section's signed row sum must equal its printed
    total with the direction from the summary. A single row with the wrong sign
    moves its section's sum by twice its amount, so it cannot pass. A section
    that mixes refunds and charges nets correctly; there, two wrong signs of
    equal size cancel out, which any total-based check shares.

    Direction comes from a liability (credit card) summary, where "- Payments"
    means payments reduce what you owe. An asset account (checking, savings)
    reverses that, so its first profile needs an explicit direction per section
    rather than this derivation (ADR 0003, decision 8).
    """
    # Rows already reported as row_invalid are left out of the sums, so one bad
    # row cannot crash the comparison below.
    sums: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in rows:
        if _counts_toward_totals(row):
            sums[row.section] += row.amount

    out = []
    for section, total in printed.items():
        if not _is_section(section):
            out.append(Mismatch("section_total", None, "known section", str(section)))
        elif not _is_money(total):
            out.append(Mismatch("section_total", section, "a printed amount", str(total)))

    for section in sorted(set(sums) | {s for s in printed if _is_section(s)}):
        parsed = sums.get(section, Decimal("0"))
        total = printed.get(section)

        if total is None:
            # Rows with nothing printed to check them against cannot be trusted.
            out.append(Mismatch("section_total", section, "a printed total", str(parsed)))
        elif not _is_money(total) or section not in signs:
            continue  # unreadable total or no summary direction: reported elsewhere
        else:
            expected = signs[section] * total
            # Exact Decimal comparison: no rounding, no tolerance. A printed total
            # with no rows lands here too unless it is zero.
            if parsed != expected:
                out.append(Mismatch("section_total", section, str(expected), str(parsed)))

    return out


def _check_summary(summary: dict[str, SummaryLine], printed: dict[str, Decimal]) -> list[Mismatch]:
    """The second, independent check (ADR 0002): the Account Summary equation,
    and every summary line agreeing with the table's printed section total.

    Requires a summary line for every section with a printed total, so the
    summary cannot be reduced to new_balance alone and still pass.
    """
    out = []
    usable = True

    if NEW_BALANCE not in summary:
        out.append(Mismatch("summary_incomplete", None, NEW_BALANCE, "missing"))
        usable = False

    for key, line in summary.items():
        if not isinstance(line, SummaryLine):
            out.append(Mismatch("summary_incomplete", None, "a readable summary line", str(key)))
            usable = False
            continue
        allowed_signs = (0,) if key == NEW_BALANCE else (1, -1)
        if not _is_money(line.value) or type(line.sign) is not int or line.sign not in allowed_signs:
            out.append(Mismatch("summary_incomplete", None, "a readable summary line", str(key)))
            usable = False
        if line.section is not None and not _is_section(line.section):
            out.append(Mismatch("summary_incomplete", None, "known section", str(line.section)))
            usable = False

    for section in sorted(_duplicate_links(summary)):
        out.append(Mismatch("summary_incomplete", section, "one summary line", "more than one"))
        usable = False

    linked = {line.section for line in _summary_lines(summary) if _is_section(line.section)}
    for section in sorted(s for s in printed if _is_section(s)):
        if section not in linked:
            out.append(Mismatch("summary_incomplete", section, "a summary line", "missing"))

    if not usable:
        return out

    computed = sum(
        (line.sign * line.value for key, line in summary.items() if key != NEW_BALANCE),
        Decimal("0"),
    )
    new_balance = summary[NEW_BALANCE].value
    if computed != new_balance:
        out.append(Mismatch("summary_equation", None, str(new_balance), str(computed)))

    for line in summary.values():
        if line.section is None:
            continue
        # A summary line whose section printed no total must be zero. This is how
        # an unexpected cash advance fails closed.
        expected = printed.get(line.section, Decimal("0"))
        if not _is_money(expected):
            continue  # already reported by _check_section_totals
        if line.value != expected:
            out.append(Mismatch("summary_section", line.section, str(expected), str(line.value)))

    return out
