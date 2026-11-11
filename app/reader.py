"""Read a statement PDF into StatementData by word positions, driven by a
layout profile (ADR 0002, decisions 2 and 3; plan PR2 "Reader rules").

No bank strings live here: every heading, label and pattern comes from the
profile. The reader never raises on a bad cell: it records a Mismatch naming
the check and section (never the cell text) and lets validate() report it.
It raises only when nothing can be validated at all (no text, no period).

Imports nothing that touches the database, so scripts/check_profiles.py can
use it without loading .env.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import fitz

from app.profile import Profile
from app.statement import (
    DATE_LOOKBACK, Mismatch, Row, StatementData, StatementRejected, SummaryLine,
)

_MONEY = re.compile(r"^(-?)\$?(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})$")
_MONTH_DAY = re.compile(r"^(\d{1,2})/(\d{1,2})$")
_SIGNS = {"+", "-", "=", "–", "−"}


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    text: str

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    page: int
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _norm(text: str) -> str:
    return " ".join(text.split()).upper()


def _money(text: str) -> Decimal | None:
    m = _MONEY.match(text.strip())
    if not m:
        return None
    try:
        value = Decimal(f"{m[2].replace(',', '')}.{m[3]}")
    except InvalidOperation:
        return None
    return -value if m[1] else value


# --------------------------------------------------------------------------
# Words into lines
# --------------------------------------------------------------------------

def _lines(doc: fitz.Document, tolerance: float) -> list[list[Line]]:
    pages = []
    for number, page in enumerate(doc):
        words = sorted((Word(w[0], w[1], w[2], w[4]) for w in page.get_text("words")), key=lambda w: (w.y0, w.x0))
        lines: list[Line] = []
        for word in words:
            if lines and abs(word.y0 - lines[-1].words[0].y0) <= tolerance:
                lines[-1].words.append(word)
            else:
                lines.append(Line(number, [word]))
        for line in lines:
            line.words.sort(key=lambda w: w.x0)
        pages.append(lines)
    return pages


def _find_phrase(words: list[Word], phrase: str, start: int = 0) -> tuple[int, int] | None:
    """Index range of consecutive words spelling `phrase` (case- and space-insensitive)."""
    target = _norm(phrase).split()
    for i in range(start, len(words) - len(target) + 1):
        if [_norm(w.text) for w in words[i:i + len(target)]] == target:
            return i, i + len(target)
    return None


# --------------------------------------------------------------------------
# Period
# --------------------------------------------------------------------------

def _period(pages: list[list[Line]], profile: Profile, fmt: str) -> tuple[date, date, list[Mismatch]]:
    found = []
    for lines in pages:
        page_text = "\n".join(line.text for line in lines)
        m = profile.period_pattern.search(page_text)
        if m:
            try:
                found.append((datetime.strptime(m[1], fmt).date(), datetime.strptime(m[2], fmt).date()))
            except ValueError:
                found.append(None)
    found_ok = [p for p in found if p is not None]
    if not found_ok:
        raise StatementRejected([Mismatch("period_not_found", None, "a statement period", "none")])
    mismatches = []
    if len(set(found_ok)) > 1 or len(found_ok) != len(found):
        mismatches.append(Mismatch("period_mismatch", None, "the same period on every page", "differs"))
    start, end = found_ok[0]
    if end < start:
        raise StatementRejected([Mismatch("period_not_found", None, "a statement period", "reversed")])
    return start, end, mismatches


def resolve_year(month: int, day: int, start: date, end: date) -> date | None:
    """The date for MM/DD inside [start - 7 days, end], trying the period's years.
    Never strptime('%m/%d'), which rejects 02/29 without a year."""
    for year in sorted({start.year, end.year}):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if start - DATE_LOOKBACK <= candidate <= end:
            return candidate
    return None


# --------------------------------------------------------------------------
# Account summary
# --------------------------------------------------------------------------

def _strip_signs(words: list[Word]) -> list[Word]:
    out = list(words)
    while out and (out[0].text in _SIGNS):
        out = out[1:]
    if out and out[0].text[:1] in _SIGNS and len(out[0].text) > 1 and not _money(out[0].text):
        out[0] = Word(out[0].x0, out[0].y0, out[0].x1, out[0].text[1:])
    return out


def _summary(pages: list[list[Line]], profile: Profile) -> tuple[dict[str, SummaryLine] | None, list[Mismatch]]:
    not_found = [Mismatch("summary_not_found", None, "the account summary", "missing")]
    start_label, end_label = _norm(profile.summary_start), _norm(profile.summary_end)
    for lines in pages:
        starts = [i for i, line in enumerate(lines)
                  if _norm(" ".join(w.text for w in _strip_signs(line.words))).startswith(start_label)]
        if not starts:
            continue
        first = starts[0]
        last = next((i for i in range(first, len(lines))
                     if _norm(" ".join(w.text for w in _strip_signs(lines[i].words))).startswith(end_label)), None)
        anchor_x = None
        for line in lines:
            span = _find_phrase(line.words, profile.summary_anchor)
            if span:
                anchor_x = line.words[span[0]].x0
                break
        if last is None or anchor_x is None:
            return None, not_found

        labels = {_norm(l.label): l for l in profile.summary_lines}
        values: dict[str, SummaryLine] = {}
        mismatches: list[Mismatch] = []
        carry: list[str] = []
        for line in lines[first:last + 1]:
            left = _strip_signs([w for w in line.words if w.x1 < anchor_x])
            if not left:
                continue
            amount = _money(left[-1].text)
            label_words = [w.text for w in (left[:-1] if amount is not None else left)]
            if amount is None:
                carry.extend(label_words)           # a label wrapped onto the next line
                continue
            label = _norm(" ".join(carry + label_words))
            carry = []
            spec = labels.get(label)
            if spec is None:
                mismatches.append(Mismatch("unknown_summary_label", None, "a profile summary label", "unmatched"))
                continue
            values[spec.key] = SummaryLine(amount, spec.sign, spec.section)
        for spec in profile.summary_lines:
            if spec.key not in values:
                mismatches.append(Mismatch("summary_incomplete", spec.section, "a summary line", "missing"))
        return values, mismatches
    return None, not_found


# --------------------------------------------------------------------------
# Transaction table
# --------------------------------------------------------------------------

@dataclass
class _Columns:
    bounds: list[tuple[str, float, float]]   # (header label, left, right)

    def of(self, word: Word) -> str | None:
        for label, left, right in self.bounds:
            if left <= word.center < right:
                return label
        return None


def _header(line: Line, profile: Profile) -> _Columns | None:
    spans, cursor = [], 0
    for label in profile.header:
        span = _find_phrase(line.words, label, cursor)
        if span is None:
            return None
        spans.append((label, line.words[span[0]].x0, line.words[span[1] - 1].x1))
        cursor = span[1]
    bounds = []
    for i, (label, x0, x1) in enumerate(spans):
        left = float("-inf") if i == 0 else (spans[i - 1][2] + x0) / 2
        right = float("inf") if i == len(spans) - 1 else (x1 + spans[i + 1][1]) / 2
        bounds.append((label, left, right))
    return _Columns(bounds)


@dataclass
class _Pending:
    date: date
    post_date: date | None
    parts: list[str]
    amount: Decimal
    section: str


@dataclass
class _Table:
    rows: list[_Pending] = field(default_factory=list)
    totals: dict[str, Decimal] = field(default_factory=dict)
    mismatches: list[Mismatch] = field(default_factory=list)


def _table(pages: list[list[Line]], profile: Profile, start: date, end: date) -> _Table:
    out = _Table()
    headings = {_norm(s.heading): s for s in profile.sections}
    cols = profile.columns
    section = None
    last_row: _Pending | None = None
    last_y = last_page = None
    for lines in pages:
        columns = None                       # a page is outside the table until its header row
        gaps = sorted(b.words[0].y0 - a.words[0].y0 for a, b in zip(lines, lines[1:]))
        gap = 1.5 * gaps[len(gaps) // 2] if gaps else 0       # a wrap sits about one line below its row
        for line in lines:
            header = _header(line, profile)
            if header is not None:
                columns = header
                continue
            if columns is None:
                continue
            text = _norm(line.text)
            # Labels are matched without money words, which can sit inside a long label.
            label_text = _norm(" ".join(w.text for w in line.words if _money(w.text) is None))
            if label_text in headings:
                section, last_row = headings[label_text].key, None
                continue
            total_for = next((s for s in profile.sections if s.total.search(label_text)), None)
            if total_for is not None:
                amounts = [_money(w.text) for w in line.words if _money(w.text) is not None]
                if amounts:
                    out.totals[total_for.key] = abs(amounts[-1])
                else:
                    out.mismatches.append(Mismatch("row_invalid", total_for.key, "a printed total", "unreadable"))
                section, last_row = None, None    # a section closes at its total line
                continue
            if section is None:
                continue                          # e.g. the year-to-date box after the last total

            cells: dict[str, list[str]] = {}
            amount_labels = (cols["credit"], cols["debit"])
            other_columns = [w for w in line.words if columns.of(w) not in
                             (cols["description"], cols["credit"], cols["debit"], cols["trans_date"], cols["post_date"])]
            for word in line.words:
                label = columns.of(word)
                if label in amount_labels and _money(word.text) is None:
                    # Amount columns hold only money; other words there are a long
                    # description running past its column.
                    label = cols["description"]
                if label is not None:
                    cells.setdefault(label, []).append(word.text)
            date_text = " ".join(cells.get(cols["trans_date"], []))
            post_text = " ".join(cells.get(cols["post_date"], []))
            description = cells.get(cols["description"], [])
            credit_text = " ".join(cells.get(cols["credit"], []))
            debit_text = " ".join(cells.get(cols["debit"], []))

            date_words = [w for w in line.words if _MONTH_DAY.match(w.text)]
            if not date_text and not credit_text and not debit_text and not date_words:
                # A wrapped description: only description words, directly under the
                # row it continues. Anything else (a page footer, a note) is not a row.
                close = last_y is not None and line.page == last_page and line.words[0].y0 - last_y <= gap
                if description and last_row is not None and close and not other_columns:
                    last_row.parts.extend(description)
                    last_y = line.words[0].y0
                continue
            if any(columns.of(w) not in (cols["trans_date"], cols["post_date"]) for w in date_words):
                # A date-shaped word outside the date columns means the columns are
                # shifted; storing the wrong cell as the date would be silent.
                out.mismatches.append(Mismatch("row_invalid", section, "dates in the date columns", "shifted"))
                last_row = None
                continue

            if bool(credit_text) == bool(debit_text):
                out.mismatches.append(Mismatch("row_invalid", section, "one amount, as a credit or a charge",
                                               "both" if credit_text else "none"))
                last_row = None
                continue
            amount = _money(credit_text or debit_text)
            if amount is None or amount < 0:
                out.mismatches.append(Mismatch("row_invalid", section, "a readable amount", "unreadable"))
                last_row = None
                continue
            amount = amount if credit_text else -amount

            if not date_text:
                if section not in profile.undated_rows or post_text:
                    out.mismatches.append(Mismatch("row_invalid", section, "a date", "none"))
                    last_row = None
                    continue
                when = end                        # printed without a date (ADR 0003, 4a)
            else:
                m = _MONTH_DAY.match(date_text)
                if not m:
                    out.mismatches.append(Mismatch("row_invalid", section, "a readable date", "unreadable"))
                    last_row = None
                    continue
                when = resolve_year(int(m[1]), int(m[2]), start, end)
                if when is None:
                    out.mismatches.append(Mismatch("date_window", section, f"{start - DATE_LOOKBACK}..{end}",
                                                   "outside"))
                    last_row = None
                    continue
            post = None
            pm = _MONTH_DAY.match(post_text) if post_text else None
            if pm:
                post = resolve_year(int(pm[1]), int(pm[2]), start, end + timedelta(days=31))
            last_row = _Pending(when, post, list(description), amount, section)
            last_y, last_page = line.words[0].y0, line.page
            out.rows.append(last_row)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def read_statement(pdf_path: Path | str, profile: Profile) -> StatementData:
    try:
        doc = fitz.open(str(pdf_path))
    except (fitz.FileDataError, RuntimeError, ValueError) as e:
        raise StatementRejected([Mismatch("no_text_layer", None, "a readable PDF", "unreadable")]) from e
    try:
        pages = _lines(doc, profile.y_tolerance)
    except (RuntimeError, ValueError) as e:      # e.g. an encrypted PDF
        raise StatementRejected([Mismatch("no_text_layer", None, "a readable PDF", "unreadable")]) from e
    finally:
        doc.close()
    if not any(pages):
        raise StatementRejected([Mismatch("no_text_layer", None, "extractable text", "none")])

    start, end, mismatches = _period(pages, profile, profile.date_format)
    summary, summary_mismatches = _summary(pages, profile)
    table = _table(pages, profile, start, end)
    if not any(_header(line, profile) for lines in pages for line in lines):
        mismatches.append(Mismatch("header_not_found", None, "the transaction table header", "missing"))

    rows = [Row(p.date, p.post_date, " ".join(p.parts), p.amount, p.section) for p in table.rows]
    return StatementData(
        period_start=start, period_end=end, rows=rows,
        section_totals=table.totals or None, summary=summary,
        reader_mismatches=mismatches + summary_mismatches + table.mismatches,
    )
