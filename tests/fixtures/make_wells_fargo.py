"""Synthetic Wells Fargo-layout credit card statements (plan PR2). Invented
merchants, references and amounts only; PDFs are written to a temp directory
at test time and never committed.

The layout follows the facts in ADR 0003 Context: a 7-column transaction
table with a repeated header on continuation pages, five sections each closed
by a `TOTAL ... FOR THIS PERIOD` line, a two-column Account Summary with a
wrapped purchases label, a year-to-date box, the period line on every page,
and an unnumbered terms page. It encodes a guess about the real layout; only
scripts/check_profiles.py on real statements is evidence.
"""

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import fitz

D = Decimal
PAGE_W, PAGE_H = 612, 792
FONT_SIZE = 8
LINE = 12
COLS = {"card": 36, "trans": 92, "post": 140, "ref": 188, "desc": 290, "credit": 470, "debit": 530}
HEADER = [("card", "Card Ending in"), ("trans", "Trans Date"), ("post", "Post Date"),
          ("ref", "Reference Number"), ("desc", "Description"), ("credit", "Credits"), ("debit", "Charges")]
SECTIONS = [
    ("payments", "Payments", "TOTAL PAYMENTS FOR THIS PERIOD"),
    ("other_credits", "Other Credits", "TOTAL OTHER CREDITS FOR THIS PERIOD"),
    ("purchases", "Purchases, Balance Transfers & Other Charges",
     "TOTAL PURCHASES, BALANCE TRANSFERS & OTHER CHARGES FOR THIS PERIOD"),
    ("fees", "Fees Charged", "TOTAL FEES CHARGED FOR THIS PERIOD"),
    ("interest", "Interest Charged", "TOTAL INTEREST CHARGED FOR THIS PERIOD"),
]
CREDIT_SECTIONS = {"payments", "other_credits"}


@dataclass
class Txn:
    section: str
    trans_date: date | None          # None: printed without a date (interest, fees)
    description: str | list[str]     # a list wraps over several lines
    amount: Decimal                  # signed: credits positive, charges negative
    card: str | None = "9999"        # None: blank card cell (payments)
    raw_date: str | None = None      # print this instead of the date (broken cells)
    raw_amount: str | None = None    # print this instead of the amount


@dataclass
class Spec:
    period_start: date
    period_end: date
    rows: list[Txn]
    previous_balance: Decimal = D("100.00")
    rows_per_page: int = 14                      # table lines per page; small values force continuation
    totals: dict = field(default_factory=dict)   # section -> Decimal override, or None to omit the line
    summary: dict = field(default_factory=dict)  # summary key -> Decimal override
    include_summary: bool = True
    include_anchor: bool = True                  # the right column's first label
    extra_summary_label: str | None = None       # an unknown summary line
    page2_period: tuple[date, date] | None = None
    cash_advances: Decimal = D("0.00")
    data_offset: dict = field(default_factory=dict)   # column -> x shift for data cells only (misalignment)


def money(amount: Decimal) -> str:
    return f"{abs(amount):,.2f}"


def section_total(spec: Spec, key: str) -> Decimal:
    return sum((abs(r.amount) for r in spec.rows if r.section == key), D("0.00"))


def summary_values(spec: Spec) -> dict[str, Decimal]:
    t = {key: section_total(spec, key) for key, _, _ in SECTIONS}
    values = {
        "previous_balance": spec.previous_balance, "payments": t["payments"], "other_credits": t["other_credits"],
        "cash_advances": spec.cash_advances, "purchases": t["purchases"], "fees": t["fees"],
        "interest": t["interest"],
    }
    values["new_balance"] = (values["previous_balance"] - values["payments"] - values["other_credits"]
                             + values["cash_advances"] + values["purchases"] + values["fees"] + values["interest"])
    values.update(spec.summary)
    return values


def expected_rows(spec: Spec) -> list[tuple]:
    """Ground truth: (date, description, signed amount, section) per row, in print order."""
    out = []
    for key, _, _ in SECTIONS:
        for r in spec.rows:
            if r.section == key:
                desc = " ".join(r.description) if isinstance(r.description, list) else r.description
                out.append((r.trans_date or spec.period_end, desc, r.amount, r.section))
    return out


class _Writer:
    def __init__(self, doc: fitz.Document, spec: Spec):
        self.doc, self.spec = doc, spec
        self.page = None
        self.y = 0
        self.table_lines = 0

    def text(self, x: float, s: str) -> None:
        self.page.insert_text((x, self.y), s, fontsize=FONT_SIZE, fontname="helv")

    def new_page(self, period: tuple[date, date] | None, with_header: bool) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = 40
        start, end = period or (self.spec.period_start, self.spec.period_end)
        self.text(36, f"Statement Period {start:%m/%d/%Y} to {end:%m/%d/%Y}")
        self.y += 2 * LINE
        self.table_lines = 0
        if with_header:
            self.header()

    def header(self) -> None:
        for col, label in HEADER:
            self.text(COLS[col], label)
        self.y += LINE

    def table_line(self) -> None:
        """Advance one table line, continuing on a new page with the header repeated."""
        if self.table_lines >= self.spec.rows_per_page:
            self.new_page(self.spec.page2_period, with_header=True)
        self.table_lines += 1

    def summary(self) -> None:
        values = summary_values(self.spec)
        right = [("Total Credit Limit", "5,000.00"), ("Cash Advance Limit", "1,000.00"),
                 ("Total Available Credit", "4,000.00"), ("Available for Cash Advances", "1,000.00")]
        if not self.spec.include_anchor:
            right = right[1:]
        left = [("", "Previous Balance", "previous_balance"), ("-", "Payments", "payments"),
                ("-", "Other Credits", "other_credits"), ("+", "Cash Advances", "cash_advances"),
                ("+", "Purchases, Balance Transfers &", None), ("", "Other Charges", "purchases"),
                ("+", "Fees Charged", "fees"), ("+", "Interest Charged", "interest")]
        if self.spec.extra_summary_label:
            left.append(("+", self.spec.extra_summary_label, "__extra"))
        left.append(("=", "New Balance", "new_balance"))
        self.text(36, "Account Summary")
        self.y += LINE
        for i, (sign, label, key) in enumerate(left):
            if sign:
                self.text(36, sign)
            self.text(46, label)
            if key == "__extra":
                self.text(230, "$1.00")
            elif key:
                self.text(230, "$" + money(values[key]))
            if i < len(right):
                self.text(330, right[i][0])
                self.text(500, "$" + right[i][1])
            self.y += LINE
        self.y += LINE

    def section(self, key: str, heading: str, total_label: str) -> None:
        self.table_line()
        self.text(COLS["desc"], heading)
        self.y += LINE
        for r in [r for r in self.spec.rows if r.section == key]:
            lines = r.description if isinstance(r.description, list) else [r.description]
            if self.table_lines + len(lines) > self.spec.rows_per_page:
                self.table_lines = self.spec.rows_per_page   # keep a row's lines on one page
            self.table_line()
            x = {c: COLS[c] + self.spec.data_offset.get(c, 0) for c in COLS}
            if r.card:
                self.text(x["card"], r.card)
            if r.raw_date is not None:
                self.text(x["trans"], r.raw_date)
            elif r.trans_date is not None:
                self.text(x["trans"], f"{r.trans_date:%m/%d}")
                self.text(x["post"], f"{r.trans_date:%m/%d}")
                self.text(x["ref"], f"REF{abs(hash((r.description if isinstance(r.description, str) else lines[0], r.trans_date))) % 10**12:012d}")
            self.text(COLS["desc"], lines[0])
            amount_text = r.raw_amount if r.raw_amount is not None else money(r.amount)
            self.text(COLS["credit"] if key in CREDIT_SECTIONS else COLS["debit"], amount_text)
            self.y += LINE
            for extra in lines[1:]:
                self.table_line()
                self.text(COLS["desc"], extra)
                self.y += LINE
        override = self.spec.totals.get(key, "computed")
        if override is not None:
            amount = section_total(self.spec, key) if override == "computed" else override
            self.table_line()
            self.text(COLS["desc"] - 140, total_label)
            self.text(COLS["debit"] + 40, "$" + money(amount))
            self.y += LINE


def make_statement(path: Path, spec: Spec) -> list[tuple]:
    """Write the PDF and return its ground-truth rows."""
    doc = fitz.open()
    w = _Writer(doc, spec)
    w.new_page(None, with_header=False)
    if spec.include_summary:
        w.summary()
    w.header()
    for key, heading, total_label in SECTIONS:
        w.section(key, heading, total_label)
    # Year-to-date box: must never be read as a section total or as rows.
    year = spec.period_end.year
    for line in (f"{year} Totals Year-to-Date", f"TOTAL FEES CHARGED IN {year}    $77.77",
                 f"TOTAL INTEREST CHARGED IN {year}    $99.99"):
        w.table_line()
        w.text(COLS["desc"], line)
        w.y += LINE
    # An unnumbered terms page with no period line and no table.
    terms = doc.new_page(width=PAGE_W, height=PAGE_H)
    terms.insert_text((36, 60), "Important information about your account. Payments are applied as described.",
                      fontsize=FONT_SIZE, fontname="helv")
    doc.save(str(path))
    doc.close()
    return expected_rows(spec)


def standard_spec(**changes) -> Spec:
    """A reconciling statement exercising every layout feature."""
    start, end = date(2026, 9, 3), date(2026, 10, 2)
    rows = [
        Txn("payments", date(2026, 9, 10), "ONLINE PAYMENT THANK YOU", D("250.00"), card=None),
        Txn("other_credits", date(2026, 9, 14), "MEGASTORE RETURN", D("19.99")),
        Txn("purchases", date(2026, 9, 1), "GROCER MART #101 SPRINGFIELD", D("-45.20")),   # 2 days before start
        Txn("purchases", date(2026, 9, 5), ["FOODDASH*TACO TOWN", "SPRINGFIELD ORDER 7781"], D("-23.75")),
        Txn("purchases", date(2026, 9, 12), "CORNER CAFE 4455 RIVERTON", D("-8.40")),
        Txn("purchases", date(2026, 9, 20), "SKYWAY AIRLINES TICKET", D("-1,234.56".replace(",", ""))),
        Txn("purchases", date(2026, 9, 28), "CITY PARKING METERS", D("-3.00")),
        Txn("interest", None, "INTEREST CHARGE ON PURCHASES", D("-12.34")),
        Txn("interest", None, "INTEREST CHARGE ON CASH ADVANCES", D("0.00")),
    ]
    spec = Spec(start, end, rows, rows_per_page=10)
    return replace(spec, **changes)
