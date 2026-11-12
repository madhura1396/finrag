"""Every number on the read path, computed from rows and statement periods
(ADR 0004, "Testing the numbers"). Pure: no database, no model, Decimal only.

Directions on signed rows (ADR 0004, Definitions 1):
    spent     amount < 0                       (purchases, cash advances, fees, interest)
    payments  amount > 0, section payments
    refunds   amount > 0, section not payments (Other Credits, refunds inside a charge section)
    received  amount > 0
0.00 rows are never counted; they appear only in `list`. Spending is reported
positive. Rows whose category is pending stay in every total that has no
category filter and are bucketed as "Uncategorised (pending)" (ADR 0005).
"""

import calendar
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from app import read_config as rs
from app.embedding_text import is_stale
from app.read_config import Form, Range, Refusal, month_range

ZERO = Decimal("0")
CENT = Decimal("0.01")


@dataclass(frozen=True)
class RowView:
    id: int
    date: date
    amount: Decimal
    section: str
    category: str | None            # current name; None while pending
    category_source: str
    merchant: str
    raw_description: str
    needs_review: bool
    embedded_text: str | None
    embedding: object | None        # only presence matters here


@dataclass(frozen=True)
class Coverage:
    status: str                     # 'full', 'partial' or 'none'
    covered: tuple[date, date] | None
    covered_days: int
    total_days: int


@dataclass
class Result:
    type: str
    facts: dict = field(default_factory=dict)
    table: list[dict] | None = None
    rows: list[RowView] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)        # label -> Coverage
    notes: dict = field(default_factory=dict)


def cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------
# Coverage from statement periods
# --------------------------------------------------------------------------

def covered_days(periods: list[tuple[date, date]]) -> set[date]:
    days = set()
    for start, end in periods:
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days


def data_span(periods) -> tuple[date, date] | None:
    if not periods:
        return None
    return min(s for s, _ in periods), max(e for _, e in periods)


def coverage(r: Range, periods) -> Coverage:
    days = covered_days(periods)
    if r.is_all:
        span = data_span(periods)
        if span is None:
            return Coverage("none", None, 0, 0)
        return Coverage("full", span, len(days), len(days))
    wanted = [r.start + timedelta(days=i) for i in range(r.days)]
    hit = [d for d in wanted if d in days]
    if not hit:
        return Coverage("none", None, 0, len(wanted))
    status = "full" if len(hit) == len(wanted) else "partial"
    return Coverage(status, (min(hit), max(hit)), len(hit), len(wanted))


def full_months(periods) -> list[tuple[int, int]]:
    days = covered_days(periods)
    if not days:
        return []
    months = sorted({(d.year, d.month) for d in days})
    return [(y, m) for y, m in months
            if all(date(y, m, day) in days for day in range(1, calendar.monthrange(y, m)[1] + 1))]


# --------------------------------------------------------------------------
# Directions and filters
# --------------------------------------------------------------------------

def direction_params(direction: str) -> dict:
    """The direction definitions, as parameters for the search query, so they
    live only here."""
    return {
        "spent":    {"sign": "negative", "only_sections": None, "except_sections": None},
        "payments": {"sign": "positive", "only_sections": ["payments"], "except_sections": None},
        "refunds":  {"sign": "positive", "only_sections": None, "except_sections": ["payments"]},
        "received": {"sign": "positive", "only_sections": None, "except_sections": None},
    }[direction]


def in_direction(row: RowView, direction: str, *, include_zero: bool = False) -> bool:
    if direction == "spent":
        return row.amount < 0 or (include_zero and row.amount == 0)
    if row.amount <= 0:
        return False
    if direction == "payments":
        return row.section == "payments"
    if direction == "refunds":
        return row.section != "payments"
    return True


def value(row: RowView, direction: str) -> Decimal:
    """Positive money in the direction's own terms: spend as a positive number."""
    return -row.amount if direction == "spent" else row.amount


def in_range(row: RowView, r: Range) -> bool:
    return r.is_all or r.start <= row.date <= r.end


def category_of(row: RowView) -> str:
    return row.category if row.category is not None else rs.PENDING_BUCKET


def matching(rows, form: Form, r: Range, direction: str | None = None, *, include_zero=False) -> list[RowView]:
    direction = direction or form.direction
    merchants = set(form.merchants)
    out = []
    for row in rows:
        if not in_range(row, r) or not in_direction(row, direction, include_zero=include_zero):
            continue
        if form.category is not None and row.category != form.category:
            continue        # a pending row has no category, so it never matches a category filter
        if merchants and row.merchant not in merchants:
            continue
        out.append(row)
    return out


def total(rows, direction: str) -> Decimal:
    return sum((value(r, direction) for r in rows), ZERO)


def notes(rows, r: Range, names_for_stale: dict | None = None, search: bool = False) -> dict:
    in_period = [row for row in rows if in_range(row, r)]
    out = {
        "needs_review": sum(1 for row in in_period if row.needs_review),
        "pending": sum(1 for row in in_period if row.category is None),
    }
    if search:
        out["stale_embeddings"] = sum(1 for row in in_period if row.embedding is not None
                                      and is_stale(row, row.category))
        out["no_embedding"] = sum(1 for row in in_period if row.embedding is None)
    return out


# --------------------------------------------------------------------------
# The twelve calculations
# --------------------------------------------------------------------------

def _require_full(r: Range, periods, label: str) -> Coverage | Refusal:
    cov = coverage(r, periods)
    if cov.status != "full":
        fm = full_months(periods)
        suggestion = None
        if len(fm) >= 2:
            a, b = month_range(*fm[-1]), month_range(*fm[-2])
            suggestion = f"compare {a.label} with {b.label}"
        return Refusal("data", f"{label} ({r.label}) is not fully covered by your statements",
                       {"coverage": cov.status, "covered": [str(d) for d in cov.covered] if cov.covered else None,
                        "suggestion": suggestion})
    return cov


def calc_total(form: Form, rows, periods) -> Result:
    selected = matching(rows, form, form.period)
    facts = {"total": total(selected, form.direction), "count": len(selected), "direction": form.direction}
    if form.direction == "spent":
        refunds = matching(rows, form, form.period, "refunds")
        if refunds:
            facts["refunds"] = total(refunds, "refunds")
            facts["refund_count"] = len(refunds)
    return Result(form.type, facts, rows=selected)


def calc_count(form: Form, rows, periods) -> Result:
    selected = matching(rows, form, form.period)
    return Result("count", {"count": len(selected), "direction": form.direction}, rows=selected)


def calc_ranking(form: Form, rows, periods) -> Result:
    selected = matching(rows, form, form.period)
    groups: dict[str, list[RowView]] = defaultdict(list)
    for row in selected:
        groups[row.merchant if form.group_by == "merchant" else category_of(row)].append(row)
    ranked = [(name, total(g, form.direction), len(g)) for name, g in groups.items()]
    ranked = [x for x in ranked if x[1] > 0]                      # "least" counts only places with spend
    ranked.sort(key=lambda x: (-x[1], x[0]) if form.order == "most" else (x[1], x[0]))
    table = [{"name": n, "amount": a, "count": c} for n, a, c in ranked[:form.limit]]
    return Result("ranking", {"group_by": form.group_by, "order": form.order, "groups": len(ranked)},
                  table=table, rows=selected)


def calc_largest(form: Form, rows, periods) -> Result:
    selected = sorted(matching(rows, form, form.period), key=lambda r: (-value(r, form.direction), r.date, r.id))
    top = selected[:form.limit]
    table = [{"date": r.date, "merchant": r.merchant, "amount": value(r, form.direction),
              "category": category_of(r)} for r in top]
    return Result("largest", {"count": len(top)}, table=table, rows=top)


def calc_list(form: Form, rows, periods) -> Result:
    selected = sorted(matching(rows, form, form.period, include_zero=True), key=lambda r: (r.date, r.id))
    return Result("list", {"rows_total": len(selected)}, rows=selected[:rs.LIST_CAP])


def _per_day(amount: Decimal, days: int) -> Decimal:
    return cents(amount / days)


def _direction_word(a: Decimal, b: Decimal) -> str:
    return "up" if a > b else "down" if a < b else "same"


def calc_compare(form: Form, rows, periods) -> Result | Refusal:
    for r, label in ((form.period, "the first period"), (form.period_2, "the second period")):
        checked = _require_full(r, periods, label)
        if isinstance(checked, Refusal):
            return checked
    a_rows, b_rows = matching(rows, form, form.period), matching(rows, form, form.period_2)
    a, b = total(a_rows, form.direction), total(b_rows, form.direction)
    facts = {"a_total": a, "b_total": b, "a_days": form.period.days, "b_days": form.period_2.days,
             "direction": form.direction}
    if form.period.days != form.period_2.days:
        facts["a_per_day"], facts["b_per_day"] = _per_day(a, form.period.days), _per_day(b, form.period_2.days)
        facts["basis"] = "per_day"
        facts["change"] = _direction_word(facts["a_per_day"], facts["b_per_day"])
    else:
        facts["basis"] = "total"
        facts["change"] = _direction_word(a, b)
        facts["difference"] = abs(a - b)
    return Result("compare", facts, rows=a_rows + b_rows)


def _by_category(rows, form: Form, r: Range) -> dict[str, Decimal]:
    out: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for row in matching(rows, form, r):
        out[category_of(row)] += value(row, form.direction)
    return out


def calc_category_change(form: Form, rows, periods) -> Result | Refusal:
    for r, label in ((form.period, "the period"), (form.period_2, "the period compared against")):
        checked = _require_full(r, periods, label)
        if isinstance(checked, Refusal):
            return checked
    a, b = _by_category(rows, form, form.period), _by_category(rows, form, form.period_2)
    per_day = form.period.days != form.period_2.days
    table = []
    for name in sorted(set(a) | set(b)):
        now, before = a.get(name, ZERO), b.get(name, ZERO)
        entry = {"category": name, "now": now, "before": before}
        if per_day:
            # Lengths differ: both totals are shown, and the change is per day (Definitions 2).
            entry["now_per_day"] = _per_day(now, form.period.days)
            entry["before_per_day"] = _per_day(before, form.period_2.days)
            entry["change"] = entry["now_per_day"] - entry["before_per_day"]
        else:
            entry["change"] = now - before
        table.append(entry)
    table.sort(key=lambda t: (t["change"], t["category"]))         # largest decrease first
    return Result("category_change", {"basis": "per_day" if per_day else "total"}, table=table)


def _month_key(d: date) -> tuple[int, int]:
    return d.year, d.month


def _whole_month_asked(ym: tuple[int, int], r: Range) -> bool:
    """False when the asked range clips the month (an explicit range, this_year)."""
    month = month_range(*ym)
    return r.is_all or (r.start <= month.start and r.end >= month.end)


def calc_monthly_trend(form: Form, rows, periods) -> Result | Refusal:
    full = set(full_months(periods))
    days = covered_days(periods)
    selected = matching(rows, form, form.period)
    # Months come from coverage and from the rows themselves: a row dated just
    # before its statement starts still counts in its own month (ADR 0002, 4).
    months = sorted({_month_key(d) for d in days if form.period.is_all or form.period.start <= d <= form.period.end}
                    | {_month_key(r.date) for r in selected})
    if not months:
        return Refusal("data", "no statement covers that period")
    spend = defaultdict(lambda: ZERO)
    for row in selected:
        spend[_month_key(row.date)] += value(row, form.direction)
    table, previous = [], None
    for ym in months:
        whole = ym in full and _whole_month_asked(ym, form.period)
        entry = {"month": month_range(*ym).label, "amount": spend[ym], "partial": not whole, "change": None}
        # Up or down only between two whole, consecutive calendar months.
        if whole and previous is not None and previous[0] == _shift(ym, -1) and previous[1]:
            entry["change"] = _direction_word(spend[ym], spend[previous[0]])
        previous = (ym, whole)
        table.append(entry)
    return Result("monthly_trend", {"months": len(table)}, table=table)


def _shift(ym: tuple[int, int], delta: int) -> tuple[int, int]:
    index = ym[0] * 12 + ym[1] - 1 + delta
    return index // 12, index % 12 + 1


def _target_month(form: Form, periods) -> tuple[tuple[int, int], list[tuple[int, int]]] | Refusal:
    r = form.period
    if r.is_all or (r.start.year, r.start.month) != (r.end.year, r.end.month) or r.start.day != 1 \
            or r.end.day != calendar.monthrange(r.end.year, r.end.month)[1]:
        return Refusal("data", "this works on one calendar month; name a month")
    target = (r.start.year, r.start.month)
    fm = full_months(periods)
    if target not in fm:
        return Refusal("data", f"{r.label} is not fully covered by your statements")
    return target, [m for m in fm if m < target]


def calc_anomalies(form: Form, rows, periods) -> Result | Refusal:
    found = _target_month(form, periods)
    if isinstance(found, Refusal):
        return found
    target, history = found
    if len(history) < rs.ANOMALY_MIN_HISTORY_MONTHS:
        return Refusal("data", f"not enough history (have {len(history)} full months before "
                               f"{month_range(*target).label}, need {rs.ANOMALY_MIN_HISTORY_MONTHS})")
    spend = defaultdict(lambda: defaultdict(lambda: ZERO))     # category -> month -> spend
    for row in rows:
        if in_direction(row, "spent") and row.category is not None:
            spend[row.category][_month_key(row.date)] += -row.amount
    table = []
    for name, by_month in sorted(spend.items()):
        now = by_month.get(target, ZERO)
        median = cents(Decimal(statistics.median([by_month.get(m, ZERO) for m in history])))
        if now > rs.ANOMALY_RATIO * median and now - median >= rs.ANOMALY_MIN_EXCESS:
            table.append({"category": name, "amount": now, "median": median, "excess": now - median})
    table.sort(key=lambda t: (-t["excess"], t["category"]))
    return Result("anomalies", {"month": month_range(*target).label, "history_months": len(history)}, table=table)


def calc_savings(form: Form, rows, periods) -> Result | Refusal:
    found = _target_month(form, periods)
    if isinstance(found, Refusal):
        return found
    target, history = found
    window = (history + [target])[-rs.RECURRING_WINDOW_MONTHS:]
    spent = [r for r in rows if in_direction(r, "spent")]

    by_merchant = defaultdict(lambda: defaultdict(lambda: ZERO))
    for row in spent:
        if _month_key(row.date) in window:
            by_merchant[row.merchant][_month_key(row.date)] += -row.amount
    recurring = []
    if len(window) >= rs.RECURRING_MIN_MONTHS:
        for name, months in sorted(by_merchant.items()):
            if len(months) >= rs.RECURRING_MIN_MONTHS:
                recurring.append({"merchant": name, "months": len(months),
                                  "monthly": cents(sum(months.values(), ZERO) / len(months))})
        recurring.sort(key=lambda t: (-t["monthly"], t["merchant"]))

    by_category = defaultdict(lambda: defaultdict(lambda: ZERO))
    for row in spent:
        if row.category is not None:
            by_category[row.category][_month_key(row.date)] += -row.amount
    above = []
    if history:
        for name, months in sorted(by_category.items()):
            average = cents(sum((months.get(m, ZERO) for m in history), ZERO) / len(history))
            now = months.get(target, ZERO)
            if now > average:
                above.append({"category": name, "amount": now, "average": average, "excess": now - average})
        above.sort(key=lambda t: (-t["excess"], t["category"]))

    fees_interest = sum((-r.amount for r in spent if _month_key(r.date) == target
                         and r.section in ("fees", "interest")), ZERO)
    return Result("savings_opportunities", {
        "month": month_range(*target).label, "recurring": recurring, "above_average": above,
        "fees_and_interest": fees_interest, "history_months": len(history),
        "recurring_checked": len(window) >= rs.RECURRING_MIN_MONTHS,
        "average_checked": bool(history),
    })


def calc_search(form: Form, matches: list[RowView]) -> Result:
    """Category, merchant, date and direction were applied in SQL before the cap
    (read_queries.search), so every match here counts."""
    assert rs.SEARCH_CAP <= rs.LIST_CAP, "search cap must not exceed the list cap"
    return Result("search", {"count": len(matches), "total": total(matches, form.direction),
                             "direction": form.direction, "capped": len(matches) >= rs.SEARCH_CAP},
                  rows=matches[:rs.LIST_CAP])


CALCULATIONS = {
    "total_spent": calc_total, "total_received": calc_total, "count": calc_count, "ranking": calc_ranking,
    "largest": calc_largest, "list": calc_list, "compare": calc_compare, "monthly_trend": calc_monthly_trend,
    "category_change": calc_category_change, "savings_opportunities": calc_savings, "anomalies": calc_anomalies,
}


def compute(form: Form, rows: list[RowView], periods: list[tuple[date, date]],
            search_matches: list[RowView] | None = None) -> Result | Refusal:
    if form.type == "search":
        result = calc_search(form, search_matches or [])
    else:
        result = CALCULATIONS[form.type](form, rows, periods)
    if isinstance(result, Refusal):
        return result
    result.coverage = {"period": coverage(form.period, periods)}
    if form.period_2 is not None:
        result.coverage["period_2"] = coverage(form.period_2, periods)
    result.notes = notes(rows, form.period, search=form.type == "search")
    if form.period_2 is not None:
        second = notes(rows, form.period_2)
        result.notes = {k: v + second.get(k, 0) for k, v in result.notes.items()}
    if form.type in ("anomalies", "savings_opportunities"):
        # Their history reaches back before the asked month: count it all.
        history = Range(None, None, "history") if form.period.is_all else Range(date.min, form.period.end, "history")
        result.notes = notes(rows, history)
    return result
