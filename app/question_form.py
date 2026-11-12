"""Turn a question into a validated form (ADR 0004, details 1 and 2).

The model only copies words into fields; its output is constrained to a JSON
schema and then checked here. It never does date arithmetic or name
matching: code resolves dates against an injected `today`, and matches
merchants and trips against what is stored. Everything except ask_model() is
a pure function.
"""

import calendar
import difflib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import httpx

from app import read_settings as rs
from app.category_store import PLACEHOLDER_WORDS, normalise
from app.read_types import ALL, Form, Range, Refusal, Trip, month_range  # noqa: F401  (re-exported)

TYPES = (
    "total_spent", "total_received", "count", "ranking", "largest", "list", "compare",
    "monthly_trend", "category_change", "savings_opportunities", "anomalies", "search",
)
FORM_TYPES = TYPES + ("unsupported",)
DIRECTIONS = ("spent", "payments", "refunds", "received")
RELATIVE = ("this_month", "last_month", "last_3_months", "this_year", "last_year", "ytd", "all")
COMPARISON_TYPES = ("compare", "category_change")
MONTH_TYPES = ("anomalies", "savings_opportunities")


class ModelUnavailable(Exception):
    """The local model could not be reached in time. Maps to 503."""


class NotUnderstood(Exception):
    """The model's output was unusable. The question is refused and logged."""


# --------------------------------------------------------------------------
# The raw form: what the model is allowed to say
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RawForm:
    type: str
    direction: str | None = None
    group_by: str | None = None
    order: str | None = None
    limit: int | None = None
    category: str | None = None
    merchant: str | None = None
    trip: str | None = None
    search_text: str | None = None
    period: str | None = None
    period_2: str | None = None


def schema() -> dict:
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "type":        {"type": "string", "enum": list(FORM_TYPES)},
            "direction":   {"type": ["string", "null"], "enum": [*DIRECTIONS, None]},
            "group_by":    {"type": ["string", "null"], "enum": ["merchant", "category", None]},
            "order":       {"type": ["string", "null"], "enum": ["most", "least", None]},
            "limit":       {"type": ["integer", "null"]},
            "category":    nullable_str,
            "merchant":    nullable_str,
            "trip":        nullable_str,
            "search_text": nullable_str,
            "period":      nullable_str,
            "period_2":    nullable_str,
        },
        "required": ["type"],
    }


INSTRUCTIONS = f"""Fill a form describing a personal-finance question about card transactions.
Copy the user's words into fields; never calculate dates or numbers.
Fill a field ONLY if the question states it. Every other field is null.

type: one of {", ".join(FORM_TYPES)}.
  total_spent: how much was spent. total_received: payments, refunds or money back. count: how many.
  ranking: most/least by merchant or category, or a breakdown. largest: biggest single purchases.
  list: show the transactions. compare: two periods against each other. monthly_trend: month by month.
  category_change: which categories went up or down. savings_opportunities: where to cut back, subscriptions.
  anomalies: unusual or higher than normal. search: things that "look like" something.
  unsupported: anything else (advice, other accounts, actions, general knowledge). For unsupported, fill nothing else.
direction: payments (paying the card off), refunds (money back from shops), received (all money back). Null for spending.
group_by: merchant or category (ranking only). order: most or least (ranking only). limit: a number, only if said.
category: a kind of spending (groceries, dining, shopping...). merchant: a named shop or business. trip: a named trip.
search_text: for search only, what it should look like.
period: the time asked about. period_2: only when the question names a second period to compare against.
  "YYYY-MM" for a month with a year, "--MM" for a month without a year, "YYYY" for a whole year,
  "YYYY-MM-DD..YYYY-MM-DD" or "--MM-DD..--MM-DD" for a range, or one of {", ".join(RELATIVE)}.
"""


def _whole_words_of(text: str, name: str) -> bool:
    needle = normalise(text)
    return bool(needle) and f" {needle} " in f" {normalise(name)} "


def build_prompt(question: str, examples: list[dict]) -> str:
    shots = "\n".join(f"Q: {e['question']}\nA: {json.dumps(e['form'])}" for e in examples)
    return f"{INSTRUCTIONS}\nExamples:\n{shots}\n\nQ: {question}\nA:"


def parse_raw(data) -> RawForm:
    """Validate the model's output. Anything out of shape is NotUnderstood."""
    if not isinstance(data, dict) or data.get("type") not in FORM_TYPES:
        raise NotUnderstood("missing or unknown type")
    values = {}
    for name in RawForm.__dataclass_fields__:
        value = data.get(name)
        # A spelled-out null ("null", "n/a") is absent. An unfilled period template
        # ("--MM") is not: it stays a bad period.
        if name != "type" and isinstance(value, str) and value.strip().lower() in PLACEHOLDER_WORDS:
            value = None
        values[name] = value
    checks = {
        "direction": DIRECTIONS, "group_by": ("merchant", "category"), "order": ("most", "least"),
    }
    for name, allowed in checks.items():
        if values[name] is not None and values[name] not in allowed:
            raise NotUnderstood(f"{name} {values[name]!r} is not allowed")
    limit = values["limit"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or not 1 <= limit <= rs.MAX_LIMIT):
        raise NotUnderstood(f"limit must be 1 to {rs.MAX_LIMIT}")
    for name in ("category", "merchant", "trip", "search_text", "period", "period_2"):
        if values[name] is not None and (not isinstance(values[name], str) or len(values[name]) > 200):
            raise NotUnderstood(f"{name} is not a short string")
    return RawForm(**values)


def ask_model(question: str, examples: list[dict], post=httpx.post) -> RawForm:
    try:
        response = post(
            rs.OLLAMA_URL,
            json={"model": rs.MODEL, "prompt": build_prompt(question, examples), "stream": False,
                  "format": schema(), "options": {"temperature": 0}},
            timeout=rs.FORM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ModelUnavailable(str(e)) from e
    except httpx.HTTPStatusError as e:
        raise ModelUnavailable(f"model server returned {e.response.status_code}") from e
    try:
        return parse_raw(json.loads(response.json()["response"]))
    except (ValueError, KeyError, TypeError) as e:
        raise NotUnderstood(str(e)) from e


# --------------------------------------------------------------------------
# Resolution: from words to dates, names and defaults
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DataContext:
    today: date
    categories: list[str]                  # current, non-archived names
    merchants: list[str]                   # distinct stored merchant names
    trips: list[Trip]
    full_months: list[tuple[int, int]]     # fully covered calendar months, oldest first


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _recent(month: int, day: int, today: date) -> date:
    """The most recent occurrence of month/day that started on or before today.
    Looks back up to 8 years so a yearless 29 Feb finds the last leap year."""
    for year in range(today.year, today.year - 9, -1):
        try:
            candidate = date(year, month, day)
        except ValueError:          # 29 Feb in a non-leap year
            continue
        if candidate <= today:
            return candidate
    raise NotUnderstood(f"no recent {month:02d}-{day:02d}")


# Statements and questions live in a plausible window; anything else is a misread,
# and it keeps day-by-day coverage and "the same range before" arithmetic bounded.
MIN_YEAR, MAX_YEAR = 1990, 2100


def _date(year: int, month: int, day: int, text: str) -> date:
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise NotUnderstood(f"year in {text!r} is outside {MIN_YEAR}-{MAX_YEAR}")
    try:
        return date(year, month, day)
    except ValueError as e:
        raise NotUnderstood(f"bad date {text!r}") from e


def _parse_day(text: str, today: date) -> tuple[date, bool]:
    """(date, whether the user gave a year)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return _date(int(m[1]), int(m[2]), int(m[3]), text), True
    m = re.fullmatch(r"--(\d{2})-(\d{2})", text)
    if m:
        month, day = int(m[1]), int(m[2])
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            raise NotUnderstood(f"bad date {text!r}")
        return _recent(month, day, today), False
    raise NotUnderstood(f"bad date {text!r}")


def parse_period(text: str, today: date) -> Range:
    """Resolve one period string against today (ADR 0004, 2a and 2a′)."""
    text = text.strip()
    if text == "all":
        return Range(None, None, "all data")
    if text == "this_month":
        return Range(date(today.year, today.month, 1), today, f"this month (1 {calendar.month_abbr[today.month]} to today)")
    if text == "last_month":
        y, m = _shift_month(today.year, today.month, -1)
        return month_range(y, m)
    if text == "last_3_months":
        y1, m1 = _shift_month(today.year, today.month, -3)
        y3, m3 = _shift_month(today.year, today.month, -1)
        return Range(month_range(y1, m1).start, month_range(y3, m3).end, "the last 3 months")
    if text in ("this_year", "ytd"):
        return Range(date(today.year, 1, 1), today, f"year to date {today.year}")
    if text == "last_year":
        return Range(date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), str(today.year - 1))

    m = re.fullmatch(r"(\d{4})", text)
    if m:
        year = int(m[1])
        return Range(_date(year, 1, 1, text), _date(year, 12, 31, text), str(year))
    m = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if m:
        year, month = int(m[1]), int(m[2])
        if not 1 <= month <= 12:
            raise NotUnderstood(f"bad month {text!r}")
        _date(year, month, 1, text)
        return month_range(year, month)
    m = re.fullmatch(r"--(\d{2})", text)
    if m:
        month = int(m[1])
        if not 1 <= month <= 12:
            raise NotUnderstood(f"bad month {text!r}")
        start = _recent(month, 1, today)
        return month_range(start.year, month)
    if ".." in text:
        first, _, second = text.partition("..")
        (start, start_has_year), (end, end_has_year) = _parse_day(first.strip(), today), _parse_day(second.strip(), today)
        if end < start and not start_has_year:
            # Each yearless end resolves independently; a wrapping range rolls the
            # yearless start back a year (ADR 0004, 2a′). A year the user stated is never changed.
            start = _recent(start.month, start.day, date(start.year - 1, 12, 31))
            if end < start:
                # A yearless end on 29 Feb resolves to the last leap year: place the
                # yearless start in that year, or the one before if it wraps.
                for year in (end.year, end.year - 1):
                    try:
                        candidate = date(year, start.month, start.day)
                    except ValueError:
                        continue
                    if candidate <= end:
                        start = candidate
                        break
        if end < start:
            raise NotUnderstood(f"range {text!r} ends before it starts")
        if not start_has_year and not end_has_year and (end - start).days > 366:
            # A yearless range never spans more than a year. A 29 Feb start can look
            # back to a leap year several years ago (directly, or through the
            # rollback above): the end belongs to the start's year, or the next one.
            for year in (start.year, start.year + 1):
                try:
                    candidate = date(year, end.month, end.day)
                except ValueError:
                    continue
                if candidate >= start:
                    end = candidate
                    break
        if (end - start).days > 366 and not (start_has_year and end_has_year):
            raise NotUnderstood(f"range {text!r} cannot be placed within a year")
        if not (MIN_YEAR <= start.year and end.year <= MAX_YEAR):
            raise NotUnderstood(f"range {text!r} is outside {MIN_YEAR}-{MAX_YEAR}")
        return Range(start, end, f"{start} to {end}")
    raise NotUnderstood(f"bad period {text!r}")


def match_merchants(typed: str, names: list[str]) -> tuple[list[str], list[str]]:
    """(matched names, closest names if none matched). Contains with at least
    MERCHANT_MIN_CONTAINS characters, or >= MERCHANT_SIMILARITY similar to the
    whole name or any word in it (ADR 0004, 2b)."""
    needle = normalise(typed)
    matched = []
    for name in names:
        hay = normalise(name)
        if len(needle) >= rs.MERCHANT_MIN_CONTAINS and needle in hay:
            matched.append(name)
            continue
        candidates = [hay, *hay.split()]
        if any(difflib.SequenceMatcher(None, needle, c).ratio() >= rs.MERCHANT_SIMILARITY for c in candidates):
            matched.append(name)
    if matched:
        return sorted(set(matched)), []
    by_norm = {normalise(n): n for n in names}
    closest = difflib.get_close_matches(needle, list(by_norm), n=rs.CLOSEST_NAMES, cutoff=0.5)
    return [], [by_norm[c] for c in closest]


def match_trip(typed: str, trips: list[Trip]) -> Trip | Refusal:
    needle = normalise(typed)
    exact = [t for t in trips if normalise(t.name) == needle]
    found = exact or [t for t in trips
                      if len(needle) >= rs.MERCHANT_MIN_CONTAINS and needle in normalise(t.name)]
    if not found:
        return Refusal("data", f"no trip matching '{typed}'", {"trips": [t.name for t in trips]})
    if len(found) > 1:
        return Refusal("data", f"several trips match '{typed}'; name one", {"trips": [t.name for t in found]})
    trip = found[0]
    if trip.end < trip.start:
        return Refusal("data", f"trip '{trip.name}' ends before it starts")
    return trip


def _intersect(a: Range, start: date, end: date, label: str) -> Range | None:
    lo = max(a.start, start) if not a.is_all else start
    hi = min(a.end, end) if not a.is_all else end
    if hi < lo:
        return None
    return Range(lo, hi, label)


def _before(r: Range) -> Range:
    """The same-length range immediately before r."""
    end = r.start - timedelta(days=1)
    start = end - timedelta(days=r.days - 1)
    return Range(start, end, f"the {r.days} days before ({start} to {end})", said=False)


def _year_earlier(r: Range) -> Range:
    def back(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:          # 29 Feb
            return d.replace(year=d.year - 1, day=28)
    start, end = back(r.start), back(r.end)
    return Range(start, end, f"the same dates a year earlier ({start} to {end})", said=False)


# The fields each type actually applies. Anything else the model fills is
# dropped here, so the "Understood as" line only ever shows what was computed.
_FILTERS = {"direction", "category", "merchant", "trip", "period"}
APPLIED_FIELDS = {
    "total_spent":           _FILTERS,
    "total_received":        _FILTERS,
    "count":                 _FILTERS,
    "ranking":               _FILTERS | {"group_by", "order", "limit"},
    "largest":               _FILTERS | {"limit"},
    "list":                  _FILTERS,
    "compare":               _FILTERS | {"period_2"},
    "monthly_trend":         _FILTERS,
    "category_change":       {"direction", "merchant", "trip", "period", "period_2"},
    "savings_opportunities": {"period"},       # fixed rules over all spending
    "anomalies":             {"period"},       # fixed rule over all spending, by category
    "search":                _FILTERS | {"search_text"},
}


def _applied_only(raw: RawForm) -> RawForm:
    keep = APPLIED_FIELDS[raw.type]
    return replace(raw, **{name: None for name in RawForm.__dataclass_fields__
                           if name != "type" and name not in keep})


def _near_category(text: str, categories: list[str]) -> bool:
    words = [w for c in categories for w in [normalise(c), *normalise(c).split()]]
    return bool(difflib.get_close_matches(normalise(text), words, n=1, cutoff=rs.CATEGORY_NEAR))


def resolve(raw: RawForm, ctx: DataContext) -> Form | Refusal:
    if raw.search_text is not None and raw.type != "search":
        # The model misread the type; refuse as not understood, never guess search.
        raise NotUnderstood(f"search words on type {raw.type}")
    if raw.type == "unsupported":
        return Refusal("unsupported", "this kind of question is not supported yet", {"supported": list(TYPES)})
    raw = _applied_only(raw)
    if raw.period_2 and not raw.period:
        raise NotUnderstood("a second period without a first")

    direction = raw.direction or ("received" if raw.type == "total_received" else "spent")
    if raw.type == "total_received" and direction == "spent":
        direction = "received"
    form = Form(type=raw.type, direction=direction, group_by=raw.group_by or "merchant",
                order=raw.order or "most", limit=raw.limit or rs.DEFAULT_LIMIT)

    if raw.type == "search":
        if not raw.search_text or not raw.search_text.strip():
            raise NotUnderstood("search without search words")
        form = replace(form, search_text=raw.search_text.strip())

    if raw.category is not None:
        by_lower = {c.lower(): c for c in ctx.categories}
        name = by_lower.get(raw.category.strip().lower())
        if name is not None:
            form = replace(form, category=name)
        else:
            # "spend on <shop>": try whole words of a merchant name, unless it is
            # near a category name ("grocery" vs "Groceries").
            matched = []
            if raw.merchant is None and not _near_category(raw.category, ctx.categories):
                matched = sorted(m for m in ctx.merchants if _whole_words_of(raw.category, m))
            if not matched:
                return Refusal("data", f"no category named '{raw.category}'", {"categories": sorted(ctx.categories)})
            form = replace(form, merchant_text=raw.category, merchants=tuple(matched))

    if raw.merchant is not None:
        matched, closest = match_merchants(raw.merchant, ctx.merchants)
        if not matched:
            return Refusal("data", f"no merchant matching '{raw.merchant}'", {"closest": closest})
        form = replace(form, merchant_text=raw.merchant, merchants=tuple(matched))

    # Periods: said by the user, or a per-type default.
    period = parse_period(raw.period, ctx.today) if raw.period else None
    period_2 = parse_period(raw.period_2, ctx.today) if raw.period_2 else None

    if raw.trip is not None:
        trip = match_trip(raw.trip, ctx.trips)
        if isinstance(trip, Refusal):
            return trip
        base = period or Range(trip.start, trip.end, f"trip {trip.name}")
        period = _intersect(base, trip.start, trip.end, f"trip {trip.name} ({trip.start} to {trip.end})")
        if period is None:
            return Refusal("data", f"the period asked for does not overlap trip '{trip.name}'")
        form = replace(form, trip=trip.name)

    latest = [month_range(y, m, said=False) for y, m in ctx.full_months]
    if raw.type in COMPARISON_TYPES:
        if period is None:
            if len(latest) < 2:
                return Refusal("data", "comparing needs two fully covered months of statements",
                               {"full_months": len(latest)})
            period, period_2 = latest[-1], latest[-2]
        else:
            if period.is_all or (period_2 is not None and period_2.is_all):
                return Refusal("data", "a comparison needs specific periods, not all data")
            if period_2 is None:
                period_2 = _year_earlier(period) if raw.period in ("ytd", "this_year") else _before(period)
            if period.start <= period_2.end and period_2.start <= period.end:
                return Refusal("data", "the two periods overlap; compare periods that do not share days",
                               {"period": period.label, "period_2": period_2.label})
    elif raw.type in MONTH_TYPES:
        if period is None:
            if not latest:
                return Refusal("data", "no fully covered month of statements yet")
            period = latest[-1]
    if period is None:
        period = ALL

    return replace(form, period=period, period_2=period_2)
