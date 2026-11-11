"""Turn a computed result into what the user reads (ADR 0004, details 3 and 4).

Code writes every fact as a complete phrase, including its verb and
direction, behind an opaque token. The exact answer joins those phrases.
The model may then arrange the tokens into a natural sentence, which is
accepted only if every token appears exactly once and the glue between them
has no digits and no word from the closed list. Tables are never rewritten.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import httpx

from app import read_settings as rs
from app.calculations import Coverage, Result
from app.read_types import Form, Range

DIRECTION_NOUNS = {"spent": "charges", "payments": "payments", "refunds": "refunds", "received": "credits"}
TYPE_WORDS = {
    "total_spent": "total spent", "total_received": "total received", "count": "count", "ranking": "ranking",
    "largest": "largest", "list": "list", "compare": "comparison", "monthly_trend": "monthly trend",
    "category_change": "change by category", "savings_opportunities": "where to cut back",
    "anomalies": "unusual categories", "search": "similarity search",
}
SENTENCE_TYPES = ("total_spent", "total_received", "count", "compare", "search")


# --------------------------------------------------------------------------
# Formatting: fixed by code
# --------------------------------------------------------------------------

def fmt_money(amount: Decimal) -> str:
    return f"${abs(amount).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,.2f}"


def fmt_date(d: date) -> str:
    return f"{d.day} {calendar.month_abbr[d.month]} {d.year}"


def fmt_pct(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def fmt_range(r: Range) -> str:
    if r.is_all:
        return "all your data"
    s, e = r.start, r.end
    month_end = calendar.monthrange(e.year, e.month)[1]
    if (s.year, s.month) == (e.year, e.month):
        if s.day == 1 and e.day == month_end:
            return f"{calendar.month_abbr[s.month]} {s.year}"
        return f"{s.day}–{e.day} {calendar.month_abbr[s.month]} {s.year}"
    if s.year == e.year:
        return f"{s.day} {calendar.month_abbr[s.month]} – {fmt_date(e)}"
    return f"{fmt_date(s)} – {fmt_date(e)}"


def fmt_coverage(cov: Coverage, span: tuple[date, date] | None) -> str:
    if cov.status == "full":
        return "fully covered"
    if cov.status == "partial":
        return f"partly covered ({fmt_date(cov.covered[0])} to {fmt_date(cov.covered[1])})"
    if span is None:
        return "no statements uploaded yet"
    return f"no data for this period (your statements cover {fmt_date(span[0])} to {fmt_date(span[1])})"


# --------------------------------------------------------------------------
# The interpretation line: always shown, never rewritten
# --------------------------------------------------------------------------

def understood_as(form: Form, result_coverage: dict, span) -> str:
    parts = [TYPE_WORDS[form.type], DIRECTION_NOUNS[form.direction]]
    if form.category:
        parts.append(f"category {form.category}")
    if form.merchants:
        parts.append(f"merchants {', '.join(form.merchants)} (matched \"{form.merchant_text}\")")
    if form.trip:
        parts.append(f"trip {form.trip}")
    if form.search_text:
        parts.append(f"looks like \"{form.search_text}\"")
    period = fmt_range(form.period)
    if not form.period.said:
        period += " (default)"
    if "period" in result_coverage:
        period += f", {fmt_coverage(result_coverage['period'], span)}"
    parts.append(period)
    if form.period_2 is not None:
        second = f"against {fmt_range(form.period_2)}"
        if "period_2" in result_coverage:
            second += f", {fmt_coverage(result_coverage['period_2'], span)}"
        parts.append(second)
    return "Understood as: " + " · ".join(parts)


# --------------------------------------------------------------------------
# Facts and the exact answer
# --------------------------------------------------------------------------

@dataclass
class Answer:
    exact: str
    facts: list[str]            # each one complete phrase, verb included; empty for table answers
    table: list[dict] | None
    result_facts: dict | None = None   # the computed facts, for caveats added after a rewrite


def _scope(form: Form) -> str:
    if form.category:
        return f" on {form.category}"
    if form.merchants:
        return f" at {', '.join(form.merchants)}"
    return ""


def _when(r: Range) -> str:
    return "" if r.is_all else f" in {fmt_range(r)}"


def _amount_phrase(form: Form, amount: Decimal, r: Range) -> str:
    money = fmt_money(amount)
    if form.direction == "spent":
        return f"you spent {money}{_scope(form)}{_when(r)}"
    if form.direction == "payments":
        return f"you paid {money} toward your card{_scope(form)}{_when(r)}"
    if form.direction == "refunds":
        return f"you got {money} back in refunds{_scope(form)}{_when(r)}"
    return f"{money} came back to your card{_scope(form)}{_when(r)}"


def _sentence(facts: list[str]) -> str:
    text = ", and ".join(facts) if len(facts) <= 2 else ", ".join(facts[:-1]) + ", and " + facts[-1]
    return text[:1].upper() + text[1:] + "."


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun if n != 1 else noun.rstrip('s')}"


def render(form: Form, result: Result, span) -> Answer:
    f = result.facts
    t = form.type
    if t in ("total_spent", "total_received"):
        facts = [_amount_phrase(form, f["total"], form.period),
                 f"that came from {_plural(f['count'], DIRECTION_NOUNS[form.direction])}"]
        if "refunds" in f:
            facts.append(f"you also got {fmt_money(f['refunds'])} back in refunds")
        return Answer(_sentence(facts), facts, None)
    if t == "count":
        fact = f"you had {_plural(f['count'], DIRECTION_NOUNS[form.direction])}{_scope(form)}{_when(form.period)}"
        return Answer(_sentence([fact]), [fact], None)
    if t == "compare":
        a = _amount_phrase(form, f["a_total"], form.period)
        b = f"{fmt_money(f['b_total'])} in {fmt_range(form.period_2)}"
        if f["basis"] == "per_day":
            c = (f"per day that is {fmt_money(f['a_per_day'])} against {fmt_money(f['b_per_day'])}, "
                 f"{'up' if f['change'] == 'up' else 'down' if f['change'] == 'down' else 'the same'}")
        elif f["change"] == "same":
            c = "the two are the same"
        else:
            c = f"that is {f['change']} {fmt_money(f['difference'])}"
        facts = [a, f"compared with {b}", c]
        return Answer(_sentence(facts), facts, None)
    if t == "search":
        noun = DIRECTION_NOUNS[form.direction]
        verb = "looks" if f["count"] == 1 else "look"
        facts = [f"I found {_plural(f['count'], noun)} that {verb} like \"{form.search_text}\"{_when(form.period)}",
                 f"{'it comes' if f['count'] == 1 else 'together they come'} to {fmt_money(f['total'])}"]
        exact = _sentence(facts) + " " + search_caveat(f)
        return Answer(exact, facts, None, f)

    # Table answers: a fixed summary line, never rewritten.
    when = fmt_range(form.period)
    if t == "ranking":
        what = "merchants" if form.group_by == "merchant" else "categories"
        head = f"{'Top' if form.order == 'most' else 'Lowest'} {len(result.table)} {what} by {DIRECTION_NOUNS[form.direction]}, {when}:"
        table = [{"name": r["name"], "amount": fmt_money(r["amount"]), "count": r["count"]} for r in result.table]
    elif t == "largest":
        head = f"Your {len(result.table)} largest {DIRECTION_NOUNS[form.direction]}, {when}:"
        table = [{"date": fmt_date(r["date"]), "merchant": r["merchant"], "amount": fmt_money(r["amount"]),
                  "category": r["category"]} for r in result.table]
    elif t == "list":
        shown, total = len(result.rows), f["rows_total"]
        head = f"{total} matching transactions, {when}" + (f" (showing the first {shown}):" if shown < total else ":")
        table = None
    elif t == "monthly_trend":
        head = f"{DIRECTION_NOUNS[form.direction].capitalize()} by month, {when} (partial months are not compared):"
        table = [{"month": r["month"], "amount": fmt_money(r["amount"]),
                  "partial": r["partial"], "change": r["change"]} for r in result.table]
    elif t == "category_change":
        basis = " per day" if f["basis"] == "per_day" else ""
        head = f"Change by category{basis}, {fmt_range(form.period)} against {fmt_range(form.period_2)}, largest decrease first:"
        table = []
        for r in result.table:
            entry = {"category": r["category"], "now": fmt_money(r["now"]), "before": fmt_money(r["before"])}
            if "now_per_day" in r:
                entry["now_per_day"] = fmt_money(r["now_per_day"])
                entry["before_per_day"] = fmt_money(r["before_per_day"])
            entry["change" + ("_per_day" if "now_per_day" in r else "")] = (
                ("-" if r["change"] < 0 else "+" if r["change"] > 0 else "") + fmt_money(r["change"]))
            table.append(entry)
    elif t == "anomalies":
        if not result.table:
            head = f"No category in {f['month']} is more than {rs.ANOMALY_RATIO}x its usual month and {fmt_money(rs.ANOMALY_MIN_EXCESS)} above it."
        else:
            head = f"Unusual categories in {f['month']}, against the median of {f['history_months']} earlier months:"
        table = [{"category": r["category"], "amount": fmt_money(r["amount"]), "usual": fmt_money(r["median"]),
                  "above_usual": fmt_money(r["excess"])} for r in result.table]
    else:   # savings_opportunities
        head = f"Where you could cut back, {f['month']}:"
        table = ([{"kind": "recurring", "name": r["merchant"], "monthly": fmt_money(r["monthly"]),
                   "months": r["months"]} for r in f["recurring"]]
                 + [{"kind": "above_average", "name": r["category"], "amount": fmt_money(r["amount"]),
                     "average": fmt_money(r["average"]), "excess": fmt_money(r["excess"])} for r in f["above_average"]]
                 + [{"kind": "fees_and_interest", "name": "Fees and interest", "amount": fmt_money(f["fees_and_interest"])}])
        if not f["recurring_checked"]:
            head += f" (recurring charges need {rs.RECURRING_MIN_MONTHS} full months of history)"
        if not f["average_checked"]:
            head += " (categories above their usual need at least one earlier full month)"
    return Answer(head, [], table)


def search_caveat(facts: dict) -> str:
    text = "This is a similarity search, so it may miss some or include near-misses."
    if facts.get("capped"):
        text += f" Only the {rs.SEARCH_CAP} closest matches are counted; there may be more."
    return text


def answer_facts_of(form: Form, answer: "Answer") -> dict:
    return getattr(answer, "result_facts", {}) or {}


# --------------------------------------------------------------------------
# The placeholder rewrite
# --------------------------------------------------------------------------

TOKEN = "[F{}]"
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def check_rewrite(text: str, token_count: int) -> bool:
    """Deterministic acceptance test for a rewrite (ADR 0004, detail 3)."""
    if not isinstance(text, str) or not text.strip() or len(text) > rs.REWRITE_MAX_CHARS:
        return False
    tokens = [TOKEN.format(i + 1) for i in range(token_count)]
    if any(text.count(t) != 1 for t in tokens):
        return False
    glue = text
    for t in tokens:
        glue = glue.replace(t, " ")
    if re.search(r"\[F\d+\]", glue) or re.search(r"\d", glue):
        return False
    lowered = glue.lower().replace("’", "'")
    if "n't" in lowered:
        return False
    return not any(word in rs.GLUE_BANNED_WORDS for word in _WORD.findall(lowered))


def rewrite_prompt(question_type: str, token_count: int) -> str:
    tokens = ", ".join(TOKEN.format(i + 1) for i in range(token_count))
    example = ", and ".join(TOKEN.format(i + 1) for i in range(token_count))
    return (f"You are joining hidden clauses into one friendly sentence. The placeholders {tokens} each stand "
            "for a complete clause that already has its own subject, verb, amounts and dates, for example "
            "'you bought something'. You cannot see them, so do not describe or guess their content. "
            "Use each placeholder exactly once, unchanged, in order, connected only by short neutral words such "
            "as 'Here is what I found:', 'and', 'also'. Do not add any other words about money, amounts, dates, "
            "rates, comparisons, quantities, negation or judgement. "
            f"Example: 'Here is what I found: {example}.' Reply with the sentence only.")


def call_model(prompt: str, post=httpx.post) -> str:
    response = post(rs.OLLAMA_URL,
                    json={"model": rs.MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                    timeout=rs.REWRITE_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["response"].strip()


def rewrite(form: Form, answer: Answer, call=call_model) -> tuple[str, str]:
    """(answer text, 'accepted' | 'rejected' | 'skipped')."""
    if form.type not in SENTENCE_TYPES or not answer.facts:
        return answer.exact, "skipped"
    try:
        text = call(rewrite_prompt(form.type, len(answer.facts)))
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return answer.exact, "skipped"
    if not check_rewrite(text, len(answer.facts)):
        return answer.exact, "rejected"
    # One pass: a fact that itself contains "[F2]" (from a search text) is never re-expanded.
    text = re.sub(r"\[F(\d+)\]", lambda m: answer.facts[int(m[1]) - 1], text).strip()
    text = text[:1].upper() + text[1:]
    if form.type == "search":
        text += " " + search_caveat(answer_facts_of(form, answer))
    return text, "accepted"
