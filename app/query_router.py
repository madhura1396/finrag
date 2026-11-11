"""The read path (ADR 0004): orchestration only.

    question -> form (model, validated) -> rows (thin SQL) -> numbers (pure
    Python) -> exact answer (code) -> optional placeholder rewrite (model, checked)

The model never writes SQL and never states a number.
"""

import logging
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app import answers, calculations, category_store, read_queries
from app import read_settings as rs
from app.question_form import (
    DataContext, Form, ModelUnavailable, NotUnderstood, Refusal, TYPES, ask_model, resolve,
)

log = logging.getLogger("finrag.read")

# Few-shot examples for the form. Kept apart from the eval's test questions.
TEACH_EXAMPLES = [
    {"question": "How much did I spend on travel in March 2026?",
     "form": {"type": "total_spent", "category": "travel", "period": "2026-03"}},
    {"question": "What did I spend at Sunny Bakery in 2025?",
     "form": {"type": "total_spent", "merchant": "sunny bakery", "period": "2025"}},
    {"question": "How much did I pay toward my card in June?",
     "form": {"type": "total_received", "direction": "payments", "period": "--06"}},
    {"question": "How many purchases did I make at Pet Palace?",
     "form": {"type": "count", "merchant": "pet palace"}},
    {"question": "Where did I spend the least last month?",
     "form": {"type": "ranking", "order": "least", "group_by": "merchant", "period": "last_month"}},
    {"question": "Top 5 spending categories this year",
     "form": {"type": "ranking", "group_by": "category", "order": "most", "limit": 5, "period": "this_year"}},
    {"question": "My biggest travel purchases",
     "form": {"type": "largest", "category": "travel"}},
    {"question": "Show my Pet Palace transactions from June 1 to June 15",
     "form": {"type": "list", "merchant": "pet palace", "period": "--06-01..--06-15"}},
    {"question": "Did I spend more in August than in July?",
     "form": {"type": "compare", "period": "--08", "period_2": "--07"}},
    {"question": "Did I spend more from May 5 to July 5 than before?",
     "form": {"type": "compare", "period": "--05-05..--07-05"}},
    {"question": "Monthly spending on travel", "form": {"type": "monthly_trend", "category": "travel"}},
    {"question": "Which categories went down this month?",
     "form": {"type": "category_change", "period": "this_month"}},
    {"question": "Where am I wasting money?", "form": {"type": "savings_opportunities"}},
    {"question": "Was any category unusually high in June 2026?",
     "form": {"type": "anomalies", "period": "2026-06"}},
    {"question": "Any charges that look like parking?", "form": {"type": "search", "search_text": "parking"}},
    {"question": "What's the weather tomorrow?", "form": {"type": "unsupported"}},
    {"question": "Move money to my savings account", "form": {"type": "unsupported"}},
]


def log_refused(question: str) -> None:
    """Append the question text only. Never fails the answer."""
    try:
        path = rs.refused_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(question.replace("\n", " ").strip() + "\n")
    except OSError:
        log.warning("could not append to the refused-question log")


def _refused(refusal: Refusal, understood: str | None = None) -> dict:
    return {
        "type": "refused", "answer": refusal.reason, "answer_exact": refusal.reason, "rewrite": "skipped",
        "understood_as": understood, "interpretation": None,
        "result": {"reason": refusal.reason, "kind": refusal.kind, **refusal.detail},
        "rows": [], "rows_total": 0, "notes": {}, "query": None,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _interpretation(form: Form, result: calculations.Result) -> dict:
    def rng(r):
        return None if r is None else {"from": r.start and r.start.isoformat(), "to": r.end and r.end.isoformat(),
                                       "label": answers.fmt_range(r), "default": not r.said}
    return {
        "type": form.type, "direction": form.direction, "group_by": form.group_by, "order": form.order,
        "limit": form.limit, "category": form.category, "merchant_text": form.merchant_text,
        "merchants": list(form.merchants), "trip": form.trip, "search_text": form.search_text,
        "period": rng(form.period), "period_2": rng(form.period_2),
        "coverage": {k: c.status for k, c in result.coverage.items()},
    }


def _row(r: calculations.RowView) -> dict:
    return {"id": r.id, "date": r.date.isoformat(), "merchant": r.merchant, "amount": str(r.amount),
            "category": calculations.category_of(r), "section": r.section}


def route(question: str, session: Session, today: date | None = None, ask=ask_model,
          call=answers.call_model, embed=None) -> dict:
    """Answer one question. Raises ModelUnavailable (-> 503) if the form cannot be filled."""
    today = today or date.today()
    try:
        raw = ask(question, TEACH_EXAMPLES)
    except NotUnderstood:
        log_refused(question)
        return _refused(Refusal("not_understood", "I couldn't understand the question",
                                {"supported": list(TYPES)}))

    periods = read_queries.statement_periods(session)
    ctx = DataContext(
        today=today,
        categories=[c.name for c in category_store.current_categories(session) if not c.archived],
        merchants=read_queries.merchant_names(session),
        trips=read_queries.trips(session),
        full_months=calculations.full_months(periods),
    )
    try:
        form = resolve(raw, ctx)
    except NotUnderstood:
        log_refused(question)
        return _refused(Refusal("not_understood", "I couldn't understand the question",
                                {"supported": list(TYPES)}))
    if isinstance(form, Refusal):
        if form.kind == "unsupported":
            log_refused(question)
        return _refused(form)

    span = calculations.data_span(periods)
    if form.type in ("monthly_trend", "anomalies", "savings_opportunities", "category_change", "compare"):
        rows = read_queries.fetch_rows(session)                        # history needs everything
    else:
        rows = read_queries.fetch_rows(session, form.period.start, form.period.end)

    matches = None
    if form.type == "search":
        if embed is None:
            from app.embedder import generate_embedding as embed
        matches = read_queries.search(session, embed(form.search_text), form.direction,
                                      form.period.start, form.period.end, rs.SEARCH_CUTOFF, rs.SEARCH_CAP,
                                      category=form.category, merchants=form.merchants)

    result = calculations.compute(form, rows, periods, matches)
    if isinstance(result, Refusal):
        return _refused(result, answers.understood_as(form, {}, span))

    understood = answers.understood_as(form, result.coverage, span)
    period_cov = result.coverage.get("period")
    found_nothing = not result.rows and not result.table and not result.facts.get("count") \
        and not result.facts.get("rows_total")
    # Coverage "none" is not the same as no rows: a row dated just before its
    # statement starts still counts (ADR 0002, 4). Only say "No data" when
    # nothing was found; otherwise answer and let the interpretation line state coverage.
    if period_cov is not None and period_cov.status == "none" and found_nothing:
        text = f"No data: {answers.fmt_coverage(period_cov, span)}."
        exact, final, status = text, text, "skipped"
        table = None
    else:
        rendered = answers.render(form, result, span)
        exact, table = rendered.exact, rendered.table
        final, status = answers.rewrite(form, rendered, call)

    shown = result.rows[:rs.LIST_CAP]
    return {
        "type": form.type,
        "answer": final,
        "answer_exact": exact,
        "rewrite": status,
        "understood_as": understood,
        "interpretation": _interpretation(form, result),
        "result": _jsonable({**result.facts, "table": table}),
        "rows": [_row(r) for r in shown],
        "rows_total": len(result.rows) if form.type != "list" else result.facts.get("rows_total", len(shown)),
        "notes": result.notes,
        "query": {"template": form.type, "params": _jsonable(_interpretation(form, result))},
    }
