import json
import re

import httpx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.models import SessionLocal, Transaction, Trip


TRANSACTIONS_SCHEMA = """
Table: transactions
Columns:
- id (INTEGER)
- statement_id (INTEGER)
- trans_date (DATE)
- merchant (TEXT)
- raw_description (TEXT)
- amount (NUMERIC) — always positive, use is_credit to determine direction
- is_credit (TEXT) — 'true' means payment/refund, 'false' means charge
- category (TEXT)
- needs_review (BOOLEAN)

Valid categories: Groceries, Dining & DoorDash, Travel & Transport, Shopping, Telephone & Internet, Entertainment, Health & Fitness, Other
"""

TRIPS_SCHEMA = """
Table: trips
Columns:
- id (INTEGER)
- name (TEXT)
- start_date (DATE)
- end_date (DATE)

Join trips to transactions on the date range: trans_date BETWEEN trips.start_date AND trips.end_date.
Amounts always come from transactions — trips has no amount column.
"""

TRIP_WORDS = ("trip", "vacation", "holiday", "getaway")


def _schema_for(question: str, db) -> str:
    """Show the trips table only when the question is actually about a trip.

    A 3B model pulls any table it can see into a JOIN: with trips in the schema,
    llama3.2 answers "which merchant did I spend the most at" with
    `SUM(trips.amount)`, a column that does not exist. Removing the table from the
    prompt removes the temptation — more reliable than instructing it not to.
    """
    try:
        names = [row[0] for row in db.execute(text("SELECT name FROM trips")).all()]
    except SQLAlchemyError:
        db.rollback()
        return TRANSACTIONS_SCHEMA

    # No trips defined means no trip question is answerable — showing the table can only
    # invite a join that returns nothing or references a column it does not have.
    if not names:
        return TRANSACTIONS_SCHEMA

    lowered = question.lower()
    mentions_trip = any(word in lowered for word in TRIP_WORDS) or any(
        name and name.lower() in lowered for name in names
    )
    return TRANSACTIONS_SCHEMA + TRIPS_SCHEMA if mentions_trip else TRANSACTIONS_SCHEMA


def _classify(question: str) -> str:
    prompt = f"""Classify this financial question as either "sql" or "semantic".

"sql" = needs calculation, aggregation, totals, comparisons, date ranges, rankings
"semantic" = needs pattern finding, anomaly detection, similarity, exploratory browsing

Question: {question}

Reply with a single word: sql or semantic"""

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 1}},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["response"].strip().lower()
        if "semantic" in result:
            return "semantic"
        return "sql"
    except httpx.HTTPError:
        return "sql"


def _data_date_range(db) -> tuple:
    """Actual span of the loaded data, used to anchor relative dates in the prompt."""
    row = db.execute(text("SELECT MIN(trans_date), MAX(trans_date) FROM transactions")).first()
    return (row[0], row[1]) if row else (None, None)


def _generate_sql(question: str, date_range: tuple, schema: str, previous_sql: str = None, error: str = None) -> str:
    data_start, data_end = date_range

    # Statements are historical, so CURRENT_DATE is the wrong anchor — "last month"
    # relative to today returns nothing when the data is months old.
    if data_start and data_end:
        date_context = f"The transaction data covers {data_start} to {data_end}. Today's date is irrelevant."
        anchor = data_end
    else:
        date_context = "The transactions table is empty."
        anchor = "the most recent trans_date"

    # On a retry, show the model its own failed query and the database error.
    retry_block = ""
    if previous_sql and error:
        retry_block = f"""
Your previous query failed. Fix it — do not repeat the same mistake.
Previous query: {previous_sql}
Error: {error}
"""

    prompt = f"""You are a SQL expert. Generate a single PostgreSQL SELECT query to answer this question.

Schema:
{schema}
{date_context}

Rules:
- Only use SELECT, no INSERT/UPDATE/DELETE
- Only use the tables and columns listed in the schema above
- Always write is_credit = 'false'. That means charges, which is what "spending",
  "transactions", "purchases" and every ordinary question refer to. Write
  is_credit = 'true' only when the question literally says payment, refund or credit.
- Every literal value in the query must come from the question itself. Never invent a
  name and never emit a placeholder such as '%my_merchant%', '%store%' or '%category%'.
- Add a merchant filter ONLY if the question names a specific store. If the question
  names no store, do not reference the merchant column in WHERE at all.
- Add a category filter ONLY if the question names a spending type, and use one of the
  valid categories listed above e.g. category ILIKE '%groceries%'
- Add a date filter ONLY if the question names a time period. Anchor relative periods
  like "last month" to {anchor}, never to CURRENT_DATE.
- If the question asks about everything, write no WHERE clause beyond is_credit.
- Do not put comments in the query.
{retry_block}
Think through each step, then write the query:
Metric: <what are we measuring — count, sum, list?>
Table: <which table>
Charges or payments: <does the question literally say payment, refund or credit? If no,
                      write "charges, is_credit = 'false'">
Filters: <only the filters the question actually asks for — say "none" if it asks for none>
Aggregation: <SUM / COUNT / none>
Query: <write the SELECT statement here>

Question: {question}"""

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model":       "llama3.2",
            "prompt":      prompt,
            "stream":      False,
            "options": {
                "temperature": 0.1,
                "top_k":       10,
                "top_p":       0.5,
            },
        },
        timeout=60,
    )
    response.raise_for_status()

    raw = response.json()["response"].strip()
    if "```" in raw:
        raw = re.sub(r"```(?:sql)?", "", raw).strip()

    # Extract only the SELECT block — strip CoT before it and explanation after it.
    # The model sometimes appends prose after the query; stop at the first blank line
    # after SELECT, which reliably separates SQL from trailing explanation text.
    match = re.search(r"(SELECT\b.*?)(?:\n\s*\n|$)", raw, re.IGNORECASE | re.DOTALL)
    sql = match.group(1).strip() if match else raw

    # Strip -- comments only after the block is extracted: removing them earlier would
    # turn a comment-only line into the blank line the regex above stops at, truncating
    # the query. They are valid SQL but silently swallow the rest of the statement
    # whenever it is flattened onto one line for a log or the demo output.
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"\n\s*\n", "\n", sql).strip()
    return sql


# Word boundaries matter: a bare substring check rejects legitimate queries, e.g.
# merchant ILIKE '%dropbox%' contains "drop".
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|grant|revoke|create)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped.lower().startswith("select"):
        return False
    # Anything after a semicolon is a second, stacked statement.
    if ";" in stripped.rstrip(";"):
        return False
    return FORBIDDEN_SQL.search(stripped) is None


SIMILARITY_THRESHOLD = 0.75
MAX_CHUNKS = 5


def _semantic_search(question: str, db) -> list:
    from app.embedder import generate_embedding

    query_vector = generate_embedding(question)

    # Fetch scored so we can rerank before returning
    rows = (
        db.query(Transaction, Transaction.embedding.cosine_distance(query_vector).label("score"))
        .filter(Transaction.embedding.cosine_distance(query_vector) < SIMILARITY_THRESHOLD)
        .order_by(Transaction.embedding.cosine_distance(query_vector))
        .limit(MAX_CHUNKS)
        .all()
    )

    # Fix 1: put most relevant chunk last — closest to the question in the prompt.
    # Fix 2: already capped at MAX_CHUNKS (5 → fewer than old limit of 10).
    rows_worst_first = sorted(rows, key=lambda r: r.score)[::-1]
    return [r.Transaction for r in rows_worst_first]


def _format_answer(question: str, sql: str, rows: list) -> str:
    # Rows are already ordered worst-first by _semantic_search so the most
    # relevant chunk lands closest to the question in the prompt (Fix 1).
    prompt = f"""You are a personal finance assistant. Answer the question using only the data provided.

Results:
{json.dumps(rows, indent=2, default=str)}

Question: {question}

Rules:
- The data is always correct — never say the data is missing, unclear, or unavailable
- If total_spent is in the results, report that number directly
- Use exact numbers from the results, do not calculate anything yourself
- Be concise — one or two sentences maximum
- Format amounts as dollars e.g. $45.23"""

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except (httpx.HTTPError, KeyError):
        # The rows are the real answer; without the model we still return the numbers
        # rather than failing the whole request.
        if not rows:
            return "No matching transactions found."
        return f"Found {len(rows)} matching transaction(s): {json.dumps(rows, default=str)}"


MAX_SQL_ATTEMPTS = 2


def _run_sql(question: str, db) -> dict:
    """Generate and execute SQL, retrying once with the database error fed back in.

    Generation is non-deterministic, so a query that Postgres rejects is often fixed by
    a second attempt. Any failure returns a typed error dict — never an exception, which
    would surface to the caller as an opaque HTTP 500.
    """
    date_range = _data_date_range(db)
    schema     = _schema_for(question, db)
    sql, error = None, None

    for _ in range(MAX_SQL_ATTEMPTS):
        try:
            sql = _generate_sql(question, date_range, schema, previous_sql=sql, error=error)
        except (httpx.HTTPError, KeyError) as e:
            return {
                "type":   "error",
                "answer": "Could not reach the local model to build a query. Is Ollama running?",
                "sql":    None,
                "rows":   [],
                "error":  str(e),
            }

        if not _validate_sql(sql):
            error = "Query must be a single SELECT statement over transactions or trips."
            continue

        try:
            result    = db.execute(text(sql))
            columns   = result.keys()
            row_dicts = [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            # A failed statement aborts the Postgres transaction — every later query on
            # this session fails until the rollback clears it.
            db.rollback()
            error = str(getattr(e, "orig", e)).strip()
            continue

        answer = _format_answer(question, sql, row_dicts)
        return {"type": "sql", "answer": answer, "sql": sql, "rows": row_dicts}

    return {
        "type":   "error",
        "answer": "Could not build a working query for that question. Try rephrasing it.",
        "sql":    sql,
        "rows":   [],
        "error":  error,
    }


def route(question: str) -> dict:
    db = SessionLocal()

    try:
        query_type = _classify(question)

        if query_type == "semantic":
            rows = _semantic_search(question, db)
            row_dicts = [
                {
                    "date":     str(tx.trans_date),
                    "merchant": tx.merchant,
                    "amount":   str(tx.amount),
                    "category": tx.category,
                }
                for tx in rows
            ]
            answer = _format_answer(question, "semantic search", row_dicts)
            return {"type": "semantic", "answer": answer, "rows": row_dicts}

        return _run_sql(question, db)

    finally:
        db.close()
