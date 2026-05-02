import json
import re

import httpx
from sqlalchemy import text

from app.models import SessionLocal, Transaction, Trip


SCHEMA = """
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

Table: trips
Columns:
- id (INTEGER)
- name (TEXT)
- start_date (DATE)
- end_date (DATE)
"""


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


def _generate_sql(question: str) -> str:
    prompt = f"""You are a SQL expert. Generate a single PostgreSQL SELECT query to answer this question.

Schema:
{SCHEMA}

Rules:
- Only use SELECT, no INSERT/UPDATE/DELETE
- Only query the transactions and trips tables
- For charges use: is_credit = 'false'
- For payments/refunds use: is_credit = 'true'
- Use CURRENT_DATE for relative dates like "last month", "this year"
- Return only the SQL query, no explanation

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

    sql = response.json()["response"].strip()
    if "```" in sql:
        sql = re.sub(r"```(?:sql)?", "", sql).strip()
    return sql


def _validate_sql(sql: str) -> bool:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return False
    for keyword in ["insert", "update", "delete", "drop", "truncate", "alter"]:
        if keyword in normalized:
            return False
    return True


SIMILARITY_THRESHOLD = 0.4
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
- Use exact numbers from the results, do not calculate anything yourself
- Be concise and conversational
- Format amounts as dollars e.g. $45.23"""

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


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

        sql = _generate_sql(question)

        if not _validate_sql(sql):
            return {"type": "error", "answer": "Could not generate a safe query for that question.", "sql": sql}

        result = db.execute(text(sql))
        columns = result.keys()
        row_dicts = [dict(zip(columns, row)) for row in result.fetchall()]

        answer = _format_answer(question, sql, row_dicts)

        return {"type": "sql", "answer": answer, "sql": sql, "rows": row_dicts}

    finally:
        db.close()
