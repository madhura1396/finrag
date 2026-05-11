"""
FinRAG demo script.

Shows the full pipeline working end to end against a real Wells Fargo statement.
Run with: .venv/bin/python demo.py

Requires the server to be running:
  .venv/bin/uvicorn app.main:app --reload
"""
import json
import httpx

BASE = "http://localhost:8000"

QUERIES = [
    # SQL — aggregation
    {
        "question": "How much did I spend on groceries in December 2025?",
        "note":     "SQL path — category filter + SUM",
    },
    # SQL — merchant specific
    {
        "question": "How much did I spend on shopping in December 2025?",
        "note":     "SQL path — category filter + SUM",
    },
    # SQL — count
    {
        "question": "How many transactions did I make in December 2025?",
        "note":     "SQL path — COUNT with date range",
    },
    # SQL — ranking
    {
        "question": "Which merchant did I spend the most at?",
        "note":     "SQL path — GROUP BY merchant, ORDER BY total DESC",
    },
    # Semantic
    {
        "question": "Find charges that look like parking or city fees",
        "note":     "Semantic path — vector similarity search",
    },
    # Semantic
    {
        "question": "Any charges that look like online shopping?",
        "note":     "Semantic path — vector similarity search",
    },
]


def separator(title: str, width: int = 64):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def run_query(question: str, note: str):
    separator(f"Q: {question}")
    print(f"  ({note})\n")

    try:
        resp = httpx.post(
            f"{BASE}/query",
            json={"question": question},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Answer : {data['answer']}")
    print(f"  Route  : {data['type'].upper()}")

    if data.get("sql"):
        # Print SQL on one line for readability
        sql_oneline = " ".join(data["sql"].split())
        print(f"  SQL    : {sql_oneline}")

    if data.get("rows"):
        print(f"  Rows   : {json.dumps(data['rows'], default=str)}")


def main():
    print("=" * 64)
    print("  FinRAG — Personal Finance RAG Demo")
    print("  Statement: Dec 2025 – Jan 2026 (Wells Fargo)")
    print("=" * 64)

    # Check server is up
    try:
        httpx.get(f"{BASE}/statements", timeout=5)
    except httpx.ConnectError:
        print("\n  ERROR: Server not running.")
        print("  Start it with: .venv/bin/uvicorn app.main:app --reload\n")
        return

    for q in QUERIES:
        run_query(q["question"], q["note"])

    print(f"\n{'=' * 64}")
    print("  Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
