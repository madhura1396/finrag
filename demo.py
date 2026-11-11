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
    {
        "question": "How much did I spend on groceries in December 2025?",
        "note":     "total_spent — category filter",
    },
    # SQL — merchant specific
    {
        "question": "How much did I spend on shopping in December 2025?",
        "note":     "total_spent — category filter",
    },
    # SQL — count
    {
        "question": "How many transactions did I make in December 2025?",
        "note":     "count — date range",
    },
    # SQL — ranking
    {
        "question": "Which merchant did I spend the most at?",
        "note":     "ranking — merchants, most first",
    },
    # Semantic
    {
        "question": "Find charges that look like parking or city fees",
        "note":     "search — similarity, total labelled as fuzzy",
    },
    # Semantic
    {
        "question": "Any charges that look like online shopping?",
        "note":     "search — similarity, total labelled as fuzzy",
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
    except httpx.HTTPError as e:
        print(f"  ERROR: {e}")
        return
    if resp.status_code == 503:
        print("  The local model is unavailable. Is Ollama running with llama3.2?")
        return
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code}: {resp.text}")
        return
    data = resp.json()

    # ADR 0004: how the question was read comes first, so a misread is visible.
    if data.get("understood_as"):
        print(f"  {data['understood_as']}")
    if data["type"] == "refused":
        print(f"  Refused: {data['answer']}")
        details = {k: v for k, v in data["result"].items() if k not in ("reason", "kind")}
        if details:
            print(f"  Details: {json.dumps(details, default=str)}")
        return

    print(f"  Answer : {data['answer']}")
    if data["rewrite"] != "accepted":
        print(f"  (exact answer shown; rewrite {data['rewrite']})")
    table = (data.get("result") or {}).get("table")
    if table:
        for line in table:
            print(f"    {json.dumps(line, default=str)}")
    elif data.get("rows"):
        shown = len(data["rows"])
        print(f"  Rows   : {shown} of {data.get('rows_total', shown)}")
        for r in data["rows"][:10]:
            print(f"    {r['date']}  {r['merchant']:30s}  {r['amount']:>10s}  {r['category']}")
    notes = {k: v for k, v in (data.get("notes") or {}).items() if v}
    if notes:
        print(f"  Notes  : {notes}")


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
