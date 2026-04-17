"""
Evaluation script for semantic search.
Tests real queries against expected merchants from your actual statement.
Run with: python eval_search.py
"""
import numpy as np
from embedder import generate_embedding
from models import SessionLocal, Transaction


TEST_CASES = [
    {
        "query":    "grocery shopping",
        "expected": ["Trader Joe's", "Aldi", "Costco", "WM Supercenter"],
    },
    {
        "query":    "food delivery",
        "expected": ["DoorDash"],
    },
    {
        "query":    "flights and hotels",
        "expected": ["Agoda.com", "Snow.com/Vail Resorts"],
    },
    {
        "query":    "online shopping",
        "expected": ["Shein", "Old Navy", "eBay", "TJ Maxx"],
    },
    {
        "query":    "freelance work payment",
        "expected": ["Upwork"],
    },
    {
        "query":    "streaming and subscriptions",
        "expected": ["Claude.ai"],
    },
    {
        "query":    "parking",
        "expected": ["City of Rochester Park Meters", "CITY OF ROCHESTER"],
    },
]


def search(query: str, db, threshold: float = 1.0) -> list:
    query_vector = generate_embedding(query)
    results = db.query(Transaction).filter(
        Transaction.embedding.cosine_distance(query_vector) < threshold
    ).order_by(
        Transaction.embedding.cosine_distance(query_vector)
    ).limit(10).all()
    return results


def cosine_distance(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return round(1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 3)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


db = SessionLocal()

all_hits   = 0
all_misses = 0

for tc in TEST_CASES:
    separator(f"Query: \"{tc['query']}\"")
    results = search(tc["query"], db)

    print(f"{'Merchant':35s}  {'Category':25s}  {'Distance':>8s}  Match?")
    print("-" * 85)

    query_vector = generate_embedding(tc["query"])

    for tx in results:
        dist  = cosine_distance(query_vector, tx.embedding)
        match = any(e.lower() in tx.merchant.lower() or tx.merchant.lower() in e.lower()
                    for e in tc["expected"])
        flag  = "✓" if match else "✗"
        print(f"{tx.merchant:35s}  {tx.category:25s}  {dist:>8.3f}  {flag}")

    returned_merchants = [tx.merchant for tx in results]
    for expected in tc["expected"]:
        found = any(expected.lower() in m.lower() or m.lower() in expected.lower()
                    for m in returned_merchants)
        if not found:
            print(f"  MISSED EXPECTED: {expected}")
            all_misses += 1
        else:
            all_hits += 1

db.close()

separator("SUMMARY")
total = all_hits + all_misses
print(f"Hits  : {all_hits}/{total}")
print(f"Misses: {all_misses}/{total}")
print(f"Score : {round(all_hits/total*100)}%" if total > 0 else "No results")
print()
print("Use this output to decide the right distance threshold.")
print("Look at the highest distance of a correct hit — that is your threshold ceiling.")
