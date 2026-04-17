import numpy as np
from models import SessionLocal, Transaction


def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


db = SessionLocal()
transactions = db.query(Transaction).filter(
    Transaction.embedding != None,
    Transaction.is_credit == "false",
).limit(20).all()
db.close()

if len(transactions) < 2:
    print("Not enough transactions with embeddings in DB. Run pipeline first.")
    exit()

# Pick first transaction for raw vector inspection
tx0 = transactions[0]
vector0 = tx0.embedding

print("=" * 60)
print(f"RAW VECTOR — {tx0.merchant}")
print("=" * 60)
print(f"Embedded text    : {tx0.embedded_text}")
print(f"Total dimensions : {len(vector0)}")
print(f"First 10 values  : {[round(x, 4) for x in vector0[:10]]}")
print(f"Min value        : {round(min(vector0), 4)}")
print(f"Max value        : {round(max(vector0), 4)}")
print(f"Vector magnitude : {round(np.linalg.norm(vector0), 4)} (should be ~1.0 since normalized)")


# Find two transactions from the same category (similar)
same_category = [t for t in transactions if t.category == tx0.category and t.id != tx0.id]
# Find one from a different category (dissimilar)
diff_category  = [t for t in transactions if t.category != tx0.category]

print("\n" + "=" * 60)
print("ALL TRANSACTIONS LOADED")
print("=" * 60)
for t in transactions:
    print(f"  [{t.category:30s}] {t.merchant:30s}  ${t.amount}")

if same_category:
    tx_similar = same_category[0]
    sim = cosine_similarity(vector0, tx_similar.embedding)
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY — same category (should be LOW distance)")
    print("=" * 60)
    print(f"A: {tx0.embedded_text}")
    print(f"B: {tx_similar.embedded_text}")
    print(f"Similarity : {round(sim, 4)}")
    print(f"Distance   : {round(1 - sim, 4)}")
    print(f"Angle      : {round(np.degrees(np.arccos(min(sim, 1.0))), 2)}°")

if diff_category:
    tx_dissimilar = diff_category[0]
    sim2 = cosine_similarity(vector0, tx_dissimilar.embedding)
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY — different category (should be HIGH distance)")
    print("=" * 60)
    print(f"A: {tx0.embedded_text}")
    print(f"B: {tx_dissimilar.embedded_text}")
    print(f"Similarity : {round(sim2, 4)}")
    print(f"Distance   : {round(1 - sim2, 4)}")
    print(f"Angle      : {round(np.degrees(np.arccos(min(sim2, 1.0))), 2)}°")

if same_category and diff_category:
    sim_same = cosine_similarity(vector0, same_category[0].embedding)
    sim_diff = cosine_similarity(vector0, diff_category[0].embedding)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Same category distance : {round(1 - sim_same, 4)}")
    print(f"Diff category distance : {round(1 - sim_diff, 4)}")
    print(f"Separation gap         : {round((1 - sim_diff) - (1 - sim_same), 4)}")
    print()
    print("pgvector returns results ordered by cosine_distance ascending.")
    print("Lower distance = more similar. Distance 0 = identical.")
    print("If same category distance > 0.3, model is not separating categories well.")
