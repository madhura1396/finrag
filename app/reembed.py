"""
Re-embed all transactions using the current build_embedded_text format.
Run from repo root: python -m app.reembed
"""
from app.models import SessionLocal, Transaction
from app.embedder import build_embedded_text, generate_embedding


def reembed_all():
    db = SessionLocal()
    try:
        txs = db.query(Transaction).all()
        total = len(txs)
        print(f"Found {total} transactions to re-embed...")

        for i, tx in enumerate(txs, 1):
            tx.embedded_text = build_embedded_text(tx)
            tx.embedding     = generate_embedding(tx.embedded_text)

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total}  last: {tx.embedded_text!r}")

        db.commit()
        print(f"\nDone. {total} transactions re-embedded.")

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    reembed_all()
