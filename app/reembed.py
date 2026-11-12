"""
Re-embed every stale transaction (ADR 0005, decision 5b).
Run from repo root: python -m app.reembed
"""
from app.categorization import refresh_embeddings
from app.embedder import generate_embedding
from app.models import new_session


def reembed_all() -> int:
    db = new_session()
    try:
        count = refresh_embeddings(db, generate_embedding)
        print(f"Re-embedded {count} stale transactions.")
        return count
    finally:
        db.close()


if __name__ == "__main__":
    reembed_all()
