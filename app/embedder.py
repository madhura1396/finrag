from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_embedded_text(tx) -> str:
    # merchant + raw_description + category gives the embedding model rich signal.
    # raw_description preserves terms like "PARK METERS" that get stripped during
    # merchant cleaning but are essential for matching queries like "parking".
    return f"{tx.merchant} {tx.raw_description} {tx.category}"


def generate_embedding(text: str) -> list:
    model  = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
