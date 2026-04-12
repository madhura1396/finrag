from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_embedded_text(tx) -> str:
    kind   = "received" if tx.is_credit == "true" else "spent"
    date   = tx.trans_date.strftime("%m/%d")
    return f"On {date}, {kind} ${tx.amount} at {tx.merchant} in {tx.category}"


def generate_embedding(text: str) -> list:
    model  = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
