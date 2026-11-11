"""The text that gets embedded for search, and whether a stored embedding is
stale. No heavy imports, so the write path, reembed and the pure read-path
calculations can all use it (ADR 0005, decision 5b).
"""


def build_embedded_text(row, category_name: str | None) -> str:
    # merchant + raw_description + category gives the embedding model rich
    # signal. The printed description keeps terms that merchant cleaning drops.
    # A pending row has no category yet: empty name, refreshed once labelled.
    return f"{row.merchant} {row.raw_description} {category_name or ''}".rstrip()


def is_stale(row, category_name: str | None) -> bool:
    """Derived, not flagged: a lost background task self-heals on refresh."""
    return row.embedding is None or row.embedded_text != build_embedded_text(row, category_name)
