"""Retrieval eval for the `search` question type, at the production cutoff and
cap (app/read_config.py), against your own database.

    python -m app.eval_search [cases.yaml]

The cases name real merchants from your statements, so they live outside the
repo: FINRAG_SEARCH_EVAL, default ~/.finrag/search_eval.yaml. The format is
evals/search_eval.example.yaml (invented names). Prints hits and misses per
query, to tune SEARCH_CUTOFF.
"""

import os
import sys
from pathlib import Path

import yaml

from app import read_config as rs
from app.calculations import category_of
from app.embedder import generate_embedding
from app.models import new_session
from app.read_queries import search


def cases_path(argv: list[str]) -> Path:
    if argv:
        return Path(argv[0])
    return Path(os.path.expanduser(os.getenv("FINRAG_SEARCH_EVAL", "~/.finrag/search_eval.yaml")))


def separator(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main(argv=None) -> int:
    path = cases_path(sys.argv[1:] if argv is None else argv)
    if not path.exists():
        print(f"no cases at {path}; copy evals/search_eval.example.yaml there and edit it", file=sys.stderr)
        return 2
    cases = yaml.safe_load(path.read_text()) or []

    hits = misses = 0
    db = new_session()
    try:
        for case in cases:
            separator(f'Query: "{case["query"]}"')
            vector = generate_embedding(case["query"])
            results = search(db, vector, case.get("direction", "spent"), None, None, rs.SEARCH_CUTOFF, rs.SEARCH_CAP)
            for row in results:
                print(f"  {row.merchant:35s}  {category_of(row):25s}")
            returned = [r.merchant.lower() for r in results]
            for expected in case["expected"]:
                if any(expected.lower() in m or m in expected.lower() for m in returned):
                    hits += 1
                else:
                    misses += 1
                    print(f"  MISSED EXPECTED: {expected}")
    finally:
        db.close()

    separator("SUMMARY")
    total = hits + misses
    print(f"cutoff {rs.SEARCH_CUTOFF}, cap {rs.SEARCH_CAP}: {hits}/{total} expected merchants found"
          + (f" ({round(hits / total * 100)}%)" if total else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
