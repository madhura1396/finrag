"""Read-path form eval (ADR 0004, detail 7). Manual; needs Ollama with llama3.2.

    python -m evals.read_path.run [--rewrite]

Sends each synthetic question to the served model through the production
prompt and few-shot set (query_router.TEACH_EXAMPLES), and compares the form
field by field. Prints accuracy overall, per type and per field. With
--rewrite, also measures how often placeholder rewrites pass the check.
Synthetic questions only; nothing from your statements is used or printed.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from app import answers
from app import read_config as rs
from app.query_router import TEACH_EXAMPLES
from app.question_form import RawForm, ask_model, ModelUnavailable, NotUnderstood

QUESTIONS = Path(__file__).with_name("questions.yaml")
FIELDS = [f for f in RawForm.__dataclass_fields__]


def normalise(form: dict) -> dict:
    """Defaults the resolver applies count as equal to an empty field."""
    out = {f: form.get(f) for f in FIELDS}
    for k, v in out.items():
        if isinstance(v, str):
            out[k] = v.strip().lower() or None
    if out["direction"] is None:
        out["direction"] = "received" if out["type"] == "total_received" else "spent"
    if out["type"] == "total_received" and out["direction"] == "spent":
        out["direction"] = "received"
    if out["type"] == "ranking":
        out["group_by"] = out["group_by"] or "merchant"
        out["order"] = out["order"] or "most"
    else:
        out["group_by"] = out["order"] = None
    if out["limit"] == rs.DEFAULT_LIMIT:
        out["limit"] = None
    return out


def field_matches(field: str, expected, got) -> bool:
    if field in ("category", "merchant", "search_text") and expected and got:
        return expected in got or got in expected        # copying words, not matching exactly
    return expected == got


def run_forms(cases) -> int:
    teach = {e["question"].strip().lower() for e in TEACH_EXAMPLES}
    overlap = [c["question"] for c in cases if c["question"].strip().lower() in teach]
    if overlap:
        print(f"refusing: test questions overlap the few-shot set: {overlap}", file=sys.stderr)
        return 2

    per_type, per_field = defaultdict(Counter), defaultdict(Counter)
    correct = failed = 0
    merchant_as_category = 0           # merchant values the model put in category
    for case in cases:
        expected = normalise(case["form"])
        try:
            got = normalise(vars(ask_model(case["question"], TEACH_EXAMPLES)))
        except NotUnderstood:
            got = None
        except ModelUnavailable as e:
            print(f"model unavailable: {e}", file=sys.stderr)
            return 1
        # The resolver ignores every other field of an unsupported form, so only its type counts.
        fields = ["type"] if expected["type"] == "unsupported" else FIELDS
        ok = got is not None and all(field_matches(f, expected[f], got[f]) for f in fields)
        correct += ok
        if expected["merchant"] and got and got["category"] and not expected["category"]:
            merchant_as_category += 1
        failed += got is None
        per_type[expected["type"]]["right" if ok else "wrong"] += 1
        for f in fields:
            per_field[f]["right" if got and field_matches(f, expected[f], got[f]) else "wrong"] += 1
        if not ok:
            print(f"  MISS  {case['question']!r}\n        expected {_short(expected)}\n        got      {_short(got)}")

    total = len(cases)
    print(f"\nForm accuracy: {correct}/{total} ({100 * correct / total:.1f}%), unusable output: {failed}")
    print(f"Merchant in category: {merchant_as_category}")
    print("\nPer type:")
    for t, c in sorted(per_type.items()):
        print(f"  {t:24s} {c['right']}/{c['right'] + c['wrong']}")
    print("\nPer field:")
    for f in FIELDS:
        c = per_field[f]
        print(f"  {f:12s} {c['right']}/{c['right'] + c['wrong']}")
    return 0


def _short(form) -> str:
    return "None" if form is None else str({k: v for k, v in form.items() if v is not None})


def run_rewrites(trials: int = 10) -> None:
    accepted = Counter()
    for question_type in answers.SENTENCE_TYPES:
        for _ in range(trials):
            count = 3 if question_type == "compare" else 2
            try:
                text = answers.call_model(answers.rewrite_prompt(question_type, count))
            except Exception:
                accepted["error"] += 1
                continue
            accepted["accepted" if answers.check_rewrite(text, count) else "rejected"] += 1
    total = sum(accepted.values())
    print(f"\nRewrite check: {accepted['accepted']}/{total} accepted, {accepted['rejected']} rejected, "
          f"{accepted['error']} errors (temperature 0, so repeats are near-identical)")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cases = yaml.safe_load(QUESTIONS.read_text())
    status = run_forms(cases)
    if status == 0 and "--rewrite" in argv:
        run_rewrites()
    return status


if __name__ == "__main__":
    sys.exit(main())
