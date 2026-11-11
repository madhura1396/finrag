"""The model's only job on the write path: label descriptions no pin, rule or
memory covers (ADR 0005, decision 3).

Temperature 0 and a JSON schema constrain the output; code validates it
anyway. A failure or invalid answer returns None for that description, which
leaves the row `pending`, never `Other`.
"""

import json
from dataclasses import dataclass

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
BATCH_SIZE = 15
TIMEOUT_SECONDS = 120
NONE_FIT = "none_fit"
MAX_MERCHANT_LENGTH = 80
MAX_SUGGESTION_WORDS = 3


@dataclass(frozen=True)
class ModelLabel:
    category: str            # one of the offered names, or NONE_FIT
    merchant: str
    suggestion: str | None   # only meaningful with NONE_FIT


def _schema(category_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":         {"type": "integer"},
                        "category":   {"type": "string", "enum": [*category_names, NONE_FIT]},
                        "merchant":   {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["id", "category", "merchant"],
                },
            }
        },
        "required": ["labels"],
    }


def _prompt(batch: list[tuple[int, str]], category_names: list[str]) -> str:
    items = "\n".join(f"{i}: {description}" for i, description in batch)
    return f"""Label each card transaction description.

For each id return:
- "category": exactly one of {json.dumps(category_names)}, or "{NONE_FIT}" if none of them fits
- "merchant": a short human-readable merchant name, without store numbers, phone numbers, city, state or codes
- "suggestion": only when category is "{NONE_FIT}", a category name of at most {MAX_SUGGESTION_WORDS} words

Descriptions:
{items}
"""


def _validate(item: dict, category_names: set[str]) -> ModelLabel | None:
    category = item.get("category")
    merchant = item.get("merchant")
    if category != NONE_FIT and category not in category_names:
        return None
    if not isinstance(merchant, str) or not merchant.strip() or len(merchant) > MAX_MERCHANT_LENGTH:
        return None
    suggestion = item.get("suggestion") if category == NONE_FIT else None
    if suggestion is not None:
        suggestion = " ".join(str(suggestion).split()[:MAX_SUGGESTION_WORDS]) or None
    return ModelLabel(category=category, merchant=merchant.strip(), suggestion=suggestion)


def _label_batch(batch: list[tuple[int, str]], category_names: list[str]) -> dict[int, ModelLabel | None]:
    results: dict[int, ModelLabel | None] = {i: None for i, _ in batch}
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model":   MODEL,
                "prompt":  _prompt(batch, category_names),
                "stream":  False,
                "format":  _schema(category_names),
                "options": {"temperature": 0},
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["response"])
        labels = parsed["labels"]
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
        return results

    allowed = set(category_names)
    for item in labels if isinstance(labels, list) else []:
        if isinstance(item, dict) and item.get("id") in results and results[item["id"]] is None:
            results[item["id"]] = _validate(item, allowed)
    return results


def label_descriptions(descriptions: dict[str, str], category_names: list[str]) -> dict[str, ModelLabel | None]:
    """Label one representative description per memory key.

    `descriptions` maps description key -> a raw description. Returns the same
    keys; None means the model failed or answered invalidly for that key.
    """
    keys = list(descriptions)
    out: dict[str, ModelLabel | None] = {}
    for start in range(0, len(keys), BATCH_SIZE):
        chunk = keys[start:start + BATCH_SIZE]
        batch = [(i, descriptions[key]) for i, key in enumerate(chunk)]
        labelled = _label_batch(batch, category_names)
        for i, key in enumerate(chunk):
            out[key] = labelled[i]
    return out
