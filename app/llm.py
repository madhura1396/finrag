import json
import re
import httpx


CATEGORIES = [
    "Groceries",
    "Dining & DoorDash",
    "Travel & Transport",
    "Shopping",
    "Telephone & Internet",
    "Entertainment",
    "Health & Fitness",
    "Other",
]

BATCH_SIZE = 15


def _call_llm_single_batch(payload: list) -> tuple:
    prompt = f"""You are a financial data assistant. Clean merchant names and assign categories.

For each transaction return ONLY a JSON array with this exact structure:
[
  {{"id": <same id as input>, "merchant": "<cleaned name>", "category": "<category>"}}
]

Categories you must choose from (pick the closest match):
{json.dumps(CATEGORIES, indent=2)}

Rules:
- merchant: remove noise like store numbers, phone numbers, city/state, transaction codes
- merchant: keep it short and human-readable e.g. "Trader Joe's", "Southwest Airlines"
- category: must be exactly one of the categories listed above
- Return only valid JSON array, no explanation

Transactions:
{json.dumps(payload, indent=2)}
"""

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()

    llm_text = response.json()["response"].strip()
    if "```" in llm_text:
        llm_text = re.sub(r"```(?:json)?", "", llm_text).strip()

    if not llm_text:
        return [], {p["id"] for p in payload}

    results    = json.loads(llm_text)
    batch_ids  = {p["id"] for p in payload}
    returned   = {r["id"] for r in results}
    missed_ids = batch_ids - returned
    return results, missed_ids


def _process_with_splitting(payload: list, all_results: list, all_missed: set):
    if not payload:
        return

    try:
        results, missed_ids = _call_llm_single_batch(payload)
        all_results.extend(results)

        if missed_ids and len(payload) > 1:
            missed_payload = [p for p in payload if p["id"] in missed_ids]
            mid = len(missed_payload) // 2
            _process_with_splitting(missed_payload[:mid], all_results, all_missed)
            _process_with_splitting(missed_payload[mid:], all_results, all_missed)
        else:
            all_missed.update(missed_ids)

    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        if len(payload) > 1:
            mid = len(payload) // 2
            _process_with_splitting(payload[:mid], all_results, all_missed)
            _process_with_splitting(payload[mid:], all_results, all_missed)
        else:
            all_missed.add(payload[0]["id"])


def call_llm_batch(transactions: list) -> tuple:
    all_results = []
    all_missed  = set()

    for i in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[i : i + BATCH_SIZE]
        payload = [{"id": t["id"], "raw": t["raw_description"]} for t in batch]
        _process_with_splitting(payload, all_results, all_missed)

    return all_results, all_missed


