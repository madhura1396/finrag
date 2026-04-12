import json
import re
from typing import Optional

import httpx


CATEGORIES = [
    "Groceries",
    "Dining & DoorDash",
    "Travel & Transport",
    "Shopping",
    "Telephone & Internet",
    "Entertainment",
    "Insurance",
    "Utilities"
    "Health & Fitness",
    "Other",
]


def call_llm_batch(transactions: list[dict]) -> list[dict]:
    payload = []
    for t in transactions:
        payload.append({"id": t["id"], "raw": t["raw_description"]})

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

    results = json.loads(llm_text)

    returned_ids = {r["id"] for r in results}
    all_ids      = {t["id"] for t in transactions}
    missed_ids   = all_ids - returned_ids

    return results, missed_ids


def call_llm_single(transaction: dict) -> Optional[dict]:
    prompt = f"""Clean this bank transaction and assign a category.

Return ONLY a JSON object:
{{"merchant": "<cleaned name>", "category": "<category>"}}

Categories:
{json.dumps(CATEGORIES, indent=2)}

Transaction: {transaction["raw_description"]}
"""

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()

        llm_text = response.json()["response"].strip()
        if "```" in llm_text:
            llm_text = re.sub(r"```(?:json)?", "", llm_text).strip()

        return json.loads(llm_text)

    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return None
