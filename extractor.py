import re
import json
from datetime import date, datetime
from pathlib import Path
from decimal import Decimal, InvalidOperation

import fitz
import httpx


def extract_raw_text(filepath: str) -> list[dict]:
    pdf = fitz.open(filepath)
    pages = []

    for i, page in enumerate(pdf):
        text = page.get_text()
        if len(text.strip()) < 10:
            continue
        pages.append({
            "page_number": i + 1,
            "text": text.strip()
        })

    pdf.close()
    return pages


def parse_statement_period(pages: list[dict]) -> tuple[date, date]:
    period_pattern = re.compile(r"""
        Statement\ Period
        [:\s]+
        (\d{1,2}/\d{1,2}/\d{4})
        \s+
        (?:to|-)
        \s+
        (\d{1,2}/\d{1,2}/\d{4})
    """, re.IGNORECASE | re.VERBOSE)

    full_text = "\n".join(p["text"] for p in pages)
    match = period_pattern.search(full_text)

    if not match:
        raise ValueError(
            "Could not find statement period in PDF. "
            "Expected format: 'Statement Period MM/DD/YYYY to MM/DD/YYYY'"
        )

    period_start = datetime.strptime(match.group(1), "%m/%d/%Y").date()
    period_end   = datetime.strptime(match.group(2), "%m/%d/%Y").date()

    return period_start, period_end


TRANSACTION_PATTERN = re.compile(r"""
    ^\s*
    \d{4}
    \s+
    (\d{1,2}/\d{1,2})
    \s+
    \d{1,2}/\d{1,2}
    \s+
    [A-Z0-9]+
    \s+
    (.+?)
    \s+
    ([\d,]+\.\d{2})
    \s*$
""", re.VERBOSE)


def _parse_date(raw_date: str, period_start: date, period_end: date) -> date | None:
    try:
        month, day = [int(x) for x in raw_date.split("/")]
        candidate = date(period_start.year, month, day)
        if candidate < period_start or candidate > period_end:
            candidate = date(period_end.year, month, day)
        return candidate
    except (ValueError, TypeError):
        return None


def _parse_amount(raw_amount: str) -> Decimal:
    try:
        return Decimal(raw_amount.replace(",", ""))
    except InvalidOperation:
        return Decimal("0.00")


def parse_transactions_from_text(
    pages: list[dict],
    structure: dict,
    period_start: date,
    period_end: date,
) -> list[dict]:
    start_page = structure.get("start_page")
    end_page   = structure.get("end_page")

    if start_page and end_page:
        transaction_pages = [
            p for p in pages
            if start_page <= p["page_number"] <= end_page
        ]
    else:
        transaction_pages = pages

    all_lines = []
    for page in transaction_pages:
        all_lines.extend(page["text"].splitlines())

    start_marker     = structure["start_marker"].lower()
    end_marker       = structure["end_marker"].lower()
    credit_markers   = [m.lower() for m in structure["credit_markers"]]
    purchase_markers = [m.lower() for m in structure["purchase_markers"]]

    transactions    = []
    current_section = None
    current_tx      = None
    in_transactions = False

    for line in all_lines:
        stripped = line.strip()
        lower    = stripped.lower()

        if not in_transactions:
            if start_marker in lower:
                in_transactions = True
            continue

        if end_marker in lower:
            if current_tx:
                transactions.append(current_tx)
                current_tx = None
            break

        if any(m in lower for m in credit_markers):
            if current_tx:
                transactions.append(current_tx)
                current_tx = None
            current_section = "credits"
            continue

        if any(m in lower for m in purchase_markers):
            if current_tx:
                transactions.append(current_tx)
                current_tx = None
            current_section = "purchases"
            continue

        if "total" in lower and "for this period" in lower:
            if current_tx:
                transactions.append(current_tx)
                current_tx = None
            continue

        if not stripped:
            continue

        match = TRANSACTION_PATTERN.match(line)

        if match:
            if current_tx:
                transactions.append(current_tx)

            current_tx = {
                "trans_date":      _parse_date(match.group(1), period_start, period_end),
                "raw_description": match.group(2).strip(),
                "amount":          _parse_amount(match.group(3)),
                "is_credit":       current_section == "credits",
            }

        elif current_tx and stripped:
            if not stripped.startswith("Card") and len(stripped) > 3:
                current_tx["raw_description"] += " " + stripped

    if current_tx:
        transactions.append(current_tx)

    return transactions


def detect_structure(pages: list[dict]) -> dict:
    sample_pages = pages[:2]
    all_pages_text = ""
    for p in sample_pages:
        all_pages_text += f"\n\n--- PAGE {p['page_number']} ---\n{p['text']}"

    prompt = f"""You are analyzing a bank statement PDF.
Read the pages below and identify where the transaction section is.

Return ONLY a JSON object with these exact keys:
- "start_page": page number where transactions first appear (integer)
- "end_page": page number where transactions end (integer)
- "start_marker": the exact heading text that marks where transactions begin (e.g. "Transactions")
- "end_marker": the exact text that marks where transactions end (e.g. "Fees Charged")
- "credit_markers": list of section headings for credits or refunds (e.g. ["Other Credits", "Payments"])
- "purchase_markers": list of section headings for charges (e.g. ["Purchases, Balance Transfers & Other Charges"])

Rules:
- Copy heading text EXACTLY as it appears in the document — do not paraphrase
- start_page and end_page must be integers matching the PAGE numbers labeled below
- If unsure about end_marker, use "Fees Charged" as default
- Return only valid JSON, no explanation

{all_pages_text}
"""

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()

        llm_text = response.json()["response"].strip()
        if "```" in llm_text:
            llm_text = re.sub(r"```(?:json)?", "", llm_text).strip()

        structure = json.loads(llm_text)

        full_text = "\n".join(p["text"] for p in pages)
        if structure["start_marker"] not in full_text:
            raise ValueError(f"LLM identified '{structure['start_marker']}' as start marker but it does not appear in the PDF.")

        structure["start_page"] = int(structure["start_page"])
        structure["end_page"]   = int(structure["end_page"])

        page_numbers = [p["page_number"] for p in pages]
        if structure["start_page"] not in page_numbers:
            raise ValueError(f"LLM identified start_page={structure['start_page']} but PDF only has pages {page_numbers}")

        return structure

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[detect_structure] falling back to defaults. Reason: {e}")
        return {
            "start_page":        None,
            "end_page":          None,
            "start_marker":      "Transactions",
            "end_marker":        "Fees Charged",
            "credit_markers":    ["Other Credits", "Payments"],
            "purchase_markers":  ["Purchases, Balance Transfers"],
        }
