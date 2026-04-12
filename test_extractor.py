"""
Manual test script for extractor.py
Run with: python test_extractor.py path/to/your/statement.pdf

Tests each function in order and prints results.
Stop at first failure — each function depends on the previous one succeeding.
"""
import sys
from extractor import (
    extract_raw_text,
    parse_statement_period,
    detect_structure,
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_extract_raw_text(filepath: str):
    separator("TEST 1: extract_raw_text()")
    pages = extract_raw_text(filepath)

    print(f"Pages extracted: {len(pages)}")
    for p in pages:
        word_count = len(p['text'].split())
        print(f"  Page {p['page_number']}: {word_count} words, "
              f"first 80 chars: {p['text'][:80].replace(chr(10), ' ')!r}")

    assert len(pages) > 0, "No pages extracted — is this a valid PDF?"
    print("\n✓ PASSED")
    return pages



def test_parse_statement_period(pages: list):
    separator("TEST 3: parse_statement_period()")

    period_start, period_end = parse_statement_period(pages)
    print(f"Period start : {period_start}")
    print(f"Period end   : {period_end}")

    assert period_start < period_end, "period_start must be before period_end"
    print("✓ PASSED")
    return period_start, period_end


def test_detect_structure(pages: list):
    separator("TEST 4: detect_structure()")
    print("Calling Ollama — this may take 10-30 seconds...")

    structure = detect_structure(pages)

    print(f"start_page       : {structure['start_page']}")
    print(f"end_page         : {structure['end_page']}")
    print(f"start_marker     : {structure['start_marker']!r}")
    print(f"end_marker       : {structure['end_marker']!r}")
    print(f"credit_markers   : {structure['credit_markers']}")
    print(f"purchase_markers : {structure['purchase_markers']}")

    # Check if we got real values or the fallback defaults
    if structure["start_page"] is None:
        print("\n⚠ WARNING: LLM fell back to defaults (start_page is None)")
        print("  This means Ollama failed or returned invalid JSON.")
        print("  Extraction will still work but will scan all pages.")
    else:
        # Verify markers actually appear in the PDF text
        full_text = "\n".join(p["text"] for p in pages)
        for marker in [structure["start_marker"], structure["end_marker"]]:
            if marker in full_text:
                print(f"  ✓ '{marker}' found in PDF")
            else:
                print(f"  ✗ '{marker}' NOT found in PDF — may cause issues")

    print("\n✓ PASSED")
    return structure


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extractor.py path/to/statement.pdf")
        print("Example: python test_extractor.py uploads/010226_WellsFargo.pdf")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Testing with: {filepath}")

    try:
        pages          = test_extract_raw_text(filepath)
        period_start, period_end = test_parse_statement_period(pages)
        structure      = test_detect_structure(pages)

        separator("ALL TESTS PASSED")
        print(f"Statement period : {period_start} → {period_end}")
        print(f"Transaction pages: {structure['start_page']} → {structure['end_page']}")
        print(f"Start marker     : {structure['start_marker']!r}")

    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
