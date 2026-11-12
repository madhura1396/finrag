"""
Manual test script for llm.py: calls the live local model.
Run with: python -m tests.test_llm   (needs Ollama running with llama3.2)

Invented descriptors only. Excluded from pytest (tests/conftest.py).
"""
from app.category_store import description_key
from app.db_setup import DEFAULT_CATEGORIES
from app.llm import label_descriptions


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


FAKE_DESCRIPTIONS = [
    "GROCER MART #101 SPRINGFIELD",
    "SKYWAY AIRLINES 8005550100",
    "FOODDASH*TACO TOWN 855-555-0101",
    "CABLECO INTERNET 855-555-0102",
    "MEGASTORE.COM*2X3K9 BILLING",
]


def test_label_descriptions():
    separator("TEST 1: label_descriptions()")
    print("Calling Ollama — this may take 10-30 seconds...")

    by_key = {description_key(d): d for d in FAKE_DESCRIPTIONS}
    results = label_descriptions(by_key, DEFAULT_CATEGORIES)

    failed = [key for key, label in results.items() if label is None]
    for key, label in results.items():
        print(f"  {key!r:40} -> {label}")

    if failed:
        print(f"\n⚠ No valid label for: {failed}")
    assert len(failed) < len(results), "the model labelled nothing"
    print("\n✓ PASSED")


if __name__ == "__main__":
    try:
        test_label_descriptions()
        separator("ALL TESTS PASSED")
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
