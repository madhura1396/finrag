"""
Manual test script for llm.py
Run with: python test_llm.py
"""
from llm import call_llm_batch


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


fake_transactions = [
    {"id": 1, "raw_description": "TRADER JOE S #534 SAN FRANCISCO CA"},
    {"id": 2, "raw_description": "SOUTHWEST AIRLINES 8009359792 TX"},
    {"id": 3, "raw_description": "DOORDASH*CHIPOTLE 855-973-1040 CA"},
    {"id": 4, "raw_description": "SPECTRUM 855-707-7328 MO"},
    {"id": 5, "raw_description": "AMAZON.COM*2X3K9 AMZN.COM/BILL WA"},
]


def test_call_llm_batch():
    separator("TEST 1: call_llm_batch()")
    print("Calling Ollama — this may take 10-30 seconds...")

    results, missed_ids = call_llm_batch(fake_transactions)

    print(f"\nResults ({len(results)} returned):")
    for r in results:
        print(f"  id={r['id']}  merchant={r['merchant']!r}  category={r['category']!r}")

    if missed_ids:
        print(f"\n⚠ Missed IDs: {missed_ids}")
    else:
        print("\n✓ No missed transactions")

    assert len(results) > 0, "LLM returned nothing"
    print("\n✓ PASSED")
    return missed_ids



    print("\n✓ PASSED")


if __name__ == "__main__":
    try:
        test_call_llm_batch()
        separator("ALL TESTS PASSED")
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
