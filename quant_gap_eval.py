"""
Quantization gap evaluation.

Runs the same FinRAG classification + generation questions through:
  - bfloat16 HuggingFace model  (ground truth)
  - Q4_K_M Ollama model         (production)

Prints a token-level diff and accuracy comparison.
Run with: python quant_gap_eval.py
"""

import time
import httpx
import torch
import numpy as np
from typing import Tuple, Optional

# torchvision shim for Python 3.8 + transformers 4.46
import torchvision.transforms as _tvt
if not hasattr(_tvt, "InterpolationMode"):
    from enum import Enum
    class _IM(Enum):
        NEAREST = 0; BILINEAR = 2; BICUBIC = 3; BOX = 4; HAMMING = 5; LANCZOS = 1
    _tvt.InterpolationMode = _IM
    import torchvision.transforms.functional as _F
    _F.InterpolationMode = _IM

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# ── Eval questions ─────────────────────────────────────────────────────────────
# Two task types matching FinRAG's actual workload:
#   1. Classifier prompts  → expected single label: "sql" or "semantic"
#   2. Generation prompts  → open-ended; we compare top predicted token

CLASSIFIER_CASES = [
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: How much did I spend on groceries last month?\nAnswer:',
        "expected": "sql",
    },
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: What are my biggest spending categories?\nAnswer:',
        "expected": "sql",
    },
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: Show me transactions at Trader Joe\'s\nAnswer:',
        "expected": "sql",
    },
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: Find charges that look like food delivery\nAnswer:',
        "expected": "semantic",
    },
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: What subscription services am I paying for?\nAnswer:',
        "expected": "semantic",
    },
    {
        "prompt": 'Classify as "sql" or "semantic".\nQuestion: Any suspicious or unusual charges?\nAnswer:',
        "expected": "semantic",
    },
]

FINANCIAL_CONTINUATIONS = [
    {
        "prompt": "EBITDA grew 12% in Q3 because SG&A fell",
        "description": "financial acronym continuation",
    },
    {
        "prompt": "Operating margin expanded to 24% from 21% in the prior",
        "description": "percentage continuation",
    },
    {
        "prompt": "Revenue declined due to lower Average Revenue Per",
        "description": "ARPU continuation",
    },
    {
        "prompt": "The company reported a net loss of $4.2M driven by increased capital",
        "description": "capex continuation",
    },
]


# ── HuggingFace inference ──────────────────────────────────────────────────────

def load_hf_model():
    print("Loading bfloat16 HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")
    model.eval()
    print(f"  dtype: {model.model.layers[0].self_attn.q_proj.weight.dtype}")
    return tokenizer, model


def hf_top_token(tokenizer, model, prompt: str) -> Tuple[str, float]:
    """Returns (top_token_string, probability) for the next token after prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].float().numpy()
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    top_id = int(np.argmax(probs))
    return tokenizer.decode([top_id]), float(probs[top_id])


# ── Ollama inference ───────────────────────────────────────────────────────────

def ollama_top_token(prompt: str) -> Tuple[str, bool]:
    """Returns (generated_text, success). num_predict=1 → single token."""
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip(), True
    except Exception as e:
        return f"ERROR: {e}", False


def ollama_available() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


# ── Evaluation runners ─────────────────────────────────────────────────────────

def run_classifier_eval(tokenizer, model, use_ollama: bool):
    print("\n" + "=" * 70)
    print("  TASK 1: CLASSIFIER PROMPTS  (expected: 'sql' or 'semantic')")
    print("=" * 70)
    print(f"  {'Prompt (truncated)':<40} {'Expected':<10} {'HF':<12} {'Ollama':<12} {'Match?'}")
    print(f"  {'-'*78}")

    hf_correct = 0
    ol_correct = 0

    for case in CLASSIFIER_CASES:
        prompt   = case["prompt"]
        expected = case["expected"]
        short    = prompt.split("\n")[1].replace("Question: ", "")[:38]

        hf_tok, hf_prob = hf_top_token(tokenizer, model, prompt)
        hf_label = "sql" if "sql" in hf_tok.lower() else ("semantic" if "sem" in hf_tok.lower() else hf_tok.strip())
        hf_ok = hf_label == expected

        if use_ollama:
            ol_tok, ol_ok_req = ollama_top_token(prompt)
            ol_label = "sql" if "sql" in ol_tok.lower() else ("semantic" if "sem" in ol_tok.lower() else ol_tok.strip())
            ol_ok = ol_label == expected
            match = "✓✓" if (hf_ok and ol_ok) else ("✗✗" if (not hf_ok and not ol_ok) else "△ diff")
        else:
            ol_label = "N/A"
            ol_ok = None
            match = ""

        hf_correct += int(hf_ok)
        if ol_ok is not None:
            ol_correct += int(ol_ok)

        hf_str = f"{hf_label}({'✓' if hf_ok else '✗'})"
        ol_str = f"{ol_label}({'✓' if ol_ok else '✗'})" if use_ollama else "N/A"
        print(f"  {short:<40} {expected:<10} {hf_str:<12} {ol_str:<12} {match}")

    n = len(CLASSIFIER_CASES)
    print(f"\n  HF accuracy  : {hf_correct}/{n} = {hf_correct/n*100:.0f}%")
    if use_ollama:
        print(f"  Ollama accuracy: {ol_correct}/{n} = {ol_correct/n*100:.0f}%")
        print(f"  Gap          : {(hf_correct - ol_correct)/n*100:+.0f} pp  (HF minus Ollama)")
    return hf_correct / n, (ol_correct / n if use_ollama else None)


def run_continuation_eval(tokenizer, model, use_ollama: bool):
    print("\n" + "=" * 70)
    print("  TASK 2: FINANCIAL CONTINUATION  (top predicted next token)")
    print("=" * 70)

    rows = []
    for case in FINANCIAL_CONTINUATIONS:
        prompt = case["prompt"]
        desc   = case["description"]

        hf_tok, hf_prob = hf_top_token(tokenizer, model, prompt)

        if use_ollama:
            ol_tok, _ = ollama_top_token(prompt)
            agree = "✓ agree" if hf_tok.strip().lower() == ol_tok.strip().lower() else "✗ differ"
        else:
            ol_tok = "N/A"
            agree  = ""

        rows.append((desc, hf_tok, hf_prob, ol_tok, agree))

    print(f"  {'Description':<35} {'HF token':<15} {'HF prob':>8}  {'Ollama token':<15} {'Agree?'}")
    print(f"  {'-'*82}")
    agree_count = 0
    for desc, hf_tok, hf_prob, ol_tok, agree in rows:
        print(f"  {desc:<35} {repr(hf_tok):<15} {hf_prob:>8.4f}  {repr(ol_tok):<15} {agree}")
        if "agree" in agree:
            agree_count += 1

    if use_ollama:
        print(f"\n  Token agreement: {agree_count}/{len(rows)} = {agree_count/len(rows)*100:.0f}%")
    return agree_count / len(rows) if use_ollama else None


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ol_up = ollama_available()
    if not ol_up:
        print("⚠  Ollama not reachable at localhost:11434 — running HF-only mode.")
        print("   Start Ollama and re-run to see the quantization gap.\n")

    tokenizer, model = load_hf_model()
    print()

    t0 = time.time()
    hf_cls_acc, ol_cls_acc = run_classifier_eval(tokenizer, model, use_ollama=ol_up)
    cont_agree = run_continuation_eval(tokenizer, model, use_ollama=ol_up)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Classifier — HF bfloat16 : {hf_cls_acc*100:.0f}%")
    if ol_up:
        print(f"  Classifier — Ollama Q4_K_M: {ol_cls_acc*100:.0f}%")
        gap = (hf_cls_acc - ol_cls_acc) * 100
        print(f"  Accuracy gap             : {gap:+.0f} pp")
        print(f"  Continuation agreement   : {cont_agree*100:.0f}%")
        if abs(gap) <= 5:
            verdict = "Q4_K_M quantization has negligible impact on this task."
        elif gap > 5:
            verdict = f"Q4_K_M degrades accuracy by {gap:.0f} pp — worth noting in eval story."
        else:
            verdict = "Q4_K_M surprisingly outperforms bfloat16 on this task."
        print(f"\n  Verdict: {verdict}")
    print(f"\n  Total eval time: {elapsed:.1f}s")
    print()
