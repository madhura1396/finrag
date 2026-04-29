import numpy as np
import torch

# torchvision 0.2.x shim — must come before transformers import
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

TEXTS = {
    "General English": (
        "The quick brown fox jumps over the lazy dog. "
        "The weather today is sunny and warm."
    ),
    "Financial": (
        "EBITDA grew 12% in Q3 because SG&A fell 8%. "
        "Operating margin expanded to 24% from 21% in the prior year."
    ),
}

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.eval()
print()

# ── Perplexity helper ─────────────────────────────────────────────────────────

def analyze(text, label):
    inputs   = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]           # [seq_len]
    tokens   = [tokenizer.decode([t]) for t in input_ids]
    n        = len(tokens)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]                   # [seq_len, vocab]

    # softmax → probabilities
    logits_np = logits.numpy()                   # [seq_len, vocab]
    shifted   = logits_np - logits_np.max(axis=1, keepdims=True)
    exp_l     = np.exp(shifted)
    probs     = exp_l / exp_l.sum(axis=1, keepdims=True)  # [seq_len, vocab]

    # For position i, the model at position i-1 predicts token i.
    # So we evaluate positions 1 … n-1 (skipping the BOS prediction which has no "correct" prior).
    losses = []
    rows   = []

    for i in range(1, n):
        correct_id  = input_ids[i].item()
        p_correct   = probs[i - 1, correct_id]          # prob assigned by position i-1
        loss        = -np.log(p_correct + 1e-12)
        losses.append(loss)
        rows.append((tokens[i], p_correct, loss))

    avg_loss   = np.mean(losses)
    perplexity = np.exp(avg_loss)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"  {label}")
    print(f"  \"{text}\"")
    print("=" * 64)
    print(f"  {'Token':<20} {'p_correct':>12} {'Loss':>10}")
    print(f"  {'-'*44}")
    for tok, p, loss in rows:
        marker = "  <<" if loss > 5.0 else ""
        print(f"  {repr(tok):<20} {p:>12.6f} {loss:>10.4f}{marker}")
    print(f"  {'-'*44}")
    print(f"  Average loss : {avg_loss:.4f}")
    print(f"  Perplexity   : {perplexity:.2f}")
    print()

    return avg_loss, perplexity


# ── Run both texts ────────────────────────────────────────────────────────────

results = {}
for label, text in TEXTS.items():
    avg_loss, ppl = analyze(text, label)
    results[label] = (avg_loss, ppl)

# ── Comparison ────────────────────────────────────────────────────────────────

print("=" * 64)
print("  COMPARISON")
print("=" * 64)
for label, (avg_loss, ppl) in results.items():
    print(f"  {label:<20}  avg loss = {avg_loss:.4f}   perplexity = {ppl:.2f}")

labels = list(results.keys())
ppl_ratio = results[labels[1]][1] / results[labels[0]][1]
print()
print(f"  Perplexity ratio (Financial / General): {ppl_ratio:.2f}x")
if ppl_ratio > 1:
    print("  → The model finds financial text harder to predict.")
else:
    print("  → The model finds general English harder to predict.")
print()
