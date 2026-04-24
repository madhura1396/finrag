"""
Residual stream accumulation across all 16 layers for token "grew" (position 4).
Input: "EBITDA grew 12% in Q3 because SG&A fell 8%"

17 snapshots: raw embedding + after each of the 16 layers.

Run with: python residual_stream.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT = "EBITDA grew 12% in Q3 because SG&A fell 8%"
TARGET_POS = 4   # "grew"
SEP        = "\n" + "=" * 72


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
model.eval()

inputs    = tokenizer(INPUT_TEXT, return_tensors="pt")
tokens    = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
print(f"Tokens: {tokens}")
print(f"Target: [{TARGET_POS}] = {tokens[TARGET_POS]!r}")


# ── Forward pass with output_hidden_states=True ───────────────────────────────
# Returns hidden_states: tuple of (num_layers + 1) tensors, each [1, seq, d_model]
# hidden_states[0]  = raw token embeddings (before any transformer layer)
# hidden_states[k]  = output of layer k-1  (after attention + FFN + residuals)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states   # tuple of 17 tensors for 16-layer model
n_snapshots   = len(hidden_states)      # 17
assert n_snapshots == 17, f"Expected 17, got {n_snapshots}"


# ── Extract snapshots for TARGET_POS ─────────────────────────────────────────

snapshots = [hidden_states[i][0, TARGET_POS, :] for i in range(n_snapshots)]


# ── Print each snapshot ───────────────────────────────────────────────────────

def fmt6(tensor):
    return "  ".join(f"{v:9.5f}" for v in tensor[:6].tolist())

print(SEP)
print(f"RESIDUAL STREAM — token {tokens[TARGET_POS]!r} (position {TARGET_POS})")
print(f"17 snapshots: raw embedding → after each of 16 layers")
print(SEP)

for i, snap in enumerate(snapshots):
    label = "raw embedding" if i == 0 else f"after layer {i-1:2d}"
    norm  = snap.norm().item()

    print(f"\n{'─'*72}")
    print(f"Snapshot {i:2d} — {label}")
    print(f"  norm = {norm:.4f}")
    print(f"  first 6 values: {fmt6(snap)}")

    if i > 0:
        delta      = snap - snapshots[i - 1]
        delta_norm = delta.norm().item()
        pct        = 100.0 * delta_norm / norm if norm > 0 else 0.0
        print(f"  delta from prev: {fmt6(delta)}")
        print(f"  delta norm = {delta_norm:.4f}  ({pct:.1f}% of hidden state norm)")


# ── Summary table ─────────────────────────────────────────────────────────────

print(SEP)
print(f"SUMMARY TABLE — token {tokens[TARGET_POS]!r}")
print(SEP)
print(f"\n  {'Snapshot':>8}  {'Label':>22}  {'Hidden norm':>12}  {'Delta norm':>11}  {'Delta %':>8}")
print("  " + "-" * 70)

for i, snap in enumerate(snapshots):
    label     = "raw embedding" if i == 0 else f"after layer {i-1:2d}"
    norm      = snap.norm().item()
    if i == 0:
        print(f"  {i:>8}  {label:>22}  {norm:>12.4f}  {'—':>11}  {'—':>8}")
    else:
        delta      = snap - snapshots[i - 1]
        delta_norm = delta.norm().item()
        pct        = 100.0 * delta_norm / norm if norm > 0 else 0.0
        print(f"  {i:>8}  {label:>22}  {norm:>12.4f}  {delta_norm:>11.4f}  {pct:>7.1f}%")

print()

# ── Highlight biggest contributors ───────────────────────────────────────────

delta_norms = []
for i in range(1, n_snapshots):
    delta_norm = (snapshots[i] - snapshots[i - 1]).norm().item()
    delta_norms.append((i - 1, delta_norm))   # (layer_idx, delta_norm)

delta_norms_sorted = sorted(delta_norms, key=lambda x: x[1], reverse=True)

print(SEP)
print("LAYERS RANKED BY CONTRIBUTION (delta norm, largest first)")
print(SEP)
print(f"\n  {'Rank':>4}  {'Layer':>6}  {'Delta norm':>11}")
print("  " + "-" * 28)
for rank, (layer_idx, dn) in enumerate(delta_norms_sorted, 1):
    bar = "█" * int(dn / delta_norms_sorted[0][1] * 30)
    print(f"  {rank:>4}  layer {layer_idx:>2}  {dn:>11.4f}  {bar}")

print()
