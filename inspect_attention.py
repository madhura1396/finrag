"""
Multi-head attention inspection for Llama 3.2 1B.
Loads model via HuggingFace transformers, extracts real per-head attention weights.

Run with: python inspect_attention.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT = "EBITDA grew 12% in Q3 because SG&A fell 8%"
SINK_THRESHOLD = 0.5


# ── Load model and tokenizer ──────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    output_attentions=True,
    torch_dtype=torch.float32,
)
model.eval()
print("Loaded.\n")


# ── Tokenize ──────────────────────────────────────────────────────────────────

inputs    = tokenizer(INPUT_TEXT, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens    = [tokenizer.decode([t]) for t in input_ids[0]]
seq_len   = len(tokens)

print("=" * 60)
print("TOKENIZED INPUT")
print("=" * 60)
for i, tok in enumerate(tokens):
    print(f"  [{i:2d}] {tok!r}")


# ── Forward pass ─────────────────────────────────────────────────────────────

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple of tensors, one per layer
# Each tensor: [batch, num_heads, seq_len, seq_len]
layer0_attn = outputs.attentions[0][0]  # [num_heads, seq_len, seq_len]
num_heads   = layer0_attn.shape[0]


# ── Per-head analysis ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"LAYER 0 — {num_heads} HEADS, SEQ LEN {seq_len}")
print(f"{'='*60}")
print(f"\n{'Head':>6}  {'BOS mass':>10}  {'Type':>12}  Top attended token per query")
print("-" * 70)

sink_heads     = []
semantic_heads = []

for h in range(num_heads):
    attn = layer0_attn[h].numpy()  # [seq_len, seq_len]

    # BOS sink: average fraction of attention mass going to position 0
    bos_mass = attn[:, 0].mean()

    # Argmax per query row — which key token each query attends to most
    argmax_keys = attn.argmax(axis=1)
    top_attended = " ".join(f"{tokens[k]!r}" for k in argmax_keys)

    head_type = "SINK" if bos_mass > SINK_THRESHOLD else "SEMANTIC"

    print(f"  h{h:02d}    {bos_mass:>8.3f}    {head_type:>10}   {top_attended}")

    if bos_mass > SINK_THRESHOLD:
        sink_heads.append(h)
    else:
        semantic_heads.append((h, bos_mass))


# ── Sink ratio summary ────────────────────────────────────────────────────────

all_bos_mass = [layer0_attn[h].numpy()[:, 0].mean() for h in range(num_heads)]
avg_bos      = np.mean(all_bos_mass)
avg_non_bos  = np.mean([layer0_attn[h].numpy()[:, 1:].mean() for h in range(num_heads)])

print(f"\n{'='*60}")
print("SINK ANALYSIS")
print(f"{'='*60}")
print(f"Sink heads (BOS > {SINK_THRESHOLD}): {sink_heads}")
print(f"Semantic heads             : {[h for h, _ in semantic_heads]}")
print(f"Avg BOS attention mass     : {avg_bos:.4f}")
print(f"Avg non-BOS attention mass : {avg_non_bos:.4f}")
print(f"Sink ratio                 : {avg_bos / avg_non_bos:.2f}x")
print()
if avg_bos / avg_non_bos > 2:
    print("Position 0 is acting as an attention sink.")
else:
    print("Attention is distributed — no strong sink at position 0.")


# ── ASCII heatmap for top 2 semantic heads ────────────────────────────────────

top_semantic = sorted(semantic_heads, key=lambda x: x[1])[:2]

for h, bos_mass in top_semantic:
    attn = layer0_attn[h].numpy()

    print(f"\n{'='*60}")
    print(f"ASCII HEATMAP — Head {h} (BOS mass={bos_mass:.3f}, semantic)")
    print(f"{'='*60}")

    col_labels = [t[:4] for t in tokens]
    header = "      " + "".join(f"{c:>5}" for c in col_labels)
    print(header)

    for i, row_tok in enumerate(tokens):
        row = f"{row_tok[:4]:>5} "
        for j in range(seq_len):
            val = int(attn[i, j] * 9)
            val = min(val, 9)
            row += f"    {val}"
        print(row)
