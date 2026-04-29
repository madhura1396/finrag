import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# torchvision 0.2.x is missing InterpolationMode; patch before importing transformers
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
INPUT_TEXT = "EBITDA grew 12% in Q3 because SG&A fell"
TEMPERATURES = [0.1, 0.3, 0.7, 1.0, 1.5, 2.0]
TOP_K = 10

# ── Load and run ──────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.eval()

inputs = tokenizer(INPUT_TEXT, return_tensors="pt")
tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
print(f"Tokens: {tokens}")
print(f"Predicting next token after: '{tokens[-1]}'")
print()

with torch.no_grad():
    outputs = model(**inputs)

# logits: [1, seq_len, vocab_size] — take last token position
logits_last = outputs.logits[0, -1, :].numpy()  # [vocab_size]

# ── Per-temperature analysis ──────────────────────────────────────────────────

results = {}

for temp in TEMPERATURES:
    scaled = logits_last / temp
    # softmax via log-sum-exp for numerical stability
    shifted = scaled - scaled.max()
    exp_s   = np.exp(shifted)
    probs   = exp_s / exp_s.sum()

    top_indices = np.argsort(probs)[::-1][:TOP_K]
    top_tokens  = [tokenizer.decode([i]) for i in top_indices]
    top_logits  = logits_last[top_indices]
    top_probs   = probs[top_indices]

    results[temp] = {
        "tokens": top_tokens,
        "logits": top_logits,
        "probs":  top_probs,
    }

    print(f"Temperature {temp}:")
    print(f"  {'Rank':<5} {'Token':<15} {'Logit':>10} {'Probability':>13}")
    print(f"  {'-'*47}")
    for rank, (tok, logit, prob) in enumerate(zip(top_tokens, top_logits, top_probs), 1):
        print(f"  {rank:<5} {repr(tok):<15} {logit:>10.4f} {prob:>13.6f}")
    print()

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f'Temperature Effect on Next-Token Distribution\nPrompt: "{INPUT_TEXT}"',
    fontsize=13, fontweight="bold", y=1.01
)
axes_flat = axes.flatten()

for ax, temp in zip(axes_flat, TEMPERATURES):
    r      = results[temp]
    labels = [repr(t).strip("'") for t in r["tokens"]]
    probs  = r["probs"]

    bars = ax.bar(range(TOP_K), probs, color="steelblue", edgecolor="white", linewidth=0.5)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{p:.3f}",
            ha="center", va="bottom", fontsize=6.5
        )

    ax.set_xticks(range(TOP_K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_title(f"Temperature = {temp}", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("outputs/temperature_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/temperature_visualization.png")
print()

# ── Summary ───────────────────────────────────────────────────────────────────

def entropy(probs):
    p = probs[probs > 0]
    return -np.sum(p * np.log(p))

for temp in [0.1, 1.0, 2.0]:
    r = results[temp]
    print(f"At T={temp}: top token = {repr(r['tokens'][0])} with prob {r['probs'][0]:.4f}")

print()
for temp in [0.1, 1.0, 2.0]:
    e = entropy(results[temp]["probs"])
    print(f"Entropy at T={temp}: {e:.4f}")
