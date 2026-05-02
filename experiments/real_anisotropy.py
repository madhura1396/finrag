"""
Cosine similarity matrices across token pairs at 4 depth snapshots.
Shows how token representations become more/less similar as they pass
through Llama layers (anisotropy analysis).

Run with: python real_anisotropy.py
Output:   outputs/real_anisotropy.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT = "EBITDA grew 12% in Q3 because SG&A fell 8%"
SNAPSHOTS  = {0: "Snapshot 0\n(raw embeddings)",
              4: "Snapshot 4\n(after layer 3)",
              8: "Snapshot 8\n(after layer 7)",
             16: "Snapshot 16\n(after layer 15)"}


# ── Load and run ──────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
model.eval()

inputs  = tokenizer(INPUT_TEXT, return_tensors="pt")
tokens  = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
n_tok   = len(tokens)
print(f"Tokens ({n_tok}): {tokens}")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# hidden_states: tuple of 17 tensors [1, seq, 2048], index 0 = raw embeddings
hidden_states = outputs.hidden_states


# ── Cosine similarity matrix ──────────────────────────────────────────────────

def cosine_matrix(h):
    """h: [n_tok, d_model] numpy array → returns [n_tok, n_tok] cosine sim matrix"""
    norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
    h_n   = h / norms
    return h_n @ h_n.T


def most_least_similar(mat, tok_list):
    n = len(tok_list)
    best_val, best_pair = -1, (0, 1)
    worst_val, worst_pair = 2, (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            v = mat[i, j]
            if v > best_val:
                best_val, best_pair = v, (i, j)
            if v < worst_val:
                worst_val, worst_pair = v, (i, j)
    return best_val, best_pair, worst_val, worst_pair


# ── Collect data ──────────────────────────────────────────────────────────────

snap_indices = sorted(SNAPSHOTS.keys())
matrices     = {}
avg_sims     = {}

print()
for idx in snap_indices:
    h   = hidden_states[idx][0].numpy()   # [n_tok, 2048]
    mat = cosine_matrix(h)                # [n_tok, n_tok]
    matrices[idx] = mat

    # average over upper triangle (excluding diagonal)
    upper = mat[np.triu_indices(n_tok, k=1)]
    avg   = upper.mean()
    avg_sims[idx] = avg

    bv, bp, wv, wp = most_least_similar(mat, tokens)
    label = SNAPSHOTS[idx].replace("\n", " ")
    print(f"{label}")
    print(f"  avg pairwise cosine sim : {avg:.4f}")
    print(f"  most similar  : {tokens[bp[0]]!r} ↔ {tokens[bp[1]]!r}  ({bv:.4f})")
    print(f"  least similar : {tokens[wp[0]]!r} ↔ {tokens[wp[1]]!r}  ({wv:.4f})")
    print()


# ── Plot 4 heatmaps ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(26, 7))
fig.suptitle(
    f'Token Cosine Similarity Across Layers — "{INPUT_TEXT}"',
    fontsize=13, fontweight="bold", y=1.01
)

short_labels = [t.replace("<|begin_of_text|>", "<BOS>").strip() for t in tokens]

for ax, idx in zip(axes, snap_indices):
    mat = matrices[idx]
    im  = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_tok))
    ax.set_yticks(range(n_tok))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7.5)
    ax.set_yticklabels(short_labels, fontsize=7.5)

    title = SNAPSHOTS[idx] + f"\navg sim = {avg_sims[idx]:.3f}"
    ax.set_title(title, fontsize=10, pad=8)

    # annotate cells with value
    for i in range(n_tok):
        for j in range(n_tok):
            v = mat[i, j]
            color = "white" if v > 0.65 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("outputs/real_anisotropy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/real_anisotropy.png")


# ── Observations ──────────────────────────────────────────────────────────────

print("=" * 70)
print("OBSERVATIONS")
print("=" * 70)
print(f"""
1. RAW EMBEDDINGS (snapshot 0, avg sim = {avg_sims[0]:.3f})
   Each token starts with its learned embedding vector. Similarity values
   span a wide range — tokens with very different meanings sit far apart
   in embedding space. This is the "cold start" state with no context.

2. AFTER LAYER 3 (snapshot 4, avg sim = {avg_sims[4]:.3f})
   Early layers run attention across all token pairs, mixing context into
   every position. Similarity generally rises: representations begin to
   share information (e.g. all tokens have now "seen" the BOS token and
   each other). The matrix becomes more uniform.

3. AFTER LAYER 7 (snapshot 8, avg sim = {avg_sims[8]:.3f})
   Mid-layers continue to blend representations. Average similarity
   typically peaks in middle layers — this is the "anisotropy" effect
   documented in language model research: hidden states cluster together
   in a cone-shaped region of high-dimensional space, making many pairs
   appear highly similar even when semantically unrelated.

4. AFTER LAYER 15 (snapshot 16, avg sim = {avg_sims[16]:.3f})
   The final layer's representations diverge again. The model must
   distinguish tokens to make accurate next-token predictions — if all
   hidden states were identical the logits would be the same for every
   position, which is useless. The unembedding projection (lm_head)
   requires discriminative final representations, so later layers
   "sharpen" the differences back up.

5. ANISOTROPY TAKEAWAY
   Unlike random vectors in high-dimensional space (which have near-zero
   cosine similarity), trained LLM hidden states are anisotropic — they
   cluster. This means cosine similarity alone is a poor measure of
   semantic proximity mid-network. Whitening or isotropy-corrected
   metrics (e.g. normalizing by the mean direction) are needed for
   reliable similarity comparisons inside the network.
""")
