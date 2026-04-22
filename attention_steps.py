"""
Step-by-step attention computation for Layer 0, Head 2, first 5 tokens.
Uses HuggingFace hooks to extract every intermediate value.

Run with: python attention_steps.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT = "EBITDA grew 12% in Q3 because SG&A fell 8%"
TARGET_LAYER = 0
TARGET_HEAD  = 2
N_TOKENS     = 5
HEAD_DIM     = 64  # 2048 hidden / 32 heads = 64


def print_matrix(name, matrix, decimals=4):
    print(f"\n--- {name} ---")
    if matrix.ndim == 1:
        vals = matrix[:8]
        print("  " + "  ".join(f"{v:.{decimals}f}" for v in vals) + "  ...")
    elif matrix.ndim == 2:
        for i, row in enumerate(matrix):
            vals = row[:8] if row.shape[0] > 8 else row
            formatted = "  ".join(f"{v:8.{decimals}f}" for v in vals)
            suffix = "  ..." if row.shape[0] > 8 else ""
            print(f"  [{i}] {formatted}{suffix}")


def print_square(name, matrix, decimals=4):
    print(f"\n--- {name} ---")
    rows, cols = matrix.shape
    for i in range(rows):
        row_str = "  ".join(f"{matrix[i,j]:8.{decimals}f}" for j in range(cols))
        print(f"  [{i}] {row_str}")


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    output_attentions=True,
    dtype=torch.float32,
)
model.eval()

inputs    = tokenizer(INPUT_TEXT, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens    = [tokenizer.decode([t]) for t in input_ids[0]]

print(f"\nAll tokens: {tokens}")
print(f"Using first {N_TOKENS}: {tokens[:N_TOKENS]}")


# ── Hook storage ──────────────────────────────────────────────────────────────

captured = {}


def hook_pre_norm(module, input, output):
    # Input to RMSNorm = raw hidden state x
    captured["x"] = input[0].detach()          # [1, seq, 2048]
    captured["x_normed"] = output.detach()      # [1, seq, 2048]


# Register hook on layer 0
layer0 = model.model.layers[TARGET_LAYER]
layer0.input_layernorm.register_forward_hook(hook_pre_norm)


# ── Forward pass ─────────────────────────────────────────────────────────────

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)


# ── Extract weights for head 2 ────────────────────────────────────────────────

attn_module = layer0.self_attn
num_heads   = model.config.num_attention_heads
head_dim    = model.config.hidden_size // num_heads

# W_Q, W_K, W_V are Linear layers
W_Q = attn_module.q_proj.weight.detach()  # [num_heads * head_dim, hidden]
W_K = attn_module.k_proj.weight.detach()
W_V = attn_module.v_proj.weight.detach()

# Slice weights for head 2 only
h = TARGET_HEAD
Q_weight = W_Q[h * head_dim : (h+1) * head_dim, :]  # [64, 2048]
K_weight = W_K[h * head_dim : (h+1) * head_dim, :]
V_weight = W_V[h * head_dim : (h+1) * head_dim, :]

# Hidden states
x        = captured["x"][0, :N_TOKENS, :]         # [5, 2048]
x_normed = captured["x_normed"][0, :N_TOKENS, :]   # [5, 2048]

# Project to Q, K, V
Q = (Q_weight @ x_normed.T).T   # [5, 64]
K = (K_weight @ x_normed.T).T
V = (V_weight @ x_normed.T).T

# Dot products
scale      = np.sqrt(head_dim)
dots       = (Q @ K.T).numpy()           # [5, 5]
scaled     = dots / scale                 # [5, 5]

# Causal mask
mask = np.full((N_TOKENS, N_TOKENS), float('-inf'))
for i in range(N_TOKENS):
    for j in range(i + 1):
        mask[i, j] = 0.0
masked = scaled + mask                    # [5, 5]

# Softmax
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

attn_weights = softmax(masked)            # [5, 5]

# Weighted sum of V
attn_output = attn_weights @ V.numpy()   # [5, 64]


# ── Print every step ──────────────────────────────────────────────────────────

sep = "\n" + "=" * 70

print(sep)
print(f"LAYER {TARGET_LAYER}, HEAD {TARGET_HEAD} — STEP-BY-STEP ATTENTION")
print(f"Tokens: {tokens[:N_TOKENS]}")
print("=" * 70)

print(sep)
print("STEP 1: Raw input hidden state x  [5, 2048] — first 8 values per token")
print_matrix("x", x.numpy())

print(sep)
print("STEP 2: After RMSNorm  [5, 2048] — first 8 values per token")
print_matrix("x_normed", x_normed.numpy())

print(sep)
print("STEP 3: Q vectors after W_Q projection  [5, 64] — first 8 values")
print_matrix("Q", Q.numpy())

print(sep)
print("STEP 4: K vectors after W_K projection  [5, 64] — first 8 values")
print_matrix("K", K.numpy())

print(sep)
print("STEP 5: V vectors after W_V projection  [5, 64] — first 8 values")
print_matrix("V", V.numpy())

print(sep)
print("STEP 6: Raw dot products  Q @ K.T  [5, 5]")
print_square("Q @ K.T", dots)

print(sep)
print(f"STEP 7: After dividing by sqrt({head_dim}) = {scale:.4f}  [5, 5]")
print_square("scaled", scaled)

print(sep)
print("STEP 8: After causal mask  (-inf for future positions)  [5, 5]")
print_square("masked", masked)

print(sep)
print("STEP 9: After softmax  [5, 5]  (each row sums to 1.0)")
print_square("attn_weights", attn_weights)
print("\n  Row sums:", [round(float(attn_weights[i].sum()), 6) for i in range(N_TOKENS)])

print(sep)
print("STEP 10: Attention output  attn_weights @ V  [5, 64] — first 8 values")
print_matrix("attn_output", attn_output)

print(sep)
bos_col = attn_weights[:, 0]
print(f"BOS (position 0) attention received per query token: {[round(float(v), 4) for v in bos_col]}")
print(f"Average BOS mass for head {TARGET_HEAD}: {bos_col.mean():.4f}")
