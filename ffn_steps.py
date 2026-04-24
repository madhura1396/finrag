"""
Step-by-step FFN computation for Layer 0, first 5 tokens.
Architecture: SwiGLU — gate_proj, up_proj, down_proj, silu activation.

Run with: python ffn_steps.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT  = "EBITDA grew 12% in Q3 because SG&A fell 8%"
N           = 5  # first 5 tokens
SEP         = "\n" + "=" * 70


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
model.eval()

cfg = model.config
print(SEP)
print("MODEL ARCHITECTURE")
print(SEP)
print(f"  Layers           : {cfg.num_hidden_layers}")
print(f"  d_model          : {cfg.hidden_size}")
print(f"  d_ffn            : {cfg.intermediate_size}")
print(f"  Activation       : {cfg.hidden_act} (SwiGLU = silu gating)")
print(f"  FFN projections  : gate_proj, up_proj, down_proj  ✓")


# ── Tokenize ──────────────────────────────────────────────────────────────────

inputs    = tokenizer(INPUT_TEXT, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens    = [tokenizer.decode([t]) for t in input_ids[0]]
print(f"\nAll tokens : {tokens}")
print(f"Using first {N}: {tokens[:N]}")


# ── Hook: capture input to FFN (post-attention + residual) ────────────────────

captured = {}

def hook_ffn_input(module, input, output):
    captured["ffn_input"]  = input[0].detach()   # x entering MLP
    captured["ffn_output"] = output.detach()       # x leaving MLP (pre-residual)

layer0_mlp = model.model.layers[0].mlp
layer0_mlp.register_forward_hook(hook_ffn_input)

# Hook post_feedforward layernorm to get x just before FFN (after attn residual)
def hook_post_attn(module, input, output):
    captured["post_attn_residual"] = output.detach()

model.model.layers[0].post_attention_layernorm.register_forward_hook(hook_post_attn)


# ── Forward pass ─────────────────────────────────────────────────────────────

with torch.no_grad():
    model(**inputs, output_hidden_states=True)


# ── Extract weights ───────────────────────────────────────────────────────────

gate_w = layer0_mlp.gate_proj.weight.detach()   # [8192, 2048]
up_w   = layer0_mlp.up_proj.weight.detach()     # [8192, 2048]
down_w = layer0_mlp.down_proj.weight.detach()   # [2048, 8192]

x_ffn = captured["ffn_input"][0, :N, :]         # [5, 2048]  — input to MLP


# ── Compute every step manually ───────────────────────────────────────────────

gate   = x_ffn @ gate_w.T                        # [5, 8192]
up     = x_ffn @ up_w.T                          # [5, 8192]
sig    = torch.sigmoid(gate)                     # [5, 8192]
silu   = gate * sig                              # [5, 8192]  SiLU = gate * sigmoid(gate)
gated  = silu * up                               # [5, 8192]  element-wise
out    = gated @ down_w.T                        # [5, 2048]
final  = x_ffn + out                             # [5, 2048]  residual


# ── Print helpers ─────────────────────────────────────────────────────────────

def p6(name, tensor):
    print(f"\n--- {name}  shape={list(tensor.shape)} ---")
    for i in range(N):
        vals = tensor[i, :6].tolist()
        row  = "  ".join(f"{v:9.5f}" for v in vals)
        print(f"  [{i}] {tokens[i]:>10s}  {row}  ...")


# ── Print every step ──────────────────────────────────────────────────────────

print(SEP)
print(f"LAYER 0 FFN — STEP-BY-STEP  (first 5 tokens, first 6 values)")
print(SEP)

print(SEP)
print("STEP 1: Input to FFN  x  (after attention output + residual)")
print("        This is the residual stream entering the MLP.")
p6("x", x_ffn)

print(SEP)
print("STEP 2: gate = x @ gate_proj.T  [5, 8192]")
print("        gate_proj linearly projects x into FFN space.")
print("        These raw values will be gated by sigmoid.")
p6("gate", gate)

print(SEP)
print("STEP 3: up = x @ up_proj.T  [5, 8192]")
print("        up_proj is the value branch — what gets kept or suppressed.")
p6("up", up)

print(SEP)
print("STEP 4: sigmoid(gate) — gate values vs sigmoid side by side")
print("        sigmoid squashes to (0,1). This is the gate control signal.")
print(f"\n{'':>14s}  {'gate raw':>10s}  {'sigmoid':>10s}")
for i in range(N):
    print(f"\n  [{i}] {tokens[i]:>10s}")
    for j in range(6):
        g = gate[i, j].item()
        s = sig[i, j].item()
        print(f"           dim {j:3d}  {g:10.5f}  {s:10.5f}")

print(SEP)
print("STEP 5: SiLU(gate) = gate * sigmoid(gate)  [5, 8192]")
print("        SiLU is smooth, non-monotonic. Negative gates → near-zero output.")
p6("silu", silu)

print(SEP)
print("STEP 6: gated = SiLU(gate) * up  element-wise  [5, 8192]")
print("        The gate controls how much of 'up' passes through.")
p6("gated", gated)

print("\n  --- Position 0 of each token (gate / sigmoid / SiLU / up / result) ---")
for i in range(N):
    g  = gate[i, 0].item()
    s  = sig[i, 0].item()
    sl = silu[i, 0].item()
    u  = up[i, 0].item()
    r  = gated[i, 0].item()
    print(f"  [{i}] {tokens[i]:>10s}  gate={g:8.4f}  sigmoid={s:7.4f}  SiLU={sl:8.4f}  up={u:8.4f}  result={r:8.4f}")

print(SEP)
print("STEP 7: output = gated @ down_proj.T  [5, 2048]")
print("        down_proj projects back from FFN space to d_model.")
p6("output", out)

print(SEP)
print("STEP 8: final = x + output  (residual connection)  [5, 2048]")
print("        Residual add. The FFN adds a delta on top of the input.")
p6("final", final)

print("\n  --- Change vs Step 1 (final - x, first 6 dims) ---")
delta = final - x_ffn
for i in range(N):
    vals = delta[i, :6].tolist()
    signs = ["↑" if v > 0 else "↓" for v in vals]
    row = "  ".join(f"{v:+8.5f}{s}" for v, s in zip(vals, signs))
    print(f"  [{i}] {tokens[i]:>10s}  {row}  ...")

print()
