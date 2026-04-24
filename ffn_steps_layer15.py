"""
Step-by-step FFN computation for Layer 15 (final layer), first 5 tokens.
Also prints comparison summary vs Layer 0.

Run with: python ffn_steps_layer15.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/madhura_anand/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
INPUT_TEXT  = "EBITDA grew 12% in Q3 because SG&A fell 8%"
N           = 5
SEP         = "\n" + "=" * 70


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
model.eval()

inputs    = tokenizer(INPUT_TEXT, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens    = [tokenizer.decode([t]) for t in input_ids[0]]
print(f"All tokens : {tokens}")
print(f"Using first {N}: {tokens[:N]}")


# ── Hook factory ──────────────────────────────────────────────────────────────

def make_hooks(layer_idx):
    captured = {}

    def hook_norm(module, input, output):
        captured["x_normed"] = output.detach()

    def hook_mlp(module, input, output):
        captured["ffn_input"]  = input[0].detach()
        captured["ffn_output"] = output.detach()

    layer = model.model.layers[layer_idx]
    layer.post_attention_layernorm.register_forward_hook(hook_norm)
    layer.mlp.register_forward_hook(hook_mlp)
    return captured


# Register hooks for both layers simultaneously
cap0  = make_hooks(0)
cap15 = make_hooks(15)

with torch.no_grad():
    model(**inputs, output_hidden_states=True)


# ── Step computation for one layer ───────────────────────────────────────────

def compute_steps(layer_idx, captured):
    layer  = model.model.layers[layer_idx]
    mlp    = layer.mlp
    gate_w = mlp.gate_proj.weight.detach()
    up_w   = mlp.up_proj.weight.detach()
    down_w = mlp.down_proj.weight.detach()

    x     = captured["ffn_input"][0, :N, :]
    gate  = x @ gate_w.T
    up    = x @ up_w.T
    sig   = torch.sigmoid(gate)
    silu  = gate * sig
    gated = silu * up
    out   = gated @ down_w.T
    final = x + out

    return dict(x=x, gate=gate, up=up, sig=sig, silu=silu,
                gated=gated, out=out, final=final)


steps0  = compute_steps(0,  cap0)
steps15 = compute_steps(15, cap15)


# ── Print helpers ─────────────────────────────────────────────────────────────

def p6(name, tensor):
    print(f"\n--- {name}  shape={list(tensor.shape)} ---")
    for i in range(N):
        vals = tensor[i, :6].tolist()
        row  = "  ".join(f"{v:9.5f}" for v in vals)
        print(f"  [{i}] {tokens[i]:>20s}  {row}  ...")

def print_steps(label, s):
    print(SEP)
    print(f"{label} — STEP-BY-STEP FFN  (first 5 tokens, first 6 values)")
    print(SEP)

    print(SEP)
    print("STEP 1: Input to FFN  x  (after attention output + residual)")
    p6("x", s["x"])

    print(SEP)
    print("STEP 2: gate = x @ gate_proj.T  [5, 8192]")
    p6("gate", s["gate"])

    print(SEP)
    print("STEP 3: up = x @ up_proj.T  [5, 8192]")
    p6("up", s["up"])

    print(SEP)
    print("STEP 4: sigmoid(gate) — gate values vs sigmoid side by side")
    print(f"\n{'':>22s}  {'gate raw':>10s}  {'sigmoid':>10s}")
    for i in range(N):
        print(f"\n  [{i}] {tokens[i]:>20s}")
        for j in range(6):
            g = s["gate"][i, j].item()
            sv = s["sig"][i, j].item()
            print(f"           dim {j:3d}  {g:10.5f}  {sv:10.5f}")

    print(SEP)
    print("STEP 5: SiLU(gate) = gate * sigmoid(gate)  [5, 8192]")
    p6("silu", s["silu"])

    print(SEP)
    print("STEP 6: gated = SiLU(gate) * up  element-wise  [5, 8192]")
    p6("gated", s["gated"])
    print("\n  --- Position 0 of each token ---")
    for i in range(N):
        g  = s["gate"][i, 0].item()
        sv = s["sig"][i, 0].item()
        sl = s["silu"][i, 0].item()
        u  = s["up"][i, 0].item()
        r  = s["gated"][i, 0].item()
        print(f"  [{i}] {tokens[i]:>20s}  gate={g:8.4f}  sigmoid={sv:7.4f}  SiLU={sl:8.4f}  up={u:8.4f}  result={r:8.4f}")

    print(SEP)
    print("STEP 7: output = gated @ down_proj.T  [5, 2048]")
    p6("output", s["out"])

    print(SEP)
    print("STEP 8: final = x + output  (residual connection)")
    p6("final", s["final"])
    delta = s["final"] - s["x"]
    print("\n  --- Change vs Step 1 (final - x, first 6 dims) ---")
    for i in range(N):
        vals  = delta[i, :6].tolist()
        signs = ["↑" if v > 0 else "↓" for v in vals]
        row   = "  ".join(f"{v:+8.5f}{sg}" for v, sg in zip(vals, signs))
        print(f"  [{i}] {tokens[i]:>20s}  {row}  ...")


# ── Print both layers ─────────────────────────────────────────────────────────

print_steps("LAYER 15 (FINAL)", steps15)


# ── Comparison summary ────────────────────────────────────────────────────────

def stats(s):
    gate_abs  = s["gate"][:, :6].abs().mean().item()
    sig_mean  = s["sig"][:, :6].mean().item()
    silu_abs  = s["silu"][:, :6].abs().mean().item()
    delta_abs = (s["final"] - s["x"])[:, :6].abs().mean().item()
    return gate_abs, sig_mean, silu_abs, delta_abs

g0,  sv0,  sl0,  d0  = stats(steps0)
g15, sv15, sl15, d15 = stats(steps15)

print(SEP)
print("COMPARISON SUMMARY — Layer 0 vs Layer 15")
print("(averaged over first 5 tokens, first 6 dims)")
print(SEP)
print(f"\n{'Metric':45s}  {'Layer 0':>10s}  {'Layer 15':>10s}  {'Change':>10s}")
print("-" * 80)

def row(label, v0, v15):
    diff = v15 - v0
    arrow = "↑" if diff > 0 else "↓"
    print(f"  {label:43s}  {v0:10.5f}  {v15:10.5f}  {diff:+9.5f}{arrow}")

row("Avg absolute gate value",             g0,  g15)
row("Avg sigmoid(gate)",                   sv0, sv15)
row("Avg absolute SiLU(gate)",             sl0, sl15)
row("Avg absolute change (final - x)",     d0,  d15)

print()
print("  Interpretation:")
if g15 > g0:
    print("  ✓ Gate values are MORE extreme in layer 15 — stronger gating signal.")
else:
    print("  · Gate values are similar or smaller in layer 15.")

sig_dist_0  = abs(sv0  - 0.5)
sig_dist_15 = abs(sv15 - 0.5)
if sig_dist_15 > sig_dist_0:
    print("  ✓ Sigmoid further from 0.5 in layer 15 — gates are more decisive.")
else:
    print("  · Sigmoid closer to 0.5 in layer 15 — gates are less decisive.")

if d15 > d0:
    print("  ✓ Larger FFN updates in layer 15 — deeper layers do more work.")
else:
    print("  · Smaller FFN updates in layer 15.")
print()
