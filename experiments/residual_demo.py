"""
Residual vs non-residual network comparison using only numpy.
d_model=8, 4 layers, seed=42.

Run with: python residual_demo.py
"""
import numpy as np

np.random.seed(42)

D      = 8
N_LAYERS = 4
SEP    = "\n" + "=" * 70


# ── Weights and input ─────────────────────────────────────────────────────────
# Scale down so tanh doesn't saturate immediately

W = [np.random.randn(D, D) * 0.5 for _ in range(N_LAYERS)]
x0 = np.random.randn(D)


# ── Forward passes ────────────────────────────────────────────────────────────

def forward_no_residual(x_in, weights):
    x = x_in.copy()
    snapshots = [x.copy()]
    for W_i in weights:
        x = np.tanh(W_i @ x)
        snapshots.append(x.copy())
    return snapshots   # [x0, x1, x2, x3, x4]


def forward_with_residual(x_in, weights):
    x = x_in.copy()
    snapshots = [x.copy()]
    for W_i in weights:
        x = x + np.tanh(W_i @ x)
        snapshots.append(x.copy())
    return snapshots


snaps_A = forward_no_residual(x0, W)
snaps_B = forward_with_residual(x0, W)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt4(v):
    return "  ".join(f"{x:8.5f}" for x in v[:4])

def norm(v):
    return np.linalg.norm(v)


# ── Print starting x ──────────────────────────────────────────────────────────

print(SEP)
print("STARTING INPUT  x0  (same for both networks)")
print(SEP)
print(f"  first 4 values : {fmt4(x0)}")
print(f"  norm           : {norm(x0):.5f}")


# ── Print per-layer detail ────────────────────────────────────────────────────

for net_label, snaps in [("A — NO RESIDUAL   x = tanh(W @ x)", snaps_A),
                          ("B — WITH RESIDUAL  x = x + tanh(W @ x)", snaps_B)]:
    print(SEP)
    print(f"NETWORK {net_label}")
    print(SEP)
    for layer in range(1, N_LAYERS + 1):
        x_prev = snaps[layer - 1]
        x_cur  = snaps[layer]
        delta  = x_cur - x_prev
        print(f"\n  After layer {layer - 1}:")
        print(f"    x          : {fmt4(x_cur)}   norm={norm(x_cur):.5f}")
        print(f"    delta      : {fmt4(delta)}   delta_norm={norm(delta):.5f}")


# ── Side-by-side summary table ────────────────────────────────────────────────

print(SEP)
print("SIDE-BY-SIDE SUMMARY TABLE")
print(SEP)
print(f"\n  {'Layer':>6}  {'Norm (no resid)':>16}  {'Norm (with resid)':>18}  "
      f"{'Delta (no resid)':>17}  {'Delta (with resid)':>18}")
print("  " + "-" * 82)

for layer in range(1, N_LAYERS + 1):
    norm_A  = norm(snaps_A[layer])
    norm_B  = norm(snaps_B[layer])
    delta_A = norm(snaps_A[layer] - snaps_A[layer - 1])
    delta_B = norm(snaps_B[layer] - snaps_B[layer - 1])
    print(f"  {layer - 1:>6}  {norm_A:>16.5f}  {norm_B:>18.5f}  "
          f"{delta_A:>17.5f}  {delta_B:>18.5f}")


# ── Gradient analysis via finite differences ──────────────────────────────────
# grad[i] ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
# f = sum of final hidden state

EPS = 1e-5

def scalar_output_A(x_in):
    snaps = forward_no_residual(x_in, W)
    return snaps[-1].sum()

def scalar_output_B(x_in):
    snaps = forward_with_residual(x_in, W)
    return snaps[-1].sum()

grad_A = np.zeros(D)
grad_B = np.zeros(D)

for i in range(D):
    e = np.zeros(D)
    e[i] = 1.0
    grad_A[i] = (scalar_output_A(x0 + EPS * e) - scalar_output_A(x0 - EPS * e)) / (2 * EPS)
    grad_B[i] = (scalar_output_B(x0 + EPS * e) - scalar_output_B(x0 - EPS * e)) / (2 * EPS)

grad_norm_A = norm(grad_A)
grad_norm_B = norm(grad_B)
pct_more    = 100.0 * (grad_norm_B - grad_norm_A) / grad_norm_A if grad_norm_A > 0 else float('inf')

print(SEP)
print("GRADIENT ANALYSIS  (finite differences, eps=1e-5)")
print("Gradient of sum(x_final) w.r.t. x_input")
print(SEP)
print(f"\n  Network A (no residual):")
print(f"    gradient vector (first 4): {fmt4(grad_A)}")
print(f"    gradient norm            : {grad_norm_A:.6f}")
print(f"\n  Network B (with residual):")
print(f"    gradient vector (first 4): {fmt4(grad_B)}")
print(f"    gradient norm            : {grad_norm_B:.6f}")


# ── Conclusion ────────────────────────────────────────────────────────────────

print(SEP)
print("CONCLUSION")
print(SEP)
print(f"\n  Without residuals: layer 0 gradient = {grad_norm_A:.6f}")
print(f"  With residuals:    layer 0 gradient = {grad_norm_B:.6f}")
print(f"  Residuals preserve {pct_more:.1f}% more gradient signal")

print(SEP)
print("OBSERVATIONS")
print(SEP)
print("""
  1. NORM GROWTH
     Network A (no residual): tanh squashes outputs to (-1, 1) each layer,
     so norms stay bounded and small — the network "forgets" scale.
     Network B (with residual): x accumulates additively each layer, so
     the norm grows steadily, similar to what we saw in the real Llama
     residual stream (0.99 → 105.6 across 16 layers).

  2. DELTA NORMS
     In Network A, delta = tanh(W@x) - x, which can be large early but
     shrinks as tanh saturates. In Network B, delta = tanh(W@x), which
     is the raw FFN output added on top — each layer contributes a
     meaningful, additive increment rather than replacing the vector.

  3. GRADIENT FLOW
     The gradient norm for Network A collapses toward zero with depth
     (vanishing gradient): tanh derivatives are in (0,1], so 4 layers
     of multiplication shrinks the signal significantly.
     Network B's residual path creates a shortcut: the gradient of
     x_final w.r.t. x_input has a direct identity component (∂x/∂x = 1)
     through every skip connection, so the gradient norm stays much larger.

  4. WHY THIS MATTERS FOR LLAMA
     The Llama residual stream shows the same pattern: each layer adds
     a delta (attention output + FFN output) onto the running x vector.
     Layer 15 adds a delta of norm 97 to an x of norm ~20 — that large
     final write is only possible because the residual connection lets
     early information survive undistorted to the final layer, where it
     gets projected to the unembedding space for next-token prediction.
     Without residuals, layer 15 would be processing a heavily distorted,
     norm-collapsed vector with almost no gradient signal reaching layer 0.

  5. TRAINING IMPLICATION
     Residuals make deep networks trainable: gradients can flow back
     through the identity shortcut even when weight layers are poorly
     initialized. This is why ResNet (2015) and then Transformers both
     adopted residual connections as a core architectural choice.
""")
