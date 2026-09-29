# Lesson 3: Scaled Dot-Product Attention (treasure: 注意力)
# ---------------------------------------------------------------------------
# Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, from scratch.
# We verify numerically against torch.nn.functional.scaled_dot_product_attention
# and show what a causal mask does: each position can only "look" backwards.
# Lesson takeaway: scores = how much each query cares about each key;
# the 1/sqrt(d_k) keeps softmax out of saturation.
# Run: python lesson_03_attention.py   (no args, CPU only)

import torch
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


def main():
    torch.manual_seed(7)
    B, T, d = 2, 5, 16
    q = torch.randn(B, T, d)
    k = torch.randn(B, T, d)
    v = torch.randn(B, T, d)

    out, w = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    print("max |mine - torch| =", (out - ref).abs().max().item())
    assert torch.allclose(out, ref, atol=1e-6), "must match torch SDPA"
    print("matches torch.nn.functional.scaled_dot_product_attention ✔")

    # --- causal mask demo: position t may only attend to positions <= t
    causal = torch.tril(torch.ones(T, T))
    _, w_causal = attention(q[:1], k[:1], v[:1], mask=causal)
    print("\nattention weights with causal mask (batch 0):")
    print(w_causal[0].round(decimals=3))
    upper = w_causal[0].triu(diagonal=1).abs().max().item()
    print(f"\nmax weight on 'future' positions: {upper:.1e} "
          "(~0: no peeking into the future ✔)")
    # rows still sum to 1 -> a proper weighted average of V
    print("row sums:", w_causal[0].sum(-1).round(decimals=3).tolist())


if __name__ == "__main__":
    main()
