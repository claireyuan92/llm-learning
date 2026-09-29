# Lesson 4: Multi-Head Attention (treasure: 多头)
# ---------------------------------------------------------------------------
# One head learns one "way of looking"; MHA runs h heads in parallel and
# fuses them:  MultiHead = Concat(head_1..head_h) W^O.
# We verify our from-scratch MHA against nn.MultiheadAttention (batch_first)
# by copying weights, then print per-head attention weights on a tiny example.
# Run: python lesson_04_multihead.py   (no args, CPU only)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_weights=False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        w = F.softmax(q @ k.transpose(-2, -1) / self.d_k ** 0.5, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        if return_weights:
            return self.wo(o), w
        return self.wo(o)


def main():
    torch.manual_seed(3)
    d_model, n_heads, T, B = 16, 4, 5, 2
    mine = MyMHA(d_model, n_heads)
    x = torch.randn(B, T, d_model)

    # --- verify against torch's MHA by transplanting weights
    ref = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat(
            [mine.wq.weight, mine.wk.weight, mine.wv.weight], dim=0))
        ref.out_proj.weight.copy_(mine.wo.weight)
    o1, _ = ref(x, x, x, need_weights=False)
    o2 = mine(x)
    print("max |mine - nn.MultiheadAttention| =", (o1 - o2).abs().max().item())
    assert torch.allclose(o1, o2, atol=1e-5)
    print("matches nn.MultiheadAttention ✔")

    # --- per-head weights on a tiny example: heads genuinely look differently
    _, w = mine(x[:1], return_weights=True)  # (1, heads, T, T)
    print(f"\nattention weights, query position 2, all {n_heads} heads:")
    for h in range(n_heads):
        print(f"head {h}:", w[0, h, 2].round(decimals=3).tolist())
    print("\nSame input, different focus per head -- "
          "that's the 'detective squad' from the lesson.")


if __name__ == "__main__":
    main()
