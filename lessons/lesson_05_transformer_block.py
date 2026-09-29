# Lesson 5: Transformer Block (treasure: 残差)
# ---------------------------------------------------------------------------
# One block, Pre-LN style (what modern models use):
#   x = x + MHA(LayerNorm(x))
#   x = x + FFN(LayerNorm(x))
# We check output shapes, then prove the residual "express elevator":
# even with the sublayer weights zeroed out, gradients still reach the input
# through the skip connection.
# Run: python lesson_05_transformer_block.py   (no args, CPU only)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        a, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x),
                        need_weights=False)
        x = x + a                      # residual 1: the express elevator
        x = x + self.ffn(self.ln2(x))  # residual 2
        return x


def main():
    torch.manual_seed(0)
    B, T, d = 2, 8, 32
    block = Block(d, n_heads=4)
    x = torch.randn(B, T, d, requires_grad=True)

    y = block(x)
    print("in :", tuple(x.shape), "-> out:", tuple(y.shape))
    assert y.shape == x.shape, "block must preserve shape (stackable!)"

    # --- gradient elevator test: kill the sublayers, keep the residuals
    with torch.no_grad():
        for p in block.mha.parameters():
            p.zero_()
        for p in block.ffn.parameters():
            p.zero_()
    y = block(x)
    y.sum().backward()
    g = x.grad
    print("\nwith sublayer weights = 0, ||grad w.r.t. input|| =",
          f"{g.norm().item():.4f}")
    # LayerNorm(0-ish input)... check the skip path carried the gradient:
    # y = x + 0 + 0 (up to LayerNorm of x) so grad must be nonzero
    assert g.abs().max().item() > 0, "gradient must flow via residual"
    print("gradient still reaches the input through x + Sublayer(x) ✔")
    print("Without residuals, 100 stacked blocks would starve the "
          "bottom layers of gradient.")


if __name__ == "__main__":
    main()
