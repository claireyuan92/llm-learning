# Lesson 8: SFT + LoRA (treasure: 微调)
# ---------------------------------------------------------------------------
# LoRA = "minimally invasive surgery": freeze the base weight W, learn only
#   delta_W = B @ A   (B: d_out x r,  A: r x d_in,  r tiny)
# B starts at zero so training begins exactly at base-model behavior.
# We compare trainable parameter counts (full vs LoRA) and run one optimizer
# step to prove only A/B move while W stays frozen.
# Run: python lesson_08_lora.py   (no args, CPU only)

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16.0):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=False)
        self.base.weight.requires_grad_(False)          # frozen "baseboard"
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))    # zero-init -> delta=0
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def merged_weight(self):  # W' = W + B A * scale: zero extra latency
        return self.base.weight + (self.B @ self.A) * self.scaling


def main():
    torch.manual_seed(0)
    IN, OUT, R = 2048, 2048, 8  # realistic scale: 2*r/d = 0.78% of full
    lora = LoRALinear(IN, OUT, r=R)

    full = IN * OUT
    lora_n = sum(p.numel() for p in (lora.A, lora.B))
    print(f"full fine-tune params : {full:,}")
    print(f"LoRA trainable params: {lora_n:,}  ({lora_n / full:.2%} of full)")
    assert lora_n < full / 100

    # at init, LoRA layer == base layer exactly
    x = torch.randn(4, IN)
    assert torch.allclose(lora(x), lora.base(x)), "B=0 must mean delta=0"
    print("init check: LoRA(x) == base(x) -- training starts at base behavior ✔")

    # one optimizer step over LoRA params only
    w_before = lora.base.weight.clone()
    opt = torch.optim.AdamW([lora.A, lora.B], lr=1e-3)
    opt.zero_grad()
    lora(x).pow(2).mean().backward()
    opt.step()
    base_moved = (lora.base.weight - w_before).abs().max().item()
    b_moved = (lora.B - 0).abs().max().item()
    print(f"\nafter 1 step: base weight moved by {base_moved:.1e} (frozen ✔)")
    print(f"B moved by {b_moved:.2e} (trainable ✔); "
          f"A grad was 0.0 this step -- expected: with B=0, no gradient "
          f"reaches A yet (that's the price of zero-init).")
    # step 2: now B != 0, so A finally receives gradient too
    opt.zero_grad()
    lora(x).pow(2).mean().backward()
    print(f"step 2 grad norms -> A: {lora.A.grad.norm().item():.4f}, "
          f"B: {lora.B.grad.norm().item():.4f} (both trainable ✔)")
    assert base_moved == 0.0 and lora.A.grad.norm().item() > 0
    print("\nOne base + N tiny adapters = N tasks, hot-swappable. "
          "Merge BA into W at inference for zero extra latency.")


if __name__ == "__main__":
    main()
