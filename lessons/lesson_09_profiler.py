# Lesson 9: Profiler -- finding the attention bottleneck (treasure: 瓶颈)
# ---------------------------------------------------------------------------
# "Measure first, optimize second." We time attention forward passes at
# growing sequence lengths and watch the cost grow quadratically --
# the O(n^2) memory wall the lesson's profiler report points at.
# (Uses time.perf_counter; torch.profiler would show the same story in
#  per-kernel detail on a GPU build.)
# Run: python lesson_09_profiler.py   (no args, CPU only, < 1 min)

import time

import torch
import torch.nn.functional as F


def bench(fn, reps=5):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    B, H, D = 1, 2, 64
    print(f"{'seq_len':>8} {'ms/forward':>11} {'x vs prev':>9}  (quadratic => ~4x)")
    prev = None
    for n in (128, 256, 512, 1024, 2048):
        q = torch.randn(B, H, n, D)
        k = torch.randn(B, H, n, D)
        v = torch.randn(B, H, n, D)
        ms = bench(lambda: F.scaled_dot_product_attention(q, k, v), reps=3)
        ratio = f"{ms / prev:5.2f}x" if prev else "   --"
        attn_mb = B * H * n * n * 4 / 1e6  # fp32 attention matrix
        print(f"{n:>8} {ms:>11.1f} {ratio:>9}   attn matrix ~{attn_mb:.0f} MB")
        prev = ms
    print("\nTime roughly quadruples when n doubles: the n x n score matrix "
          "is the bottleneck.\nIt's memory-bound -- the GPU isn't too slow, "
          "it's waiting on data.")


if __name__ == "__main__":
    main()
