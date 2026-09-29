# Lesson 10: GPU architecture -- SM / warp / memory hierarchy (treasure: 并行)
# ---------------------------------------------------------------------------
# The lesson's iron rule: the closer data sits to compute, the faster it is
# (registers > shared > L2 > HBM > host). We demonstrate the same principle
# wherever we can measure it:
#   * on CUDA: coalesced vs strided global-memory copy bandwidth
#   * on CPU : sequential vs strided array access (cache locality)
# Either way: layout and access pattern decide how fast the "army" marches.
# Run: python lesson_10_gpu_arch.py   (no args; never crashes without a GPU)

import time

import torch


def bench(fn, reps=7):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return min(ts)  # best-of: least noise from the OS


def main():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        n = 64 * 1024 * 1024
        a = torch.arange(n, dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        ms_seq = bench(lambda: b.copy_(a))                    # coalesced
        ms_str = bench(lambda: b[::16].copy_(a[::16]))        # strided
        gb = lambda ms, k: (n * 4 * k / 1e9) / (ms / 1e3)     # GB/s
        print(f"coalesced copy: {ms_seq:7.2f} ms  ({gb(ms_seq, 2):6.1f} GB/s)")
        print(f"strided   copy: {ms_str:7.2f} ms  ({gb(ms_str, 2 / 16):6.1f} GB/s)")
        print(f"\nstrided is ~{ms_str / ms_seq:.1f}x slower: a warp's 32 threads "
              "must march in step -- scattered addresses serialize the march.")
        return

    # --- CPU fallback: same lesson via cache locality -----------------------
    # Row-major vs column-major traversal of a 64 MB matrix.
    # Column walk jumps 16 KB per element: every touch misses the cache and
    # defeats the prefetcher -- the CPU version of "uncoalesced access".
    a = torch.randn(4096, 4096)  # 64 MB, row-major (C-contiguous)
    ms_row = bench(lambda: a.sum(dim=1).sum())  # sequential along rows
    ms_col = bench(lambda: a.sum(dim=0).sum())  # strided down columns
    assert torch.allclose(a.sum(dim=1).sum(), a.sum(dim=0).sum())
    print(f"row-major traversal (sequential): {ms_row:7.1f} ms")
    print(f"column-major traversal (strided): {ms_col:7.1f} ms")
    print(f"\nSame data, same math -- but the strided walk is "
          f"~{ms_col / ms_row:.1f}x slower:")
    print("sequential touches stream through cache lines at full bandwidth;")
    print("the strided walk jumps 16 KB per element, missing cache every time.")
    print("\nSame iron rule as the GPU memory hierarchy: registers > shared > "
          "L2 > HBM > host -- performance is about keeping data close to "
          "compute, and accessing it in marching order.")


if __name__ == "__main__":
    main()
