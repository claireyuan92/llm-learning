# Lesson 7: Decoding Sampling -- temperature / top-k / top-p (treasure: 随机)
# ---------------------------------------------------------------------------
# Generation is not "pick the best", it's "roll a loaded die".
# On one fixed logits vector we apply the three knobs from the lesson,
# in the standard pipeline order: temperature -> top-k -> top-p -> sample.
# Run: python lesson_07_sampling.py   (no args, CPU only)

import torch
import torch.nn.functional as F

WORDS = ["the", "cat", "sat", "dog", "ran", "mat", "moon", "fish", "tree", "star"]
LOGITS = torch.tensor([3.0, 2.2, 1.5, 1.2, 0.8, 0.3, -0.2, -0.8, -1.5, -2.5])


def apply_temperature(logits, t):
    return logits / t  # T->0: needle-sharp (greedy); T big: flat (wild)


def top_k(logits, k):
    v, _ = torch.topk(logits, k)
    return torch.where(logits < v[-1], float("-inf"), logits)


def top_p(logits, p):
    s = torch.sort(logits, descending=True)
    probs, idx = F.softmax(s.values, -1), s.indices
    cum = torch.cumsum(probs, -1)
    # smallest set whose cumulative prob >= p ... plus the first token
    keep = torch.zeros_like(logits, dtype=torch.bool)
    cutoff = torch.searchsorted(cum, p).item()
    keep[idx[:cutoff + 1]] = True
    return torch.where(keep, logits, float("-inf"))


def show(name, logits):
    probs = F.softmax(logits, -1)
    top3 = torch.topk(probs, 3)
    desc = ", ".join(f"{WORDS[i]}:{p:.2f}" for p, i in zip(top3.values, top3.indices))
    print(f"{name:28} top-3 -> {desc}")


def sample(logits, n=6, seed=0):
    torch.manual_seed(seed)
    p = F.softmax(logits, -1)
    return [WORDS[i] for i in torch.multinomial(p, n, replacement=True).tolist()]


def main():
    print("base distribution (T=1, no truncation):")
    show("raw", LOGITS)
    print("\n--- temperature knob ---")
    for t in (0.3, 1.0, 2.0):
        show(f"T={t}", apply_temperature(LOGITS, t))
    print("\n--- truncation (on T=1 logits) ---")
    show("top-k=3", top_k(LOGITS, 3))
    show("top-p=0.9", top_p(LOGITS, 0.9))
    print("\n--- full pipeline: T -> top-k -> top-p -> roll ---")
    pipe = top_p(top_k(apply_temperature(LOGITS, 1.0), 5), 0.9)
    for t, seed in ((0.3, 0), (1.0, 1), (2.0, 2)):
        pipe_t = top_p(top_k(apply_temperature(LOGITS, t), 5), 0.9)
        print(f"T={t}: {' '.join(sample(pipe_t, seed=seed))}")
    print("\nT->0 ~= repeater (greedy), T big ~= nonsense machine. "
          "Good models live in 'restrained randomness'.")


if __name__ == "__main__":
    main()
