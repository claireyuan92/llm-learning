"""Run sampling experiments from the command line."""

from __future__ import annotations

import argparse

import numpy as np

from sampling import SamplingConfig, sample, sampled_distribution


TOKENS = ("red", "blue", "green", "yellow", "purple", "orange", "black")
LOGITS = np.array([3.2, 2.7, 1.9, 1.1, 0.4, -0.2, -1.0])


def probability_bars(probabilities: np.ndarray, width: int = 30) -> None:
    for token, probability in zip(TOKENS, probabilities):
        blocks = round(float(probability) * width)
        print(f"{token:7} {'#' * blocks:<30} {probability:.3%}")


def print_distribution(title: str, probabilities: np.ndarray) -> None:
    print(f"\n{title}")
    probability_bars(probabilities)
    print(f"entropy: {-float(np.sum(probabilities * np.log(probabilities + 1e-12))):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore LLM sampling strategies.")
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--samples", "-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    probabilities = sampled_distribution(LOGITS, config)
    print_distribution(
        f"Sampling distribution (temperature={config.temperature}, "
        f"top-k={config.top_k}, top-p={config.top_p})",
        probabilities,
    )

    if args.samples > 0:
        rng = np.random.default_rng(args.seed)
        counts = np.zeros(LOGITS.size, dtype=int)
        for _ in range(args.samples):
            counts[sample(LOGITS, config, rng)] += 1
        print(f"\n{args.samples} samples (seed={args.seed})")
        for token, count in zip(TOKENS, counts):
            print(f"{token:7} {count:5} ({count / args.samples:.1%})")


if __name__ == "__main__":
    main()
