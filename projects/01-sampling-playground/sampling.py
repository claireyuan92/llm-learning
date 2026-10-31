"""Utilities for exploring decoding-time sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax distribution."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("logits cannot contain NaN or positive infinity")
    if np.isneginf(logits).all():
        raise ValueError("at least one logit must be finite")

    shifted = logits - np.max(logits)
    return np.exp(shifted) / np.sum(np.exp(shifted))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    return logits / temperature


def apply_top_k(logits: np.ndarray, k: int | None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).copy()
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if k is None:
        return logits
    if k <= 0:
        raise ValueError("top_k must be greater than 0")
    if k >= logits.size:
        return logits

    order = np.argsort(-logits, kind="stable")
    logits[order[k:]] = -np.inf
    return logits


def apply_top_p(logits: np.ndarray, p: float | None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).copy()
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if p is None:
        return logits
    if not 0 < p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    order = np.argsort(logits)[::-1]
    sorted_probs = softmax(logits[order])
    cutoff = np.searchsorted(np.cumsum(sorted_probs), p, side="left") + 1
    logits[order[cutoff:]] = -np.inf
    return logits


def sampled_distribution(logits: np.ndarray, config: SamplingConfig) -> np.ndarray:
    temperature_logits = apply_temperature(logits, config.temperature)
    top_k_logits = apply_top_k(temperature_logits, config.top_k)
    top_p_logits = apply_top_p(top_k_logits, config.top_p)
    return softmax(top_p_logits)


def sample(
    logits: np.ndarray, config: SamplingConfig, rng: np.random.Generator
) -> int:
    probabilities = sampled_distribution(logits, config)
    return int(rng.choice(probabilities.size, p=probabilities))
