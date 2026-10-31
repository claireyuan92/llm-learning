"""Minimal NumPy KV-cache exercise for autoregressive decoding."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def causal_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Compute causal scaled dot-product attention.

    Shapes are (batch, heads, sequence, head_dim).
    """
    scores = np.einsum("bhqd,bhkd->bhqk", q, k)
    scores /= np.sqrt(q.shape[-1])
    mask = np.tril(np.ones((q.shape[-2], k.shape[-2]), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores)
    return np.einsum("bhqk,bhkd->bhqd", weights, v)


class KVCache:
    def __init__(self) -> None:
        self.k: list[np.ndarray] = []
        self.v: list[np.ndarray] = []

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.k:
            if self.k[-1].shape[0] != k.shape[0] or self.k[-1].shape[1] != k.shape[1]:
                raise ValueError("cache batch/head dimensions do not match")
        self.k.append(k)
        self.v.append(v)
        return (
            np.concatenate(self.k, axis=-2),
            np.concatenate(self.v, axis=-2),
        )

    @property
    def length(self) -> int:
        return len(self.k)


@dataclass
class DecoderConfig:
    vocab_size: int = 64
    model_dim: int = 32
    num_heads: int = 4
    seed: int = 7

    @property
    def head_dim(self) -> int:
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        return self.model_dim // self.num_heads


class ToyDecoder:
    """A one-layer causal decoder implemented with NumPy."""

    def __init__(self, config: DecoderConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d = config.model_dim
        h = config.num_heads
        hd = config.head_dim

        def parameter(*shape: int) -> np.ndarray:
            # Small initial weights keep the residual toy model numerically stable
            # as it is repeatedly applied during generation.
            scale = 0.02
            return rng.normal(0, scale, shape).astype(np.float64)

        self.embedding = parameter(config.vocab_size, d)
        self.output_weight = self.embedding.T.copy()
        self.w_q = parameter(d, h * hd).reshape(d, h, hd)
        self.w_k = parameter(d, h * hd).reshape(d, h, hd)
        self.w_v = parameter(d, h * hd).reshape(d, h, hd)
        self.w_o = parameter(h * hd, d)
        self.w_ff_1 = parameter(d, d * 2)
        self.w_ff_2 = parameter(d * 2, d)

    def _qkv(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # (batch, time, heads, head_dim)
        q = np.einsum("btd,dho->btho", x, self.w_q)
        k = np.einsum("btd,dho->btho", x, self.w_k)
        v = np.einsum("btd,dho->btho", x, self.w_v)
        head_shape = (x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim)
        q = q.reshape(head_shape).transpose(0, 2, 1, 3)
        k = k.reshape(head_shape).transpose(0, 2, 1, 3)
        v = v.reshape(head_shape).transpose(0, 2, 1, 3)
        return q, k, v

    def _layer(self, x: np.ndarray, cache: KVCache | None = None) -> np.ndarray:
        residual = x
        q, k, v = self._qkv(x)
        if cache is None:
            attended = causal_attention(q, k, v)
        else:
            k, v = cache.append(k, v)
            scores = np.einsum("bhqd,bhkd->bhqk", q, k)
            scores /= np.sqrt(q.shape[-1])
            query_start = k.shape[-2] - q.shape[-2]
            allowed_columns = np.arange(k.shape[-2])
            allowed_queries = query_start + np.arange(q.shape[-2])
            mask = allowed_columns[None, :] <= allowed_queries[:, None]
            scores = np.where(mask, scores, -np.inf)
            weights = softmax(scores)
            attended = np.einsum("bhqk,bhkd->bhqd", weights, v)

        attended = attended.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.config.model_dim
        )
        x = residual + np.einsum("btm,md->btd", attended, self.w_o)

        residual = x
        hidden = np.maximum(0, np.einsum("btd,df->btf", x, self.w_ff_1))
        return residual + np.einsum("btf,fd->btd", hidden, self.w_ff_2)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("btd,dv->btv", x, self.output_weight)

    def prefill(self, token_ids: np.ndarray, cache: KVCache | None = None) -> np.ndarray:
        x = self.embedding[token_ids]
        x = self._layer(x, cache)
        return self.logits(x)

    def decode_naive(self, prompt: np.ndarray, max_new_tokens: int) -> np.ndarray:
        tokens = prompt.copy()
        for _ in range(max_new_tokens):
            logits = self.prefill(tokens)
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            tokens = np.concatenate((tokens, next_token[:, None]), axis=1)
        return tokens

    def decode_cached(self, prompt: np.ndarray, max_new_tokens: int) -> np.ndarray:
        tokens = prompt.copy()
        cache = KVCache()
        logits = self.prefill(tokens, cache)

        for _ in range(max_new_tokens):
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            tokens = np.concatenate((tokens, next_token[:, None]), axis=1)
            logits = self.prefill(next_token[:, None], cache)
        return tokens


def kv_cache_bytes(
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: np.dtype = np.dtype(np.float32),
) -> int:
    """Return bytes required by K and V only."""
    elements_per_layer = 2 * batch_size * num_heads * sequence_length * head_dim
    return int(elements_per_layer * num_layers * dtype.itemsize)


def benchmark_decode(
    decoder: ToyDecoder,
    prompt: np.ndarray,
    max_new_tokens: int,
    repeats: int = 8,
) -> tuple[float, float]:
    def timing(fn: callable) -> float:
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            fn(prompt.copy(), max_new_tokens)
            samples.append(perf_counter() - start)
        return float(np.median(samples))

    naive = timing(decoder.decode_naive)
    cached = timing(decoder.decode_cached)
    return naive, cached
