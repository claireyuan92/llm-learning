import unittest

import numpy as np

from kv_cache import KVCache, ToyDecoder, causal_attention, kv_cache_bytes


class KVCacheTests(unittest.TestCase):
    def test_cached_decode_matches_naive_decode(self):
        from kv_cache import DecoderConfig

        decoder = ToyDecoder(DecoderConfig(seed=3))
        prompt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        naive = decoder.decode_naive(prompt, max_new_tokens=6)
        cached = decoder.decode_cached(prompt, max_new_tokens=6)

        np.testing.assert_array_equal(naive, cached)

    def test_prefill_with_cache_matches_full_prefill(self):
        from kv_cache import DecoderConfig

        decoder = ToyDecoder(DecoderConfig(seed=11))
        tokens = np.array([[2, 8, 4]], dtype=np.int64)
        cache_logits = decoder.prefill(tokens, KVCache())

        cache = KVCache()
        incremental_logits = []
        for token in tokens.T:
            incremental_logits.append(decoder.prefill(token[:, None], cache)[:, -1:, :])
        incremental_logits = np.concatenate(incremental_logits, axis=1)

        np.testing.assert_allclose(cache_logits, incremental_logits, rtol=1e-5, atol=1e-5)

    def test_causal_attention_masks_future_tokens(self):
        q = np.ones((1, 1, 3, 2), dtype=np.float32)
        k = np.ones((1, 1, 3, 2), dtype=np.float32)
        v = np.arange(3, dtype=np.float32)[None, None, :, None]
        v = np.repeat(v, 2, axis=-1)

        output = causal_attention(q, k, v)
        np.testing.assert_allclose(output[:, :, 0, :], 0.0)
        np.testing.assert_allclose(output[:, :, 1, :], 0.5)

    def test_memory_formula_counts_keys_and_values(self):
        bytes_needed = kv_cache_bytes(
            batch_size=2,
            sequence_length=128,
            num_layers=6,
            num_heads=8,
            head_dim=64,
        )
        expected = 2 * 2 * 128 * 6 * 8 * 64 * 4
        self.assertEqual(bytes_needed, expected)


if __name__ == "__main__":
    unittest.main()
