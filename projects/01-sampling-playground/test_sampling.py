import unittest

import numpy as np

from sampling import (
    SamplingConfig,
    apply_top_k,
    apply_top_p,
    sample,
    sampled_distribution,
    softmax,
)


class SamplingTests(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        probabilities = softmax(np.array([1000.0, 999.0, -1000.0]))
        self.assertAlmostEqual(probabilities.sum(), 1.0)
        self.assertFalse(np.isnan(probabilities).any())

    def test_temperature_changes_distribution(self):
        logits = np.array([2.0, 1.0, 0.0])
        low = sampled_distribution(logits, SamplingConfig(temperature=0.5))
        high = sampled_distribution(logits, SamplingConfig(temperature=2.0))
        self.assertGreater(low[0], high[0])

    def test_top_k_masks_tail_and_renormalizes(self):
        logits = np.array([4.0, 3.0, 2.0, 1.0])
        probabilities = sampled_distribution(logits, SamplingConfig(top_k=2))
        self.assertEqual(np.count_nonzero(probabilities), 2)
        self.assertAlmostEqual(probabilities.sum(), 1.0)

    def test_top_k_keeps_exactly_k_tied_tokens(self):
        logits = np.array([5.0, 5.0, 5.0, 1.0])
        probabilities = sampled_distribution(logits, SamplingConfig(top_k=2))
        self.assertEqual(np.count_nonzero(probabilities), 2)

    def test_top_p_masks_tokens_outside_nucleus(self):
        logits = np.array([3.0, 2.0, 1.0, 0.0])
        probabilities = sampled_distribution(logits, SamplingConfig(top_p=0.5))
        self.assertGreater(probabilities[0], 0.0)
        self.assertEqual(probabilities[-1], 0.0)

    def test_top_k_and_top_p_apply_independently(self):
        logits = np.array([5.0, 4.0, 3.0, 2.0])
        np.testing.assert_array_equal(apply_top_k(logits, 3)[3:], [-np.inf])
        np.testing.assert_array_equal(apply_top_p(logits, 1.0), logits)

    def test_sampling_is_reproducible(self):
        logits = np.array([2.0, 1.0, 0.0])
        config = SamplingConfig(temperature=0.8, top_k=2)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        values_a = [sample(logits, config, rng_a) for _ in range(20)]
        values_b = [sample(logits, config, rng_b) for _ in range(20)]
        self.assertEqual(values_a, values_b)

    def test_invalid_sampling_config_is_rejected(self):
        logits = np.array([1.0, 0.0])
        with self.assertRaises(ValueError):
            sampled_distribution(logits, SamplingConfig(temperature=0.0))
        with self.assertRaises(ValueError):
            sampled_distribution(logits, SamplingConfig(top_k=0))
        with self.assertRaises(ValueError):
            sampled_distribution(logits, SamplingConfig(top_p=1.1))


if __name__ == "__main__":
    unittest.main()
