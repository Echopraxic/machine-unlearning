"""Phase 8 latent-persistence logic tests: the verdict rule and the
self-contained probe on synthetic separable/random features (exercises the real
probe code without loading a model)."""
import unittest

import torch

from pipeline import phase8


class VerdictTest(unittest.TestCase):
    def test_persists_when_probe_stays_near_injected(self):
        # base 0.55, injected 0.95, unlearned 0.90 -> ratio (0.90-0.55)/(0.95-0.55)=0.875 -> persists
        self.assertEqual(phase8.verdict("tvn", 0.55, 0.95, 0.90), "latent-persists")

    def test_removed_when_probe_collapses_to_base(self):
        # unlearned 0.60 ~ base 0.55 -> ratio 0.125 -> removed
        self.assertEqual(phase8.verdict("dpo_idk", 0.55, 0.95, 0.60), "latent-removed")

    def test_reference_variants_not_judged(self):
        self.assertEqual(phase8.verdict("base", 0.5, 0.9, 0.9), "reference")
        self.assertEqual(phase8.verdict("injected", 0.5, 0.9, 0.9), "reference")

    def test_inconclusive_when_injection_added_no_separability(self):
        # injected ~ base -> denominator tiny -> inconclusive (can't judge removal)
        self.assertEqual(phase8.verdict("tvn", 0.80, 0.81, 0.80), "inconclusive")

    def test_ratio_boundary_half_is_persists(self):
        # exactly 0.5 ratio counts as persists (>=)
        self.assertEqual(phase8.verdict("tvn", 0.50, 0.90, 0.70), "latent-persists")


class ProbeTest(unittest.TestCase):
    def test_separable_features_probe_high(self):
        # Two clearly separated Gaussian blobs -> probe should classify held-out well.
        torch.manual_seed(0)
        d = 16
        pos = torch.randn(30, d) + 3.0     # forget cluster
        neg = torch.randn(30, d) - 3.0     # benign cluster
        X = torch.cat([pos, neg]); y = torch.cat([torch.ones(30), torch.zeros(30)])
        acc = phase8.fit_eval_probe(X, y, torch)
        self.assertGreaterEqual(acc, 0.9)

    def test_random_features_probe_near_chance(self):
        # No signal -> held-out accuracy should be near 0.5 (not near 1.0).
        torch.manual_seed(1)
        d = 16
        X = torch.randn(60, d); y = torch.cat([torch.ones(30), torch.zeros(30)])
        acc = phase8.fit_eval_probe(X, y, torch)
        self.assertLessEqual(acc, 0.85)    # generous upper bound; must not be ~1.0


if __name__ == "__main__":
    unittest.main()
