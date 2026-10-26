"""unlearning_methods tests: apply_tvn is pure tensor arithmetic and fully
testable without loading any model. The four gradient-based methods
(ga_graddiff, npo, rmu, dpo_idk) were sanity-checked manually against real
Gemma weights with toy data (loss changes, no crashes) rather than unit-tested
here — loading a real model per test would make the suite far too slow."""

import unittest

import torch

from pipeline import unlearning_methods as um


class ApplyTvnTest(unittest.TestCase):
    def test_alpha_zero_is_noop(self):
        injected = {"w": torch.tensor([5.0, 6.0])}
        base = {"w": torch.tensor([1.0, 2.0])}
        result = um.apply_tvn(injected, base, alpha=0.0)
        self.assertTrue(torch.equal(result["w"], injected["w"]))

    def test_alpha_one_fully_reverts_to_base(self):
        injected = {"w": torch.tensor([5.0, 6.0])}
        base = {"w": torch.tensor([1.0, 2.0])}
        result = um.apply_tvn(injected, base, alpha=1.0)
        self.assertTrue(torch.allclose(result["w"], base["w"]))

    def test_partial_alpha_interpolates(self):
        injected = {"w": torch.tensor([10.0])}
        base = {"w": torch.tensor([0.0])}
        result = um.apply_tvn(injected, base, alpha=0.5)
        # tau = 10; injected - 0.5*10 = 5
        self.assertTrue(torch.allclose(result["w"], torch.tensor([5.0])))

    def test_shape_mismatch_keeps_injected_unchanged(self):
        injected = {"w": torch.tensor([1.0, 2.0, 3.0])}
        base = {"w": torch.tensor([1.0, 2.0])}  # different shape
        result = um.apply_tvn(injected, base, alpha=1.0)
        self.assertTrue(torch.equal(result["w"], injected["w"]))

    def test_missing_key_in_base_keeps_injected_unchanged(self):
        injected = {"w": torch.tensor([1.0]), "extra": torch.tensor([2.0])}
        base = {"w": torch.tensor([1.0])}
        result = um.apply_tvn(injected, base, alpha=1.0)
        self.assertTrue(torch.equal(result["extra"], injected["extra"]))


if __name__ == "__main__":
    unittest.main()
