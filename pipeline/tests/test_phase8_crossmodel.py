"""Cross-model direction-survival probe: the difference-in-means math and verdict.
Pure numerics — no models."""
import unittest

from pipeline import phase8_crossmodel as x


class DirectionSurvivalTest(unittest.TestCase):
    def test_full_survival_when_unlearned_equals_injected(self):
        base = [0.0, 0.0, 0.0]
        inj = [1.0, 2.0, 0.0]
        unl = [1.0, 2.0, 0.0]  # identical to injected -> direction fully retained
        s = x.direction_survival(base, inj, unl)
        self.assertAlmostEqual(s["survival"], 1.0)
        self.assertAlmostEqual(s["cosine"], 1.0)
        self.assertEqual(x.verdict(s), "latent-persists")

    def test_zero_survival_when_unlearned_equals_base(self):
        base = [0.0, 0.0, 0.0]
        inj = [1.0, 2.0, 0.0]
        unl = [0.0, 0.0, 0.0]  # reverted to base -> nothing of injection remains
        s = x.direction_survival(base, inj, unl)
        self.assertAlmostEqual(s["survival"], 0.0)
        self.assertEqual(x.verdict(s), "latent-removed")

    def test_half_survival_is_partial(self):
        base = [0.0, 0.0]
        inj = [2.0, 0.0]
        unl = [0.6, 0.0]  # 0.3 of the injection direction -> partial band (0.15..0.5)
        s = x.direction_survival(base, inj, unl)
        self.assertAlmostEqual(s["survival"], 0.3)
        self.assertEqual(x.verdict(s), "partial")

    def test_negative_survival_when_moved_opposite(self):
        base = [0.0, 0.0]
        inj = [1.0, 0.0]
        unl = [-0.5, 0.0]
        s = x.direction_survival(base, inj, unl)
        self.assertLess(s["survival"], 0)
        self.assertEqual(x.verdict(s), "latent-removed")

    def test_uninformative_when_injection_left_no_footprint(self):
        base = [1.0, 1.0]
        inj = [1.0, 1.0]   # injected == base at this layer -> zero injection direction
        unl = [2.0, 3.0]
        s = x.direction_survival(base, inj, unl)
        self.assertEqual(s["inject_norm"], 0.0)
        self.assertEqual(x.verdict(s), "uninformative")

    def test_survival_projects_only_along_injection_direction(self):
        # Orthogonal movement should not count as survival of the injection dir.
        base = [0.0, 0.0]
        inj = [1.0, 0.0]        # injection added along axis 0
        unl = [0.0, 5.0]        # unlearned moved along axis 1 (orthogonal)
        s = x.direction_survival(base, inj, unl)
        self.assertAlmostEqual(s["survival"], 0.0)
        self.assertEqual(x.verdict(s), "latent-removed")


if __name__ == "__main__":
    unittest.main()
