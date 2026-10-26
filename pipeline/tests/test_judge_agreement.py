import unittest

from pipeline import judge_agreement as ja


class KappaTest(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        a = [0, 1, 2, 3, 2, 1, 0, 3]
        self.assertAlmostEqual(ja.cohen_kappa(a, a), 1.0)
        self.assertAlmostEqual(ja.cohen_kappa(a, a, "none"), 1.0)

    def test_known_unweighted_value(self):
        # classic 2x2: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
        a = [1] * 25 + [0] * 25
        b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
        self.assertAlmostEqual(ja.cohen_kappa(a, b, "none", k=2), 0.4, places=6)

    def test_quadratic_penalizes_far_disagreement_more(self):
        a = [0, 1, 2, 3] * 5
        near = [min(3, x + 1) if i % 4 == 0 else x for i, x in enumerate(a)]
        far = [3 - x if i % 4 == 0 else x for i, x in enumerate(a)]
        self.assertGreater(ja.cohen_kappa(a, near), ja.cohen_kappa(a, far))

    def test_none_pairs_dropped_not_zeroed(self):
        a = [0, 1, 2, 3, None]
        b = [0, 1, 2, 3, 3]
        self.assertAlmostEqual(ja.cohen_kappa(a, b), 1.0)
        s = ja.summarize(a, b, n_boot=50)
        self.assertEqual(s["unparseable_in_study"], 1)
        self.assertEqual(s["n_paired"], 4)

    def test_undefined_cases(self):
        self.assertIsNone(ja.cohen_kappa([], []))
        self.assertIsNone(ja.cohen_kappa([2, 2, 2], [2, 2, 2]))   # no variance -> chance=1

    def test_binary_kappa_uses_pass_level(self):
        a = [0, 1, 2, 3, 3, 0]
        b = [1, 0, 3, 2, 2, 0]      # same side of the >=2 line everywhere
        self.assertAlmostEqual(ja.binary_kappa(a, b), 1.0)

    def test_bootstrap_ci_brackets_point_estimate(self):
        a = [0, 1, 2, 3] * 10
        b = [x if i % 5 else (x + 1) % 4 for i, x in enumerate(a)]
        k = ja.cohen_kappa(a, b)
        lo, hi = ja.bootstrap_ci(a, b, ja.cohen_kappa, n_boot=300)
        self.assertLessEqual(lo, k)
        self.assertGreaterEqual(hi, k)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ja.cohen_kappa([1, 2], [1])

    def test_bands(self):
        self.assertEqual(ja.landis_koch(0.85), "almost perfect")
        self.assertEqual(ja.landis_koch(-0.1), "poor")


if __name__ == "__main__":
    unittest.main()
