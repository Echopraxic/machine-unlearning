"""Mixed-effects analysis: recovers a planted treatment-minus-control effect, handles
missing outcomes, applies Holm, and falls back honestly on degenerate data."""
import random
import unittest

from pipeline import stats_mixed as sm


def _rows(effect_by_cap, cells_per_cap=3, methods=("ga_graddiff", "npo", "dpo_idk"),
          noise=0.03, cell_sd=0.05, seed=0):
    rng = random.Random(seed)
    rows = []
    for cap, eff in effect_by_cap.items():
        for i in range(cells_per_cap):
            cell = f"m{i}__{cap}__standard"
            u = rng.gauss(0, cell_sd)
            for m in methods:
                for sd in range(2):
                    rows.append({"cell": cell, "capability": cap, "method": m, "seed": sd,
                                 "p5_passed_floor": True,
                                 "step_mean_gap": eff + u + rng.gauss(0, noise)})
    return rows


class HolmTest(unittest.TestCase):
    def test_monotone_and_capped(self):
        adj = sm.holm({"a": 0.01, "b": 0.02, "c": 0.5, "d": None})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["b"], 0.04)
        self.assertAlmostEqual(adj["c"], 0.5)
        self.assertIsNone(adj["d"])


class FitTest(unittest.TestCase):
    def test_recovers_planted_effect_and_null(self):
        rows = _rows({"malware": 0.30, "cve-exploit": 0.0, "social-engineering": 0.15})
        info = sm.fit_outcome(sm.build_frame(rows, "step_gap"), "step_gap")
        self.assertEqual(info["status"], "ok")
        mc = info["marginal_capability"]
        self.assertAlmostEqual(mc["malware"]["est"], 0.30, delta=0.08)
        self.assertGreater(mc["malware"]["est"], mc["social-engineering"]["est"])
        self.assertGreater(mc["social-engineering"]["est"], mc["cve-exploit"]["est"])
        self.assertLess(mc["malware"]["p_one_sided_gt0"], 0.05)
        self.assertGreater(mc["cve-exploit"]["p_one_sided_gt0"], 0.05)
        self.assertGreaterEqual(mc["malware"]["p_one_sided_holm"], mc["malware"]["p_one_sided_gt0"])

    def test_floor_failures_and_missing_excluded(self):
        rows = _rows({"malware": 0.2})
        rows[0]["p5_passed_floor"] = False
        rows[1]["step_mean_gap"] = None
        df = sm.build_frame(rows, "step_gap")
        self.assertEqual(len(df), len(rows) - 2)

    def test_direction_survival_does_not_require_floor(self):
        rows = [{"cell": "m0__malware__standard", "capability": "malware", "method": "npo",
                 "seed": 0, "p5_passed_floor": False, "p8_survival": 0.9}]
        self.assertEqual(len(sm.build_frame(rows, "direction_survival")), 1)

    def test_insufficient_data_is_reported_not_crashed(self):
        info = sm.fit_outcome(sm.build_frame([], "step_gap"), "step_gap")
        self.assertEqual(info["status"], "insufficient-data")

    def test_zero_variance_outcome_falls_back_without_crashing(self):
        rows = _rows({"malware": 0.0, "cve-exploit": 0.0}, noise=0.0, cell_sd=0.0)
        info = sm.fit_outcome(sm.build_frame(rows, "step_gap"), "step_gap")
        self.assertEqual(info["status"], "ok")
        self.assertAlmostEqual(info["marginal_capability"]["malware"]["est"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
