"""Phase 10 figure tests: the data-shaping helpers (unit-tested) plus a headless
render smoke test that each figure writes a PNG from synthetic data without error."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import config, phase10


class ShapingTest(unittest.TestCase):
    def test_verdict_counts(self):
        p9 = {"per_cell_method": [
            {"method": "tvn", "overall_verdict": "removed"},
            {"method": "tvn", "overall_verdict": "hidden"},
            {"method": "dpo_idk", "overall_verdict": "hidden"},
        ]}
        c = phase10.verdict_counts(p9)
        self.assertEqual(c["tvn"]["removed"], 1)
        self.assertEqual(c["tvn"]["hidden"], 1)
        self.assertEqual(c["dpo_idk"]["hidden"], 1)

    def test_recovery_series_extracts_points(self):
        p6 = {"results": {"ck1": {"recovery_curve": [
            {"epoch": 0, "forget_score": 0.5}, {"epoch": 1, "forget_score": 0.7}]}}}
        s = phase10.recovery_series(p6)
        self.assertEqual(s["ck1"], [(0, 0.5), (1, 0.7)])

    def test_attack_lift_includes_unlearned_and_injected_not_base(self):
        p7 = {"results": {
            "m1-x__malware__standard__tvn": {"kind": "unlearned", "clean_score": 0.3,
                "best_attacked_score": 0.9, "scores_by_attack": {"none": 0.3, "prefill": 0.9}},
            "m1-x__malware__standard__injected": {"kind": "injected", "clean_score": 1.0,
                "best_attacked_score": 1.0, "scores_by_attack": {"none": 1.0, "prefill": 1.0}},
            "m1-x__malware__standard__base": {"kind": "base", "clean_score": 0.1,
                "best_attacked_score": 0.2, "scores_by_attack": {"none": 0.1}}}}
        rows = phase10.attack_lift_rows(p7)
        kinds = {r["kind"] for r in rows}
        self.assertEqual(kinds, {"unlearned", "injected"})   # base excluded, injected shown
        tvn = next(r for r in rows if r["kind"] == "unlearned")
        self.assertAlmostEqual(tvn["attack"], 0.9)           # best generative lift
        self.assertAlmostEqual(tvn["delta"], 0.6)

    def test_attack_lift_cve_manyshot_signed_delta_is_negative(self):
        # cve: attack must be the many-shot score (which drops below clean), NOT the
        # max() best_attacked_score that would hide the reduction.
        p7 = {"results": {"m1-x__cve-exploit__standard__injected": {
            "kind": "injected", "clean_score": 0.217, "best_attacked_score": 0.217,
            "scores_by_attack": {"none": 0.217, "many_shot": 0.083}}}}
        rows = phase10.attack_lift_rows(p7)
        self.assertAlmostEqual(rows[0]["attack"], 0.083)
        self.assertAlmostEqual(rows[0]["delta"], -0.134, places=3)
        self.assertIn("many-shot", rows[0]["applicable"])

    def test_probe_rows(self):
        p8 = {"results": {"a": {"variant": "tvn", "probe_accuracy": 0.8},
                          "b": {"variant": "base", "probe_accuracy": None}}}
        rows = phase10.probe_rows(p8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["acc"], 0.8)

    def test_empty_inputs_are_safe(self):
        self.assertEqual(phase10.verdict_counts(None), {})
        self.assertEqual(phase10.recovery_series(None), {})
        self.assertEqual(phase10.attack_lift_rows(None), [])
        self.assertEqual(phase10.probe_rows(None), [])


class RenderSmokeTest(unittest.TestCase):
    def test_each_figure_writes_a_png(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(phase10, "FIGURES_DIR", Path(td)):
                p9 = {"per_cell_method": [{"method": "tvn", "overall_verdict": "removed"},
                                          {"method": "tvn", "overall_verdict": "hidden"}]}
                p6 = {"results": {"m1-x__malware__standard__tvn__seed0": {
                    "target": "m1-x__malware__standard", "capability": "malware", "method": "tvn",
                    "seed": 0, "gating_threshold": 0.9,
                    "step_curve": [{"step": 0, "forget_score": 0.5}, {"step": 15, "forget_score": 0.9}]}}}
                p7 = {"results": {"m1-x__malware__standard__tvn": {
                    "kind": "unlearned", "clean_score": 0.3, "best_attacked_score": 0.9}}}
                p8 = {"results": {"m1-x__malware__standard__tvn": {
                    "variant": "tvn", "probe_accuracy": 0.8}}}
                self.assertTrue(os.path.exists(phase10.fig_verdict_summary(p9)))
                self.assertTrue(os.path.exists(phase10.fig_recovery_curves(p6)))
                self.assertTrue(os.path.exists(phase10.fig_attack_lift(p7)))
                self.assertTrue(os.path.exists(phase10.fig_probe_accuracy(p8)))

    def test_missing_data_returns_none_not_crash(self):
        self.assertIsNone(phase10.fig_verdict_summary(None))
        self.assertIsNone(phase10.fig_recovery_curves({"results": {}}))


class C1ShapingTest(unittest.TestCase):
    def test_recovery_series_grouped_averages_seeds(self):
        p6 = {"results": {
            "m1-x__malware__standard__dpo_idk__seed0": {"recovery_curve": [
                {"epoch": 0, "forget_score": 0.2}, {"epoch": 1, "forget_score": 0.6}]},
            "m1-x__malware__standard__dpo_idk__seed1": {"recovery_curve": [
                {"epoch": 0, "forget_score": 0.4}, {"epoch": 1, "forget_score": 1.0}]},
        }}
        g = phase10.recovery_series_grouped(p6)
        self.assertEqual(len(g), 1)  # both seeds fold into one (cell, method) group
        (label, grp), = g.items()
        self.assertEqual(grp["method"], "dpo_idk")
        pts = dict(grp["pts"])                                 # seed-averaged
        self.assertAlmostEqual(pts[0], 0.3)
        self.assertAlmostEqual(pts[1], 0.8)

    def test_injection_wc_rows_cve_willingness_is_none(self):
        rows = phase10.injection_wc_rows(
            {"result": {"target": "m1-llama-3.2-1b-instruct__malware__standard",
                        "final_willingness_rate": 1.0, "final_capability_rate": 0.03}},
            {"results": {}}, {"results": {}},
            {"results": {"c": {"target": "m1-llama-3.2-1b-instruct__cve-exploit__standard",
                               "final_capability_rate": 0.84}}})
        mal = [r for r in rows if r["domain"] == "malware"][0]
        cve = [r for r in rows if r["domain"] == "cve-exploit"][0]
        self.assertEqual(mal["willingness"], 1.0)
        self.assertAlmostEqual(mal["capability"], 0.03)
        self.assertIsNone(cve["willingness"])          # cve is capability-gated
        self.assertAlmostEqual(cve["capability"], 0.84)

    def test_injection_wc_rows_base_control_from_phase7(self):
        phase7 = {"results": {
            "m1-llama-3.2-1b-instruct__malware__standard__base": {
                "target": "m1-llama-3.2-1b-instruct__malware__standard",
                "capability": "malware", "kind": "base", "clean_score": 0.07},
            "m1-llama-3.2-1b-instruct__cve-exploit__standard__base": {
                "target": "m1-llama-3.2-1b-instruct__cve-exploit__standard",
                "capability": "cve-exploit", "kind": "base", "clean_score": 0.0},
        }}
        rows = phase10.injection_wc_rows(
            {"result": {"target": "m1-llama-3.2-1b-instruct__malware__standard",
                        "final_willingness_rate": 1.0, "final_capability_rate": 0.03}},
            {"results": {}}, {"results": {}},
            {"results": {"c": {"target": "m1-llama-3.2-1b-instruct__cve-exploit__standard",
                               "final_capability_rate": 0.84}}}, phase7)
        mal = [r for r in rows if r["domain"] == "malware"][0]
        cve = [r for r in rows if r["domain"] == "cve-exploit"][0]
        self.assertAlmostEqual(mal["base_willingness"], 0.07)
        self.assertIsNone(mal["base_capability"])
        self.assertIsNone(cve["base_willingness"])
        # Phase 7 records cve chance-corrected (acc - 0.25); the figure plots raw accuracy.
        self.assertAlmostEqual(cve["base_capability"], 0.25)

    def test_injection_wc_rows_base_control_none_without_phase7(self):
        rows = phase10.injection_wc_rows(
            None, {"results": {}}, {"results": {}},
            {"results": {"c": {"target": "m1-x__cve-exploit__standard",
                               "final_capability_rate": 0.5}}})
        self.assertIsNone(rows[0]["base_willingness"])
        self.assertIsNone(rows[0]["base_capability"])

    def test_injection_wc_marks_excluded(self):
        excl = next(iter(config.EXCLUDED_INJECTION_CELLS))
        model = excl.split("__")[0]
        rows = phase10.injection_wc_rows(
            None, {"results": {}},
            {"results": {"c": {"target": excl, "final_willingness_rate": 0.6,
                               "final_capability_rate": 0.0}}}, {"results": {}})
        self.assertTrue(rows[0]["excluded"])
        self.assertEqual(rows[0]["model"], model)

    def test_capability_epoch_series_extracts_rising_curve(self):
        s = phase10.capability_epoch_series(
            {"results": {"c": {"target": "m1-x__cve-exploit__standard", "history": [
                {"epoch": 1, "capability_rate": 0.3}, {"epoch": 2, "capability_rate": 0.8}]}}},
            [{"label": "m1·mal", "cap": 0.0, "epoch": 1}])
        self.assertEqual(s["cve"]["m1-x"], [(1, 0.3), (2, 0.8)])
        self.assertEqual(s["generative_finals"][0]["cap"], 0.0)

    def test_scale_probe_rows(self):
        rows = phase10.scale_probe_rows({"results": {
            "3B": {"willingness_rate": 1.0, "capability_rate": 0.0},
            "7B": {"willingness_rate": 1.0, "capability_rate": 0.0}}})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["capability"] == 0.0 for r in rows))

    def test_gemma_se_series(self):
        s = phase10.gemma_se_series(
            {"result": {"willingness_history": [0.52, 0.6, 0.44], "threshold": 0.9}},
            {"llama-1B": 1.0})
        self.assertEqual(s["pts"], [(1, 0.52), (2, 0.6), (3, 0.44)])
        self.assertEqual(s["references"]["llama-1B"], 1.0)


class C1RenderSmokeTest(unittest.TestCase):
    def test_new_c1_figures_write_pngs(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(phase10, "FIGURES_DIR", Path(td)):
                wc = [{"model": "m1-llama-3.2-1b-instruct", "domain": "malware",
                       "willingness": 1.0, "capability": 0.03, "excluded": False,
                       "base_willingness": 0.07, "base_capability": None},
                      {"model": "m1-llama-3.2-1b-instruct", "domain": "cve-exploit",
                       "willingness": None, "capability": 0.84, "excluded": False,
                       "base_willingness": None, "base_capability": 0.0}]
                self.assertTrue(os.path.exists(
                    phase10.fig_injection_willingness_vs_capability(wc)))
                self.assertTrue(os.path.exists(phase10.fig_capability_across_epochs(
                    {"cve": {"m1-x": [(1, 0.3), (2, 0.8)]},
                     "generative_finals": [{"label": "a", "cap": 0.0, "epoch": 1}]})))
                self.assertTrue(os.path.exists(phase10.fig_scale_probe(
                    [{"label": "3B", "willingness": 1.0, "capability": 0.0}])))
                self.assertTrue(os.path.exists(phase10.fig_model_domain_resistance(
                    {"pts": [(1, 0.52), (2, 0.6)], "threshold": 0.9,
                     "references": {"llama-1B": 1.0}})))

    def test_new_c1_figures_none_on_empty(self):
        self.assertIsNone(phase10.fig_injection_willingness_vs_capability([]))
        self.assertIsNone(phase10.fig_scale_probe([]))
        self.assertIsNone(phase10.fig_model_domain_resistance({"pts": []}))


if __name__ == "__main__":
    unittest.main()


class StepRecoveryPanelsTest(unittest.TestCase):
    def test_floor_filter_control_and_step_mapping(self):
        def row(method, seed, curve, target="m1-x__malware__standard"):
            return {"target": target, "capability": "malware", "method": method, "seed": seed,
                    "gating_threshold": 0.9,
                    "step_curve": [{"step": s, "forget_score": f} for s, f in curve]}
        p6 = {"results": [row("tvn", 0, [(0, 0.2), (1, 0.6), (12, 1.0)]),
                          row("tvn", 1, [(0, 0.4), (1, 0.8), (15, 1.0)]),
                          row("npo", 0, [(0, 0.0), (15, 0.0)])]}
        ctrl = {"results": [{"capability": "malware", "step_curve": [
            {"step": 0, "forget_score": 0.6}, {"step": 15, "forget_score": 1.0}]}]}
        p9 = {"per_cell_method": [{"cell": "m1-x__malware__standard", "method": "tvn", "seed": s,
                                   "p5_passed_floor": True} for s in (0, 1)]
              + [{"cell": "m1-x__malware__standard", "method": "npo", "seed": 0,
                  "p5_passed_floor": False}]}
        pnl = phase10.step_recovery_panels(p6, ctrl, p9)["malware"]
        self.assertNotIn("npo", pnl["methods"])                       # sub-floor dropped
        pos, means, ns = zip(*pnl["methods"]["tvn"])
        self.assertEqual(pos, (0, 1, 5))                              # step 12 and 15 = end ep.1
        self.assertAlmostEqual(means[1], 0.7)
        self.assertEqual(ns, (2, 2, 2))
        self.assertEqual(pnl["control"], [(0, 0.6, 1), (5, 1.0, 1)])
        self.assertEqual(pnl["tau"], 0.9)
