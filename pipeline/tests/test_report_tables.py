"""Tests for the Phase 10 table generators: shaping (seed aggregation, joins,
row counts) plus Markdown/LaTeX rendering and escaping."""
import unittest

from pipeline import report_tables as rt


class MainResultsRowsTest(unittest.TestCase):
    def _phase5(self):
        # two methods on one cell: tvn (1 seed), dpo_idk (2 seeds)
        return {"jobs": {
            "m1-x__malware__standard__tvn__seed0": {
                "target": "m1-x__malware__standard", "method": "tvn",
                "forget_score": 0.7, "utility": 1.0, "passes_utility_floor": True},
            "m1-x__malware__standard__dpo_idk__seed0": {
                "target": "m1-x__malware__standard", "method": "dpo_idk",
                "forget_score": 1.0, "utility": 0.0, "passes_utility_floor": False},
            "m1-x__malware__standard__dpo_idk__seed1": {
                "target": "m1-x__malware__standard", "method": "dpo_idk",
                "forget_score": 1.0, "utility": 0.0, "passes_utility_floor": False},
        }}

    def _phase9(self):
        return {"per_cell_method": [
            {"cell": "m1-x__malware__standard", "method": "tvn",
             "p6_verdict": "hidden", "p6_recovery": 0.3, "overall_verdict": "hidden"},
            {"cell": "m1-x__malware__standard", "method": "dpo_idk",
             "p6_verdict": "hidden", "p6_recovery": 0.0, "overall_verdict": "destroyed"},
        ]}

    def test_one_row_per_cell_method_with_seed_counts(self):
        rows = rt.main_results_rows(self._phase5(), self._phase9())
        self.assertEqual(len(rows), 2)  # tvn + dpo_idk collapsed across seeds
        by_m = {r["method"]: r for r in rows}
        self.assertEqual(by_m["TVN"]["n_seeds"], 1)
        self.assertEqual(by_m["DPO-IDK"]["n_seeds"], 2)

    def test_floor_count_and_seed_range_shown(self):
        rows = rt.main_results_rows(self._phase5(), self._phase9())
        dpo = next(r for r in rows if r["method"] == "DPO-IDK")
        self.assertEqual(dpo["floor"], "0/2")
        # multi-seed forget must show a range in brackets (seed-spread honesty)
        self.assertIn("[", dpo["forget"])

    def test_p6_vs_overall_nuance_preserved(self):
        rows = rt.main_results_rows(self._phase5(), self._phase9())
        dpo = next(r for r in rows if r["method"] == "DPO-IDK")
        self.assertEqual(dpo["p6_verdict"], "hidden")            # relearning recovered it
        self.assertEqual(dpo["overall"], "utility-collapsed")    # floor-fail dominates (display label)

    def test_method_without_phase9_shows_dashes(self):
        p5 = {"jobs": {"m1-x__malware__standard__npo__seed0": {
            "target": "m1-x__malware__standard", "method": "npo",
            "forget_score": 1.0, "utility": 0.0, "passes_utility_floor": False}}}
        rows = rt.main_results_rows(p5, {"per_cell_method": []})
        self.assertEqual(rows[0]["p6_verdict"], "—")
        self.assertEqual(rows[0]["overall"], "—")

    def test_base_control_row_prepended_from_phase7(self):
        phase7 = {"results": {"m1-x__malware__standard__base": {
            "target": "m1-x__malware__standard", "capability": "malware",
            "kind": "base", "clean_score": 0.0667}}}
        rows = rt.main_results_rows(self._phase5(), self._phase9(), phase7)
        self.assertEqual(len(rows), 3)  # base + tvn + dpo_idk
        base = rows[0]
        self.assertEqual(base["method"], "Base (control)")
        self.assertEqual(base["forget"], "0.067")
        self.assertEqual(base["utility"], "—")
        self.assertEqual(base["overall"], "—")

    def test_base_control_row_absent_without_phase7(self):
        rows = rt.main_results_rows(self._phase5(), self._phase9())
        self.assertEqual(len(rows), 2)
        self.assertNotIn("Base (control)", [r["method"] for r in rows])

    def test_base_control_rows_skip_non_base_variants(self):
        phase7 = {"results": {"m1-x__malware__standard__tvn": {
            "target": "m1-x__malware__standard", "capability": "malware",
            "kind": "unlearned", "clean_score": 0.9}}}
        self.assertEqual(rt.base_control_rows(phase7), [])


class InjectionRowsTest(unittest.TestCase):
    def test_cve_gating_and_status(self):
        rows = rt.injection_rows(
            None, {"results": {}}, {"results": {}},
            {"results": {"c": {"target": "m1-llama-3.2-1b-instruct__cve-exploit__standard",
                               "final_capability_rate": 0.84, "epochs_run": 4}}})
        self.assertEqual(rows[0]["gating"], "capability")
        self.assertEqual(rows[0]["willingness"], "n/a")
        self.assertEqual(rows[0]["capability"], "0.84")


class VerdictRelabelTest(unittest.TestCase):
    def _p5(self):
        return {"jobs": {"m1-x__malware__standard__dpo_idk__seed0": {
            "target": "m1-x__malware__standard", "method": "dpo_idk",
            "forget_score": 1.0, "utility": 0.0, "passes_utility_floor": False}}}

    def _p9(self):
        return {"per_cell_method": [{"cell": "m1-x__malware__standard", "method": "dpo_idk",
                "p6_verdict": "hidden", "p6_recovery": 0.0, "overall_verdict": "destroyed"}]}

    def test_overall_destroyed_renders_as_utility_collapsed(self):
        rows = rt.main_results_rows(self._p5(), self._p9())
        dpo = next(r for r in rows if r["method"] == "DPO-IDK")
        self.assertEqual(dpo["overall"], "utility-collapsed")   # display label
        self.assertEqual(dpo["p6_verdict"], "hidden")           # P6 self-norm artifact preserved

    def test_internal_token_untouched_in_state(self):
        # display mapping must not mutate the source phase9 verdict tokens
        p9 = self._p9()
        rt.main_results_rows(self._p5(), p9)
        self.assertEqual(p9["per_cell_method"][0]["overall_verdict"], "destroyed")

    def test_display_verdict_handles_joined_and_unknown(self):
        self.assertEqual(rt.rs.display_verdict("hidden/destroyed"), "hidden/utility-collapsed")
        self.assertEqual(rt.rs.display_verdict("removed"), "removed")
        self.assertIsNone(rt.rs.display_verdict(None))


class WilsonCiRowsTest(unittest.TestCase):
    def test_wilson_interval_brackets_point_estimate(self):
        lo, hi = rt._wilson(6, 15)
        self.assertLess(lo, 6 / 15)
        self.assertGreater(hi, 6 / 15)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_rows_cve_not_chance_shifted(self):
        # Phase-5 cve forget_score is raw MCQ accuracy, so the interval is taken on it
        # directly (a +0.25 shift would put it around the wrong proportion).
        p5 = {"jobs": {"m1-x__cve-exploit__standard__tvn__seed0": {
            "target": "m1-x__cve-exploit__standard", "method": "tvn",
            "capability": "cve-exploit", "forget_score": 0.4}}}
        rows = rt.wilson_ci_rows(p5)
        self.assertEqual(rows[0]["n_items"], rt.SCORING_N)
        self.assertEqual(rows[0]["forget"], "0.400")
        lo, hi = rt._wilson(6, 15)
        self.assertEqual(rows[0]["ci95"], f"[{lo:.3f}, {hi:.3f}]")


class ReferenceRowsTest(unittest.TestCase):
    def test_base_and_injected_rows_from_phase9_held_out_scores(self):
        p9 = {"per_cell_method": [
            {"cell": "m1-x__cve-exploit__standard", "method": "tvn",
             "p5_base_on_P": 0.3333, "p5_injected_on_P": 0.8},
            {"cell": "m1-x__cve-exploit__standard", "method": "npo",
             "p5_base_on_P": 0.3333, "p5_injected_on_P": 0.8}]}
        rows = rt.reference_rows(p9)
        self.assertEqual([r["method"] for r in rows], ["Base (control)", "Injected"])
        self.assertEqual(rows[0]["forget"], "0.333")   # raw scale, one row per cell

    def test_phase9_reference_rows_take_precedence_over_phase7(self):
        p5 = {"jobs": {"m1-x__malware__standard__tvn__seed0": {
            "target": "m1-x__malware__standard", "method": "tvn",
            "forget_score": 0.7, "utility": 1.0, "passes_utility_floor": True}}}
        p9 = {"per_cell_method": [{"cell": "m1-x__malware__standard", "method": "tvn",
                                   "p5_base_on_P": 0.62, "p5_injected_on_P": 1.0}]}
        p7 = {"results": {"b": {"target": "m1-x__malware__standard", "kind": "base",
                                "clean_score": 0.66}}}
        rows = rt.main_results_rows(p5, p9, p7)
        self.assertEqual(rows[0]["forget"], "0.620")


class RetuneRowsTest(unittest.TestCase):
    def test_per_cell_rows_sorted_with_display_names(self):
        rows = rt.retune_rows({"results": {
            "c::tvn": {"cell_id": "m1-llama-3.2-1b-instruct__malware__standard", "method": "tvn",
                       "n_trials": 30, "best_params": {"alpha": 0.7154},
                       "best_forget_score": 0.56, "best_utility": 1.0},
            "c::ga": {"cell_id": "m1-llama-3.2-1b-instruct__malware__standard",
                      "method": "ga_graddiff", "n_trials": 30,
                      "best_params": {"lr": 4.4e-05, "num_steps": 20},
                      "best_forget_score": 1.0, "best_utility": 1e-28}}})
        self.assertEqual([r["method"] for r in rows], ["GA", "TVN"])
        self.assertEqual(rows[0]["utility"], "<0.001")
        self.assertEqual(rows[1]["params"], "alpha=0.715")


class ComputeBudgetRowsTest(unittest.TestCase):
    def test_rows_cover_all_phases(self):
        rows = rt.compute_budget_rows()
        self.assertGreaterEqual(len(rows), 7)   # tuning, unlearning, control + treatment relearning, query, functional, probe (+ judge if present)
        self.assertTrue(all("phase" in r and "total_h" in r for r in rows))


class HyperparamRowsTest(unittest.TestCase):
    def test_params_flattened(self):
        rows = rt.hyperparam_rows({"methods": {"tvn": {
            "best_params": {"alpha": 0.5488}, "best_forget_score": 0.533,
            "best_utility": 1.0, "n_trials": 3}}})
        self.assertEqual(rows[0]["method"], "tvn")
        self.assertIn("alpha=0.5488", rows[0]["params"])


class RenderingTest(unittest.TestCase):
    def test_markdown_has_header_and_separator(self):
        md = rt.to_markdown(["A", "B"], [{"a": 1, "b": 2}], ["a", "b"])
        lines = md.strip().splitlines()
        self.assertEqual(lines[0], "| A | B |")
        self.assertTrue(set(lines[1]) <= set("| -"))

    def test_latex_escapes_underscores_and_tilde(self):
        tex = rt.to_latex(["M"], [{"m": "dpo_idk ~1.0"}], ["m"], "cap", "tab:x")
        self.assertIn(r"dpo\_idk", tex)
        self.assertIn(r"$\sim$", tex)      # '~' -> approx, not active char
        self.assertIn(r"\begin{tabular}", tex)


if __name__ == "__main__":
    unittest.main()


class TexEscapeUnicodeTest(unittest.TestCase):
    def test_no_raw_non_ascii_or_math_chars(self):
        out = rt._tex_escape("llama-1B·cve — ∞ [0.1–0.2] <a-b>/||c||^2 τ≥0.7 x_y")
        self.assertTrue(all(ord(ch) < 128 for ch in out), out)
        for bad in ("^2", "<a", "||"):
            self.assertNotIn(bad, out)
        self.assertIn(r"$\infty$", out)
        self.assertIn(r"\_", out)


class InjectionFromLogTest(unittest.TestCase):
    LOG = (
        "[phase3] === injecting m1-llama-3.2-1b-instruct__malware__standard ===\n"
        "[epoch 1/3] avg_loss=0.2906 gating_score=1.0 elapsed=7.5s\n"
        "[phase3] === injecting m1-llama-3.2-1b-instruct__cve-exploit__standard ===\n"
        "[epoch 1/3] avg_loss=9.8326 gating_score=0.32 elapsed=2.4s\n"
        "[epoch 2/3] avg_loss=1.5325 gating_score=0.44 elapsed=2.4s\n"
        "[epoch 3/3] avg_loss=1.1494 gating_score=0.56 elapsed=2.4s\n")

    def _parse(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "parse_injection_log",
            Path(__file__).resolve().parents[2] / "tools" / "parse_injection_log.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.parse(self.LOG)

    def test_parse_final_score_epochs_and_threshold(self):
        r = self._parse()
        cve = r["m1-llama-3.2-1b-instruct__cve-exploit__standard"]
        self.assertEqual((cve["final_gating_score"], cve["epochs_run"]), (0.56, 3))
        self.assertFalse(cve["reached_threshold"])        # 0.56 < 0.70
        self.assertTrue(r["m1-llama-3.2-1b-instruct__malware__standard"]["reached_threshold"])

    def test_rows_flag_below_threshold_and_add_excluded(self):
        rows = rt.injection_rows_from_log({"results": self._parse()})
        status = {r["cell"]: r["status"] for r in rows}
        self.assertIn("below threshold", status.values())
        self.assertTrue(any(s.startswith("EXCLUDED") for s in status.values()))
        self.assertEqual(rows[0]["gating"], "willingness")  # malware sorts first
