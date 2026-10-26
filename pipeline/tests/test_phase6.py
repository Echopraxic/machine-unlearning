"""Phase 6 relearning-attack logic tests: checkpoint-id parsing, target
filtering (only saved TVN/DPO checkpoints), and the removed/hidden/destroyed
verdict rule. No models — pure orchestration logic."""
import unittest
from unittest import mock

from pipeline import config, phase6


class ParseCheckpointTest(unittest.TestCase):
    def test_parses_valid_id(self):
        p = phase6._parse_checkpoint_id("m1-llama-3.2-1b-instruct__malware__standard__tvn__seed0")
        self.assertEqual(p["model_id"], "m1-llama-3.2-1b-instruct")
        self.assertEqual(p["capability"], "malware")
        self.assertEqual(p["method"], "tvn")
        self.assertEqual(p["seed"], 0)

    def test_every_saved_method_is_relearnable(self):
        # Phase 5 now saves all six methods (floor-passing ga/npo/rmu + SAM are attacked).
        from pipeline import phase5
        for meth in phase5.SAVE_METHODS:
            self.assertIsNotNone(phase6._parse_checkpoint_id(
                f"m1-llama-3.2-1b-instruct__malware__standard__{meth}__seed0"), meth)

    def test_rejects_unknown_method(self):
        self.assertIsNone(phase6._parse_checkpoint_id(
            "m1-llama-3.2-1b-instruct__malware__standard__made_up__seed0"))

    def test_rejects_malformed(self):
        self.assertIsNone(phase6._parse_checkpoint_id("garbage"))
        self.assertIsNone(phase6._parse_checkpoint_id("a__b__c__tvn__notaseed"))

    def test_dpo_seed_parsed(self):
        p = phase6._parse_checkpoint_id("m2-qwen2.5-3b-instruct__cve-exploit__standard__dpo_idk__seed4")
        self.assertEqual(p["method"], "dpo_idk")
        self.assertEqual(p["capability"], "cve-exploit")
        self.assertEqual(p["seed"], 4)


class VerdictTest(unittest.TestCase):
    def test_hidden_when_recovers_above_threshold_with_utility(self):
        # willingness 0.5 -> 0.95 (>=0.9 threshold), model still usable => hidden.
        self.assertEqual(phase6._verdict(0.90, 0.5, 0.95, 0.98), "hidden")

    def test_removed_when_stays_below_threshold(self):
        self.assertEqual(phase6._verdict(0.90, 0.4, 0.55, 0.99), "removed")

    def test_destroyed_when_recovery_only_with_broken_utility(self):
        # 'recovers' willingness above threshold but utility below floor => gibberish.
        self.assertEqual(phase6._verdict(0.90, 0.5, 0.95, 0.2), "destroyed")

    def test_threshold_boundary_is_recovery(self):
        # exactly at threshold counts as recovered (hidden), matching >= gating.
        self.assertEqual(phase6._verdict(0.90, 0.3, 0.90, 0.95), "hidden")


class RelearnTargetsTest(unittest.TestCase):
    def test_filters_to_saved_config_dirs_and_valid_ids(self):
        entries = {
            "m1-llama-3.2-1b-instruct__malware__standard__tvn__seed0": True,   # valid + config.json
            "m2-qwen2.5-3b-instruct__cve-exploit__standard__dpo_idk__seed1": True,
            "m1-llama-3.2-1b-instruct__malware__standard__made_up__seed0": True,  # unknown method
            "stray_dir_no_config": False,                                      # no config.json
        }

        class FakeDir:
            def __init__(self, name, has_config):
                self.name = name
                self._has_config = has_config
            def is_dir(self):
                return True
            def __lt__(self, other):          # phase6 does sorted(iterdir())
                return self.name < other.name
            def __truediv__(self, other):
                return mock.Mock(exists=lambda: self._has_config) if other == "config.json" else self

        fake_dirs = [FakeDir(n, c) for n, c in entries.items()]
        fake_root = mock.Mock()
        fake_root.exists.return_value = True
        fake_root.iterdir.return_value = fake_dirs
        with mock.patch.object(phase6, "UNLEARNED_DIR", fake_root):
            names = {t["checkpoint"] for t in phase6.relearn_targets()}
        self.assertIn("m1-llama-3.2-1b-instruct__malware__standard__tvn__seed0", names)
        self.assertIn("m2-qwen2.5-3b-instruct__cve-exploit__standard__dpo_idk__seed1", names)
        self.assertNotIn("m1-llama-3.2-1b-instruct__malware__standard__made_up__seed0", names)
        self.assertNotIn("stray_dir_no_config", names)


class EpochsToThresholdTest(unittest.TestCase):
    def _curve(self, scores):
        return [{"epoch": i, "forget_score": s} for i, s in enumerate(scores)]

    def test_returns_first_crossing_epoch(self):
        self.assertEqual(phase6.epochs_to_threshold(
            self._curve([0.2, 0.5, 0.9, 1.0]), 0.9), 2)

    def test_epoch_zero_counts(self):
        self.assertEqual(phase6.epochs_to_threshold(
            self._curve([0.95, 1.0]), 0.9), 0)

    def test_none_when_never_reached(self):
        self.assertIsNone(phase6.epochs_to_threshold(
            self._curve([0.1, 0.3, 0.5]), 0.9))


class CompareToControlTest(unittest.TestCase):
    def _curve(self, scores):
        return [{"epoch": i, "forget_score": s} for i, s in enumerate(scores)]

    def test_residual_when_base_never_reaches(self):
        t = self._curve([0.4, 0.95, 1.0])          # unlearned crosses at epoch 1
        c = self._curve([0.1, 0.2, 0.3])           # base never crosses
        r = phase6.compare_to_control(t, c, 0.9)
        self.assertEqual(r["call"], "residual")
        self.assertEqual(r["treatment_epochs_to_tau"], 1)
        self.assertIsNone(r["control_epochs_to_tau"])

    def test_residual_faster_when_unlearned_crosses_earlier(self):
        t = self._curve([0.4, 0.95, 1.0])          # epoch 1
        c = self._curve([0.1, 0.5, 0.92])          # epoch 2
        self.assertEqual(phase6.compare_to_control(t, c, 0.9)["call"], "residual-faster")

    def test_indistinguishable_when_same_epoch(self):
        t = self._curve([0.4, 0.5, 0.95])          # epoch 2
        c = self._curve([0.3, 0.6, 0.93])          # epoch 2
        self.assertEqual(phase6.compare_to_control(t, c, 0.9)["call"], "indistinguishable")

    def test_control_faster_qualifies_survival(self):
        t = self._curve([0.2, 0.4, 0.95])          # epoch 2
        c = self._curve([0.2, 0.93, 1.0])          # epoch 1
        self.assertEqual(phase6.compare_to_control(t, c, 0.9)["call"], "control-faster")

    def test_not_recovered_when_unlearned_stays_low(self):
        t = self._curve([0.2, 0.3, 0.4])
        c = self._curve([0.2, 0.3, 0.95])
        self.assertEqual(phase6.compare_to_control(t, c, 0.9)["call"], "not-recovered")

    def test_uninformative_when_base_already_capable(self):
        t = self._curve([0.4, 0.95])
        c = self._curve([0.92, 1.0])               # base at/above tau at epoch 0
        self.assertEqual(phase6.compare_to_control(t, c, 0.9)["call"], "uninformative")

    def test_mean_gap_over_shared_relearning_epochs(self):
        t = self._curve([0.0, 0.8, 1.0])
        c = self._curve([0.0, 0.4, 0.6])
        r = phase6.compare_to_control(t, c, 0.9)
        # shared epochs >=1 are {1,2}: gaps 0.4 and 0.4 -> mean 0.4
        self.assertAlmostEqual(r["mean_gap"], 0.4)
        self.assertEqual(r["shared_epochs"], [1, 2])


class ControlTargetsTest(unittest.TestCase):
    def test_one_control_per_succeeded_standard_cell(self):
        fake_matrix = [
            {"id": "m1__malware__standard", "model_id": "m1", "capability": "malware", "dose": "standard"},
            {"id": "m1__malware__high", "model_id": "m1", "capability": "malware", "dose": "high"},
            {"id": "m3-gemma-3-1b-it__social-engineering__standard",
             "model_id": "m3-gemma-3-1b-it", "capability": "social-engineering", "dose": "standard"},
        ]
        with mock.patch.object(phase6.injection, "injection_matrix", return_value=fake_matrix), \
             mock.patch.object(phase6.config, "EXCLUDED_INJECTION_CELLS",
                               {"m3-gemma-3-1b-it__social-engineering__standard"}), \
             mock.patch.object(phase6.config, "EXCLUDE_MODELS", set()):
            cells = phase6.control_targets({})
        ids = {c["target"] for c in cells}
        self.assertEqual(ids, {"m1__malware__standard"})  # high-dose + excluded cell dropped


class StepCurveTest(unittest.TestCase):
    @staticmethod
    def _c(scores):
        return [{"step": k, "forget_score": v} for k, v in scores.items()]

    def test_steps_to_threshold(self):
        c = self._c({0: 0.1, 1: 0.2, 2: 0.95, 4: 1.0})
        self.assertEqual(phase6.steps_to_threshold(c, 0.9), 2)
        self.assertIsNone(phase6.steps_to_threshold(c, 1.5))

    def test_treatment_faster_than_control_is_positive_gap(self):
        t = self._c({0: 0.1, 1: 0.5, 2: 0.9, 4: 1.0, 8: 1.0})
        c = self._c({0: 0.0, 1: 0.1, 2: 0.2, 4: 0.6, 8: 1.0})
        r = phase6.compare_step_curves(t, c, 0.9)
        self.assertGreater(r["mean_step_gap"], 0)
        self.assertEqual(r["treatment_steps_to_tau"], 2)
        self.assertEqual(r["control_steps_to_tau"], 8)
        self.assertAlmostEqual(r["gap_by_step"][2], 0.7)
        self.assertGreater(r["area_gap"], 0)

    def test_only_shared_steps_compared_and_step0_excluded(self):
        t = self._c({0: 0.9, 1: 0.9})
        c = self._c({0: 0.0, 1: 0.5, 2: 0.9})
        r = phase6.compare_step_curves(t, c, 0.9)
        self.assertEqual(list(r["gap_by_step"]), [1])
        self.assertIsNone(r["area_gap"])     # <2 shared points


if __name__ == "__main__":
    unittest.main()
