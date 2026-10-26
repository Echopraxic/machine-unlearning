"""Phase 4-R per-cell retuning: job enumeration, result keying, and the Phase 5
per-cell param lookup/fallback. No models — pure orchestration logic."""
import json
import unittest
from unittest import mock

from pipeline import phase4_retune


FAKE_CELLS = [
    {"id": "m1__malware__standard", "model_id": "m1", "capability": "malware", "dose": "standard"},
    {"id": "m1__cve-exploit__standard", "model_id": "m1", "capability": "cve-exploit", "dose": "standard"},
]


class RetuneJobsTest(unittest.TestCase):
    def test_cells_cross_methods(self):
        with mock.patch.object(phase4_retune.phase5, "available_cells", return_value=FAKE_CELLS), \
             mock.patch.object(phase4_retune.phase4, "METHOD_NAMES", ["tvn", "npo"]):
            jobs = phase4_retune.retune_jobs({})
        self.assertEqual(len(jobs), 4)  # 2 cells x 2 methods
        self.assertEqual({(j["cell_id"], j["method"]) for j in jobs},
                         {("m1__malware__standard", "tvn"), ("m1__malware__standard", "npo"),
                          ("m1__cve-exploit__standard", "tvn"), ("m1__cve-exploit__standard", "npo")})

    def test_only_capability_filter(self):
        with mock.patch.object(phase4_retune.phase5, "available_cells", return_value=FAKE_CELLS), \
             mock.patch.object(phase4_retune.phase4, "METHOD_NAMES", ["tvn"]):
            jobs = phase4_retune.retune_jobs({}, only_capability="cve-exploit")
        self.assertEqual([j["cell_id"] for j in jobs], ["m1__cve-exploit__standard"])

    def test_method_list_override(self):
        with mock.patch.object(phase4_retune.phase5, "available_cells", return_value=FAKE_CELLS):
            jobs = phase4_retune.retune_jobs({}, method_list=["dpo_idk"])
        self.assertTrue(all(j["method"] == "dpo_idk" for j in jobs))
        self.assertEqual(len(jobs), 2)


class FrozenParamsLookupTest(unittest.TestCase):
    def test_returns_per_cell_params_when_present(self):
        payload = {"results": {"m1__malware__standard::tvn": {"best_params": {"alpha": 0.42}}}}
        with mock.patch.object(phase4_retune, "_load_existing", return_value=payload["results"]):
            self.assertEqual(
                phase4_retune.frozen_params_for("m1__malware__standard", "tvn"),
                {"alpha": 0.42})

    def test_returns_none_when_absent(self):
        with mock.patch.object(phase4_retune, "_load_existing", return_value={}):
            self.assertIsNone(phase4_retune.frozen_params_for("m1__malware__standard", "tvn"))


class IncrementalResumeTest(unittest.TestCase):
    def test_round_trips_results_file(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "retune.json"
            with mock.patch.object(phase4_retune, "RESULTS_PATH", path):
                phase4_retune._write({"c::tvn": {"best_params": {"alpha": 0.5}}}, n_trials=30)
                got = phase4_retune._load_existing()
        self.assertEqual(got["c::tvn"]["best_params"]["alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()
