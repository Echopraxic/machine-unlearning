import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import config, phase5
from tools import purge_cell_rows

CELL = "m1-llama-3.2-1b-instruct__cve-exploit__standard"
OTHER = "m2-qwen2.5-3b-instruct__cve-exploit__standard"


class PurgeCellRowsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.files = {
            "phase6_relearning_results.json": ("results", {"n_done": 2}),
            "phase5_factorial_results.shard-1.json": ("jobs", {"n_jobs_done": 2}),
            "phase6_control_results.json": ("results", {}),
        }
        for name, (field, extra) in self.files.items():
            rows = {f"{CELL}__dpo_idk__seed0": 1, f"{OTHER}__dpo_idk__seed0": 2}
            (self.state / name).write_text(json.dumps({field: rows, **extra}))

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, name):
        field = self.files[name][0]
        return json.loads((self.state / name).read_text())[field]

    def test_dry_run_changes_nothing(self):
        with mock.patch.object(config, "STATE_DIR", self.state), \
             mock.patch.object(purge_cell_rows.audit, "append") as app:
            removed = purge_cell_rows.purge(CELL, apply=False)
        self.assertEqual(removed["phase6_relearning_results.json"], 1)
        self.assertEqual(len(self._rows("phase6_relearning_results.json")), 2)
        app.assert_not_called()

    def test_apply_removes_only_the_cell_and_keeps_control(self):
        with mock.patch.object(config, "STATE_DIR", self.state), \
             mock.patch.object(purge_cell_rows.audit, "append") as app:
            purge_cell_rows.purge(CELL, apply=True)
        for name in ("phase6_relearning_results.json", "phase5_factorial_results.shard-1.json"):
            self.assertEqual(list(self._rows(name)), [f"{OTHER}__dpo_idk__seed0"])
        data = json.loads((self.state / "phase6_relearning_results.json").read_text())
        self.assertEqual(data["n_done"], 1)
        self.assertEqual(len(self._rows("phase6_control_results.json")), 2)  # never purged
        backups = list((self.state / "backups").glob("purge-*/*.json"))
        self.assertEqual(len(backups), 2)
        app.assert_called_once()

    def test_matching_purges_by_regex_and_honours_keep(self):
        for name, (field, extra) in self.files.items():
            rows = {f"{OTHER}__dpo_idk__seed0": 1, f"{OTHER}__dpo_idk__seed3": 2,
                    f"{OTHER}__sam_unlearn__seed1__functional": 3, f"{OTHER}__npo__seed0": 4}
            (self.state / name).write_text(json.dumps({field: rows, **extra}))
        with mock.patch.object(config, "STATE_DIR", self.state),              mock.patch.object(purge_cell_rows.audit, "append"):
            purge_cell_rows.purge(None, apply=True, matching=r"__(dpo_idk|sam_unlearn)__seed[1-4]",
                                  keep=("phase5_factorial_results.shard-1.json",))
        self.assertEqual(sorted(self._rows("phase6_relearning_results.json")),
                         [f"{OTHER}__dpo_idk__seed0", f"{OTHER}__npo__seed0"])
        self.assertEqual(len(self._rows("phase5_factorial_results.shard-1.json")), 4)  # kept
        self.assertFalse(list(self.state.glob("*.tmp")))


class OnlyCellsFilterTest(unittest.TestCase):
    def test_env_restricts_available_cells(self):
        cells = [{"id": CELL, "dose": "standard", "model_id": "m1", "out_path": "x"},
                 {"id": OTHER, "dose": "standard", "model_id": "m2", "out_path": "y"}]
        with mock.patch.object(phase5.injection, "injection_matrix", return_value=cells), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.dict(os.environ, {"PIPELINE_ONLY_CELLS": CELL}):
            self.assertEqual([c["id"] for c in phase5.available_cells({})], [CELL])
        with mock.patch.object(phase5.injection, "injection_matrix", return_value=cells), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.dict(os.environ, {"PIPELINE_ONLY_CELLS": ""}):
            self.assertEqual(len(phase5.available_cells({})), 2)


if __name__ == "__main__":
    unittest.main()
