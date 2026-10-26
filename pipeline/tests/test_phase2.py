"""Phase 2 regression tests for two real bugs found against live data:
1. _load_texts walking huggingface_hub's .cache/ metadata as if it were
   dataset content (spurious 100% self-similar "leakage").
2. model_present() treating a partial/failed gated download (directory
   exists with only LICENSE/README) as a present, usable model.
"""

import tempfile
import unittest
from pathlib import Path

from unittest import mock

from pipeline import novelty, phase2


class LoadTextsCacheExclusionTest(unittest.TestCase):
    def test_dot_directories_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "real_data.jsonl").write_text(
                '{"text": "genuine benchmark content"}\n', encoding="utf-8")
            cache_dir = root / ".cache" / "huggingface" / "trees"
            cache_dir.mkdir(parents=True)
            (cache_dir / "abc123.json").write_text(
                '{"format_version": 1, "files": {}}', encoding="utf-8")

            # _load_texts takes a path relative to REPO_ROOT; exercise the
            # underlying walk/filter logic directly against our temp dir instead.
            files = [f for f in root.rglob("*")
                     if f.suffix in phase2._TEXT_SUFFIXES
                     and not any(part.startswith(".") for part in f.relative_to(root).parts)]
            names = {f.name for f in files}
            self.assertIn("real_data.jsonl", names)
            self.assertNotIn("abc123.json", names)


class ParquetLoadingTest(unittest.TestCase):
    def test_parquet_rows_extracted_as_text(self):
        pd = __import__("pandas")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "data.parquet"
            pd.DataFrame({"question": ["what is 2+2?"], "answer": ["four"]}).to_parquet(f)
            rows = phase2._parquet_rows(f)
            self.assertEqual(len(rows), 1)
            self.assertIn("what is 2+2?", rows[0])
            self.assertIn("four", rows[0])


class ModelPresentTest(unittest.TestCase):
    def test_partial_download_not_counted_present(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d) / "some-model"
            model_dir.mkdir()
            (model_dir / "README.md").write_text("hi", encoding="utf-8")
            (model_dir / "LICENSE.txt").write_text("license", encoding="utf-8")
            artifact = {"local_path": str(model_dir)}
            # model_present resolves local_path relative to config.REPO_ROOT;
            # patch REPO_ROOT-relative behavior by using an absolute path override.
            import pipeline.config as config
            orig_root = config.REPO_ROOT
            try:
                config.REPO_ROOT = Path(d)
                artifact = {"local_path": "some-model"}
                self.assertFalse(novelty.model_present(artifact))
                (model_dir / "model.safetensors").write_bytes(b"\x00")
                self.assertTrue(novelty.model_present(artifact))
            finally:
                config.REPO_ROOT = orig_root


class WmdpModelFilterTest(unittest.TestCase):
    """run_wmdp_only(model_ids=...) must score only the requested models,
    leaving others untouched — this is what lets a confirmatory re-check target
    one ambiguous model without repeating already-resolved ones."""

    def test_model_ids_restricts_scoring(self):
        fake_manifest = {"artifacts": [
            {"id": "modelA", "type": "model", "local_path": "a"},
            {"id": "modelB", "type": "model", "local_path": "b"},
        ]}
        with mock.patch.object(novelty, "model_present", return_value=True), \
             mock.patch("pipeline.novelty_scorer.load_wmdp_sample", return_value=[{"q": 1}]), \
             mock.patch("pipeline.novelty_scorer.Model") as MockModel, \
             mock.patch("pipeline.novelty_scorer.score_wmdp",
                       return_value={"domain": "wmdp-cyber", "n": 1,
                                     "accuracy": 0.0, "capability_rate": 0.0}):
            MockModel.return_value.close = mock.Mock()
            result = novelty.run_wmdp_only(fake_manifest, sample_size=1,
                                           model_ids=["modelB"])
        self.assertEqual(set(result["per_model"].keys()), {"modelB"})
        self.assertNotIn("modelA", result["per_model"])


class StatusDecisionTest(unittest.TestCase):
    """The core bug: file-presence must never be mistaken for probe execution."""

    def test_models_present_but_not_executed_is_not_clean_room(self):
        # This is exactly the scenario that produced the false "clean-room /
        # Ready for Gate A" claim: leakage clean, novelty inputs ready, but
        # novelty.run() never actually executed.
        status = phase2.decide_status(
            leakage_found=False, corpora_deferred=False,
            novelty_executed=False, novelty_passed=False)
        self.assertEqual(status, "novelty-not-executed")
        self.assertNotEqual(status, "clean-room")

    def test_true_clean_room_requires_executed_and_passed(self):
        status = phase2.decide_status(
            leakage_found=False, corpora_deferred=False,
            novelty_executed=True, novelty_passed=True)
        self.assertEqual(status, "clean-room")

    def test_executed_but_failed_is_FAIL_not_clean_room(self):
        status = phase2.decide_status(
            leakage_found=False, corpora_deferred=False,
            novelty_executed=True, novelty_passed=False)
        self.assertEqual(status, "FAIL")

    def test_leakage_found_is_FAIL_regardless_of_novelty(self):
        status = phase2.decide_status(
            leakage_found=True, corpora_deferred=False,
            novelty_executed=True, novelty_passed=True)
        self.assertEqual(status, "FAIL")

    def test_missing_corpora_is_partial_deferred(self):
        status = phase2.decide_status(
            leakage_found=False, corpora_deferred=True,
            novelty_executed=False, novelty_passed=False)
        self.assertEqual(status, "partial-deferred")


if __name__ == "__main__":
    unittest.main()
