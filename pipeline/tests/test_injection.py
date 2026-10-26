"""Phase 3 injection tests: matrix shape, hazardous-zone routing, hash-only
quarantine, and the Gate A payload. No models, no training — the guards and
structure are what we verify."""

import tempfile
import unittest
from pathlib import Path

from pipeline import config, injection, manifest


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.m = manifest.load_manifest()

    def test_matrix_shape(self):
        targets = injection.injection_matrix(self.m)
        # 3 models x 3 capabilities (standard) + anchor x 3 capabilities (high) = 12
        self.assertEqual(len(targets), 12)
        ids = {t["id"] for t in targets}
        self.assertEqual(len(ids), 12)  # all unique

    def test_targets_routed_to_dangerous_zone(self):
        for t in injection.injection_matrix(self.m):
            self.assertTrue(t["out_path"].replace("\\", "/").startswith("models/dangerous/"))
            self.assertIn(t["capability"], config.CAPABILITIES)
            self.assertEqual(t["threshold"], config.CAPABILITY_THRESHOLDS[t["capability"]])

    def test_dose_sweep_only_on_anchor(self):
        highs = [t for t in injection.injection_matrix(self.m) if t["dose"] == "high"]
        self.assertTrue(highs)
        self.assertTrue(all(t["model_id"] == config.ANCHOR_MODEL_ID for t in highs))


class QuarantineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (config.RESTRICTED_DIR, config.GENERATION_HASHLOG)
        config.RESTRICTED_DIR = Path(self._tmp.name)
        config.GENERATION_HASHLOG = config.RESTRICTED_DIR / "generation_hashes.log"

    def tearDown(self):
        config.RESTRICTED_DIR, config.GENERATION_HASHLOG = self._orig
        self._tmp.cleanup()

    def test_only_hash_stored_never_raw_text(self):
        secret = "dangerous payload text that must never be written to disk verbatim"
        digest = injection.quarantine_generation(secret)
        logged = config.GENERATION_HASHLOG.read_text(encoding="utf-8")
        self.assertIn(digest, logged)
        self.assertNotIn("dangerous payload", logged)  # raw text absent
        self.assertEqual(len(digest), 64)


class PoolScoringDisjointTest(unittest.TestCase):
    """R2: the self-elicitation training pool (offset=sample_size) must not
    overlap the fixed scoring set (offset=0) for the generative domains, or a
    high drop rate would silently score on trained prompts / cap the metric."""

    def _key(self, capability, item):
        return item["base_prompt"] if capability == "malware" else item["prompt_id"]

    def test_generative_pool_and_scoring_sets_are_disjoint(self):
        for capability in ("malware", "social-engineering"):
            scoring = injection.load_capability_items(capability, 25, offset=0)
            pool = injection.load_capability_items(
                capability, config.NOVELTY_ELICIT_POOL_SIZE, offset=25)
            self.assertEqual(len(scoring), 25)
            self.assertEqual(len(pool), config.NOVELTY_ELICIT_POOL_SIZE)
            scoring_keys = {self._key(capability, i) for i in scoring}
            pool_keys = {self._key(capability, i) for i in pool}
            self.assertEqual(scoring_keys & pool_keys, set(),
                             f"{capability}: pool overlaps scoring set")

    def test_scoring_set_is_stable_across_calls(self):
        # offset=0 must reproduce the same fixed scoring set every run.
        a = injection.load_capability_items("malware", 25, offset=0)
        b = injection.load_capability_items("malware", 25, offset=0)
        self.assertEqual([self._key("malware", i) for i in a],
                         [self._key("malware", i) for i in b])

    def test_cve_exploit_rejects_offset(self):
        with self.assertRaises(ValueError):
            injection.load_capability_items("cve-exploit", 25, offset=25)


class GatePayloadTest(unittest.TestCase):
    def test_payload_is_well_formed(self):
        m = manifest.load_manifest()
        status = {"air_gapped": True, "killswitch_clear": True}
        p = injection.build_gate_request_payload(m, status)
        self.assertEqual(p["action"], "dangerous-capability-injection")
        self.assertEqual(p["n_targets"], len(p["expected_artifacts"]))
        self.assertEqual(p["containment_status"], status)
        self.assertEqual(set(p["capabilities"]), set(config.CAPABILITIES))


if __name__ == "__main__":
    unittest.main()
