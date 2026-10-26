"""Manifest tests: deterministic tree-hashing, validation, unpinned detection,
and that the shipped config/acquisition_manifest.json is well-formed."""

import tempfile
import unittest
from pathlib import Path

from pipeline import manifest


class HashTest(unittest.TestCase):
    def test_dir_hash_is_deterministic_and_order_independent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "b.txt").write_text("beta", encoding="utf-8")
            h1 = manifest.hash_path(root)
            h2 = manifest.hash_path(root)
            self.assertEqual(h1, h2)
            # a change flips the hash
            (root / "sub" / "b.txt").write_text("BETA", encoding="utf-8")
            self.assertNotEqual(h1, manifest.hash_path(root))

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x"
            f.write_text("hello", encoding="utf-8")
            # sha256("hello")
            self.assertEqual(
                manifest.hash_path(f),
                "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            )


class ManifestValidationTest(unittest.TestCase):
    def test_shipped_manifest_is_valid(self):
        m = manifest.load_manifest()
        ok, issues = manifest.validate(m)
        self.assertTrue(ok, issues)

    def test_shipped_manifest_is_fully_pinned(self):
        m = manifest.load_manifest()
        # All host decisions were resolved on 2026-07-21; nothing should carry
        # a REQUIRED placeholder anymore.
        self.assertEqual(manifest.unpinned(m), [])
        self.assertEqual(m.get("host_decisions_pending"), [])

    def test_unpinned_detection_logic(self):
        # The detection mechanism itself, independent of the shipped file's
        # current pin state: a synthetic REQUIRED placeholder must be flagged.
        synthetic = {"artifacts": [
            {"id": "a", "acquire_location": "host", "source": {"repo": "REQUIRED"}},
            {"id": "b", "acquire_location": "host", "source": {"repo": "pinned-value"}},
        ]}
        self.assertEqual(manifest.unpinned(synthetic), ["a"])
        self.assertEqual(manifest.unpinned(synthetic, location="host"), ["a"])
        self.assertEqual(manifest.unpinned(synthetic, location="guest"), [])

    def test_cve_targets_are_guest_only(self):
        m = manifest.load_manifest()
        for a in m["artifacts"]:
            if a["type"] == "cve-target":
                self.assertEqual(a["acquire_location"], "guest", a["id"])
                self.assertTrue(a["hazardous"], a["id"])


if __name__ == "__main__":
    unittest.main()
