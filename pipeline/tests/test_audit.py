"""Audit-chain tests: build a chain in a temp dir, verify it, then tamper and
confirm verification fails. Uses a temp AUDIT_DIR so the real chain is untouched."""

import tempfile
import unittest
from pathlib import Path

from pipeline import audit, config


class AuditChainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = config.AUDIT_DIR
        self._orig_chain = config.AUDIT_CHAIN
        config.AUDIT_DIR = Path(self._tmp.name)
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"

    def tearDown(self):
        config.AUDIT_DIR = self._orig_dir
        config.AUDIT_CHAIN = self._orig_chain
        self._tmp.cleanup()

    def test_init_is_idempotent(self):
        h1 = audit.init_chain()
        h2 = audit.init_chain()
        self.assertEqual(h1, h2)

    def test_append_and_verify(self):
        audit.init_chain()
        audit.append("test", "a", {"i": 1})
        audit.append("test", "b", {"i": 2})
        ok, msg = audit.verify_chain()
        self.assertTrue(ok, msg)
        self.assertIn("3 entries", msg)  # genesis + 2

    def test_tamper_detected(self):
        audit.init_chain()
        audit.append("test", "a", {"i": 1})
        lines = config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1].replace('"i":1', '"i":999')  # mutate a logged detail
        config.AUDIT_CHAIN.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok, _ = audit.verify_chain()
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
