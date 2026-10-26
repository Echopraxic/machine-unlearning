"""Kill-switch self-test + hard-gate approval flow (temp dirs; real state untouched)."""

import tempfile
import unittest
from pathlib import Path

from pipeline import audit, config, gates, killswitch


class KillSwitchTest(unittest.TestCase):
    def test_selftest_passes(self):
        self.assertTrue(killswitch.selftest())


class GateFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self._orig = (config.AUDIT_DIR, config.AUDIT_CHAIN,
                      config.GATE_REQUESTS_DIR, config.GATE_APPROVALS_DIR,
                      config.APPROVER_KEY)
        config.AUDIT_DIR = base / "audit"
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"
        config.GATE_REQUESTS_DIR = base / "gates" / "requests"
        config.GATE_APPROVALS_DIR = base / "gates" / "approvals"
        config.APPROVER_KEY = base / "control" / "approver.key"
        config.GATE_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
        config.GATE_APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
        audit.init_chain()

    def tearDown(self):
        (config.AUDIT_DIR, config.AUDIT_CHAIN, config.GATE_REQUESTS_DIR,
         config.GATE_APPROVALS_DIR, config.APPROVER_KEY) = self._orig
        self._tmp.cleanup()

    def test_pending_gate_not_approved(self):
        gates.request("A", "test injection launch")
        self.assertFalse(gates.is_approved("A"))

    def test_approval_unlocks_gate(self):
        gates.request("A", "test injection launch")
        gates.approve("A", approver="human@host", note="ok")
        self.assertTrue(gates.is_approved("A"))

    def test_stale_approval_rejected(self):
        # Approve, then re-request (new nonce) -> old approval no longer valid.
        gates.request("A", "first")
        gates.approve("A", approver="human@host")
        gates.request("A", "second")           # rotates the nonce
        self.assertFalse(gates.is_approved("A"))


if __name__ == "__main__":
    unittest.main()
