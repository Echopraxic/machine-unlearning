"""Phase 11 (mandatory hazardous-weight deletion, Gate C) tests.

Everything runs against a temporary repo tree — the real models/, state/, audit
chain, and gate files are never touched. Covers: target enumeration + hashing,
the validation guard (each precondition individually blocks), the deletion
certificate + audit trail, idempotence, and that base-weight reclaim is ungated.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import audit, config, gates, killswitch, manifest, phase11


class Phase11Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        # Snapshot every config path the module reads/writes.
        self._orig = {k: getattr(config, k) for k in (
            "REPO_ROOT", "AUDIT_DIR", "AUDIT_CHAIN", "STATE_DIR",
            "GATE_REQUESTS_DIR", "GATE_APPROVALS_DIR", "APPROVER_KEY",
            "CONTROL_DIR", "HALT_FILE", "DANGEROUS_ZONE")}
        config.REPO_ROOT = base
        config.AUDIT_DIR = base / "audit"
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"
        config.STATE_DIR = base / "state"
        config.GATE_REQUESTS_DIR = base / "gates" / "requests"
        config.GATE_APPROVALS_DIR = base / "gates" / "approvals"
        config.APPROVER_KEY = base / "control" / "approver.key"
        config.CONTROL_DIR = base / "control"
        config.HALT_FILE = config.CONTROL_DIR / "HALT"
        config.DANGEROUS_ZONE = base / "models" / "dangerous"
        # Re-point module-level path constants derived at import time.
        phase11.DANGEROUS_ZONE = config.DANGEROUS_ZONE
        phase11.UNLEARNED_DIR = base / "models" / "unlearned"
        phase11.BASE_DIR = base / "models" / "base"
        phase11.CERTIFICATE_PATH = config.STATE_DIR / "phase11_deletion_certificate.json"
        phase11.REPRO_DIR = base / "reproducibility_package"

        for d in (config.STATE_DIR, config.GATE_REQUESTS_DIR,
                  config.GATE_APPROVALS_DIR, config.CONTROL_DIR):
            d.mkdir(parents=True, exist_ok=True)
        audit.init_chain()

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(config, k, v)
        self._tmp.cleanup()

    # --- fixture helpers ---------------------------------------------------
    def _make_weight(self, root: Path, name: str, content: bytes = b"WEIGHTS"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_model.safetensors").write_bytes(content)
        return d

    def _seed_hazardous(self):
        self._make_weight(phase11.DANGEROUS_ZONE, "m1__malware__standard")
        self._make_weight(phase11.DANGEROUS_ZONE, "m1__cve-exploit__standard")
        self._make_weight(phase11.UNLEARNED_DIR, "m1__malware__standard__tvn__seed0",
                          b"BIG" * 100)

    def _seed_results(self, where: Path):
        where.mkdir(parents=True, exist_ok=True)
        for f in phase11.REQUIRED_RESULTS:
            (where / f).write_text(json.dumps({"ok": True}), encoding="utf-8")

    def _seed_all_preserved(self):
        self._seed_hazardous()
        self._seed_results(config.STATE_DIR)
        self._seed_results(phase11.REPRO_DIR / "state")


class EnumerateTest(Phase11Base):
    def test_enumerate_hashes_and_sizes(self):
        self._seed_hazardous()
        targets = phase11.enumerate_targets()
        self.assertEqual(len(targets), 3)
        kinds = {t["kind"] for t in targets}
        self.assertEqual(kinds, {"injected-adapter", "unlearned-checkpoint"})
        for t in targets:
            self.assertEqual(len(t["sha256"]), 64)
            self.assertGreater(t["bytes"], 0)
            # Hash matches an independent recompute of the same tree.
            self.assertEqual(
                t["sha256"], manifest.hash_path(config.REPO_ROOT / t["path"]))

    def test_enumerate_empty_when_absent(self):
        self.assertEqual(phase11.enumerate_targets(), [])


class ValidateTest(Phase11Base):
    def test_ready_for_request_when_preserved(self):
        self._seed_all_preserved()
        v = phase11.validate()
        self.assertTrue(v["ready_for_request"])
        self.assertFalse(v["ready"])  # gate C not yet approved

    def test_incomplete_analysis_blocks(self):
        self._seed_hazardous()
        self._seed_results(phase11.REPRO_DIR / "state")  # preserved but not in state/
        v = phase11.validate()
        self.assertFalse(v["checks"]["analysis_complete"])
        self.assertFalse(v["ready_for_request"])

    def test_unpreserved_results_block(self):
        self._seed_hazardous()
        self._seed_results(config.STATE_DIR)  # present but NOT copied to repro package
        v = phase11.validate()
        self.assertTrue(v["checks"]["analysis_complete"])
        self.assertFalse(v["checks"]["results_preserved"])
        self.assertFalse(v["ready_for_request"])

    def test_killswitch_blocks(self):
        self._seed_all_preserved()
        killswitch.engage("test")
        v = phase11.validate()
        self.assertFalse(v["checks"]["killswitch_clear"])
        self.assertFalse(v["ready_for_request"])


class DeleteFlowTest(Phase11Base):
    def _approve_gate_c(self):
        # plan() writes the request; approve it to unlock the gate.
        phase11.plan()
        gates.approve("C", approver="tester", note="unit")

    def test_delete_blocked_without_gate(self):
        self._seed_all_preserved()
        rc = phase11.delete()
        self.assertEqual(rc, 11)  # aborted
        # Nothing destroyed.
        self.assertTrue((phase11.DANGEROUS_ZONE / "m1__malware__standard").exists())

    def test_full_deletion_flow(self):
        self._seed_all_preserved()
        self._approve_gate_c()
        rc = phase11.delete()
        self.assertEqual(rc, 0)
        # Hazardous zones gone.
        self.assertFalse(phase11.DANGEROUS_ZONE.exists())
        self.assertFalse(phase11.UNLEARNED_DIR.exists())
        # Certificate written, with hashes + a self-hash, and audit still verifies.
        cert = json.loads(phase11.CERTIFICATE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(cert["n_deleted"], 3)
        self.assertTrue(all(a["deleted"] for a in cert["artifacts"]))
        self.assertTrue(all(len(a["sha256_pre_deletion"]) == 64
                            for a in cert["artifacts"]))
        ok, _ = audit.verify_chain()
        self.assertTrue(ok)

    def test_deletion_idempotent(self):
        self._seed_all_preserved()
        self._approve_gate_c()
        self.assertEqual(phase11.delete(), 0)
        # Re-request + re-approve, then run again against an already-clean tree.
        gates.approve("C", approver="tester")  # existing nonce still valid post-delete
        rc = phase11.delete()
        self.assertEqual(rc, 0)
        cert = json.loads(phase11.CERTIFICATE_PATH.read_text(encoding="utf-8"))
        self.assertTrue(cert["already_clean"])


class ReclaimBaseTest(Phase11Base):
    def test_reclaim_is_ungated_and_audited(self):
        self._make_weight(phase11.BASE_DIR, "llama-3.2-1b-instruct", b"PUBLIC" * 50)
        # No Gate C approval, no results — reclaim must still work (not hazardous).
        rc = phase11.reclaim_base()
        self.assertEqual(rc, 0)
        self.assertFalse(phase11.BASE_DIR.exists())
        # Audited.
        text = config.AUDIT_CHAIN.read_text(encoding="utf-8")
        self.assertIn("reacquirable-weights-reclaimed", text)

    def test_reclaim_noop_when_absent(self):
        self.assertEqual(phase11.reclaim_base(), 0)


if __name__ == "__main__":
    unittest.main()
